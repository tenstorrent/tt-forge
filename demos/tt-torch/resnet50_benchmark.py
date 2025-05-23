# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import requests
from PIL import Image
import transformers
from transformers import AutoFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
import os
import glob
import torchvision.models as models
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.tools.device_manager import DeviceManager
import ast
import warnings

warnings.filterwarnings("ignore")
import time

# Load DeiT model and processor
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights).to(torch.bfloat16).eval()
image_processor = weights.transforms()

# Compile model for Tenstorrent backend
cc = CompilerConfig()
cc.enable_consteval = True
cc.consteval_parameters = True
options = BackendOptions()
options.compiler_config = cc
tt_model = torch.compile(model, backend=backend, dynamic=False, options=options)
output = tt_model(torch.zeros(1, 3, 224, 224).to(torch.bfloat16))
print("Model converted successfully.")


def get_imagenet_label_dict():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(dir_path, "imagenet_class_labels.txt")
    with open(path, "r") as file:
        class_labels = ast.literal_eval(file.read())
    return class_labels


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_input(image_path):
    img = Image.open(image_path)
    return img


def get_label(image_path):
    _, image_name = image_path.rsplit("/", 1)
    image_name_exact, _ = image_name.rsplit(".", 1)
    _, label_id = image_name_exact.rsplit("_", 1)
    label = list(IMAGENET2012_CLASSES).index(label_id)
    return label


def get_data_loader(input_loc, batch_size, iterations):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = glob.glob(data_path)

    def loader():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=get_input(f1),
                    label=get_label(f1),
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    def loader_hf():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=f1["image"],
                    label=f1["label"],
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    if len(files) == 0:
        files_raw = iter(load_dataset("imagenet-1k", split="validation", streaming=True))
        files = []
        sample_count = batch_size * iterations
        for _ in range(sample_count):
            files.append(next(files_raw))
        del files_raw
        return loader_hf()

    return loader()


def get_batch(data_loader, image_processor):
    loaded_images = next(data_loader)
    images = None
    labels = []
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")
        img = image_processor(img)

        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels


def download_and_preprocess_image(url):
    try:
        image = Image.open(requests.get(url, stream=True, timeout=10).raw).convert("RGB")
        inputs = image_processor(image, return_tensors="pt")
        return inputs["pixel_values"][0]
    except Exception as e:
        print(f"Failed to download/process image: {url}. Error: {e}")
        return None


def main():
    # Load validation set metadata (URLs and labels only)
    # dataset = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
    class_names = get_imagenet_label_dict()
    correct = 0
    total = 0
    print("Running Resnet50 ImageNet benchmark on 100 validation images (downloaded at runtime)...")
    count = 0
    imagenet_label_dict = get_imagenet_label_dict()
    batch_size = 1
    iterations = 100
    data_loader = get_data_loader("ImageNet_data", batch_size, iterations)
    for iter in range(iterations):
        predictions = []
        torch_pixel_values, labels = get_batch(data_loader, image_processor)
        torch_pixel_values = torch_pixel_values.to(torch.bfloat16).unsqueeze(0)
        with torch.no_grad():
            start_time = time.time()
            outputs = tt_model(torch_pixel_values)
            end_time = time.time()
            inference_time = end_time - start_time
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred = logits.argmax(-1).item()
            predictions.append(pred)
        for i in range(len(predictions)):
            pred = predictions[i]
            label = labels[i]
            pred_name = class_names[pred]
            label_name = class_names[label]
            is_correct = pred == label
            correct += int(is_correct)
            total += 1
            fps = 1 / inference_time if inference_time > 0 else 0
            print(
                f"Image {total:03d}: Label = {label_name}, Prediction = {pred_name}, Correct = {is_correct}, Inference Time = {inference_time:.4f} seconds, FPS = {fps:.2f}"
            )
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\nTop-1 Accuracy on {total} ImageNet val images: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
