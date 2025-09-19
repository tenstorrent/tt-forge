# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import ResNetForImageClassification, AutoImageProcessor
from PIL import Image
import requests


def print_top_predictions(logits, model, top_k=5):
    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, top_k)

    # Move to CPU and convert to appropriate types to avoid runtime warnings
    top5_prob_cpu = top5_prob.cpu().float()
    top5_idx_cpu = top5_idx.cpu().int()

    print("🏆 Top 5 Predictions:")
    print("-" * 40)

    # Use the model's label dictionary
    label_dict = model.config.id2label

    for i in range(top_k):
        class_idx = top5_idx_cpu[0][i].item()
        prob = top5_prob_cpu[0][i].item()
        class_name = label_dict.get(class_idx, f"Unknown Class {class_idx}")
        print(f"{i+1}. {class_name:<35} ({prob:.2%})")


def main():
    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load the model and processor
    model_name = "microsoft/resnet-50"
    model = ResNetForImageClassification.from_pretrained(model_name)
    model = model.eval()
    processor = AutoImageProcessor.from_pretrained(model_name)

    # Load the inputs
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Convert to bfloat16
    model = model.to(torch.bfloat16)
    inputs = inputs.to(torch.bfloat16)

    def attempt_to_device(x):
        if hasattr(x, "to"):
            return x.to(device)
        return x

    # Move model and inputs to the TT device
    device = xm.xla_device()
    model = model.to(device)
    inputs = tree_map(attempt_to_device, inputs)

    # Run the model and print top predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        print_top_predictions(logits, model, top_k=5)


if __name__ == "__main__":
    main()
