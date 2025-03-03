# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet Demo Script

import os

import forge
import requests
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification


def run_resnet_pytorch(variant="microsoft/resnet-50", batch_size=1):

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model_ckpt = variant
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    model = ResNetForImageClassification.from_pretrained(model_ckpt)

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = ["tiger"] * batch_size

    # Data preprocessing
    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = [inputs["pixel_values"]] * batch_size
    batch_input = torch.cat(pixel_values, dim=0)

    # Run inference on Tenstorrent device
    framework_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, batch_input)

    output = compiled_model(batch_input)
    # Data postprocessing
    predicted_value = output[0].argmax(-1)
    predicted_label = [model.config.id2label[pred.item()] for pred in predicted_value]

    for sample in range(batch_size):
        print(f"True Label: {label[sample]} | Predicted Label: {predicted_label[sample]}")


if __name__ == "__main__":
    run_resnet_pytorch()