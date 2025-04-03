#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet ONNX Demo Script

import os
import torch
import onnx
import requests
import numpy as np
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification

import forge
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker


def run_resnet_onnx(variant="microsoft/resnet-50", batch_size=1, opset_version=11):
    """
    Run a ResNet model using ONNX and the Forge compiler.

    Args:
        variant (str): The HuggingFace model variant to use
        batch_size (int): The batch size for inference
        opset_version (int): The ONNX opset version to use
    """
    print(f"Running ResNet ONNX demo with variant: {variant}, batch size: {batch_size}")

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    model_ckpt = variant
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
    torch_model = ResNetForImageClassification.from_pretrained(model_ckpt)

    # Create a temporary path for the ONNX model
    onnx_path = "resnet50.onnx"

    # Load data sample
    url = "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    label = ["tiger"] * batch_size

    # Data preprocessing
    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = [inputs["pixel_values"]] * batch_size
    batch_input = torch.cat(pixel_values, dim=0)

    # Export model to ONNX
    print(f"Exporting model to ONNX with opset version {opset_version}...")
    torch.onnx.export(
        torch_model,
        batch_input,
        onnx_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Load ONNX model
    print("Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Create a module name for the model
    module_name = f"onnx_resnet50_{variant.split('/')[-1]}"

    # Create framework model
    framework_model = forge.OnnxModule(module_name, onnx_model)

    # Prepare input sample for compilation
    input_sample = [batch_input]

    # Compile model using Forge
    print("Compiling model using Forge...")
    compiled_model = forge.compile(onnx_model, input_sample)

    # Run inference
    print("Running inference...")
    output = compiled_model(batch_input)

    # Data postprocessing
    predicted_value = output[0].argmax(-1)
    predicted_label = [torch_model.config.id2label[pred.item()] for pred in predicted_value]

    # Print results
    print("\nResults:")
    for sample in range(batch_size):
        print(f"True Label: {label[sample]} | Predicted Label: {predicted_label[sample]}")

    # Verify results
    print("\nVerifying results...")
    verify(
        input_sample,
        framework_model,
        compiled_model,
        VerifyConfig(value_checker=AutomaticValueChecker(pcc=0.95)),
    )

    print("Verification complete!")

    # Clean up
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"Removed temporary ONNX file: {onnx_path}")


if __name__ == "__main__":
    run_resnet_onnx()
