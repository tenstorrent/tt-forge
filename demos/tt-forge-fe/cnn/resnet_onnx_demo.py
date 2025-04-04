#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet ONNX Demo Script

import os
import torch
import onnx
import random
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoFeatureExtractor, ResNetForImageClassification

import forge


def run_resnet_onnx(variant="microsoft/resnet-50", batch_size=1, opset_version=17):
    """
    Run a ResNet model using ONNX and the Forge compiler.

    Args:
        variant (str): The HuggingFace model variant to use
        batch_size (int): The batch size for inference
        opset_version (int): The ONNX opset version to use
    """
    print(f"Running ResNet ONNX demo with variant: {variant}, batch size: {batch_size}")

    # Load ResNet feature extractor and model checkpoint from HuggingFace
    feature_extractor = AutoFeatureExtractor.from_pretrained(variant)
    torch_model = ResNetForImageClassification.from_pretrained(variant)

    # Create a temporary path for the ONNX model
    onnx_path = "resnet50.onnx"

    # Load data samples from a dataset
    print("Loading dataset...")
    dataset = load_dataset("zh-plus/tiny-imagenet")
    
    # Select random images for the batch
    sample_indices = random.sample(range(len(dataset["valid"])), batch_size)
    images = [dataset["valid"][i]["image"] for i in sample_indices]
    labels = [dataset["valid"][i]["label"] for i in sample_indices]
    
    # Get class names for the labels
    class_names = dataset["valid"].features["label"].names
    label_names = [class_names[label] for label in labels]
    
    # Data preprocessing
    print("Preprocessing images...")
    processed_images = []
    for img in images:
        inputs = feature_extractor(img, return_tensors="pt")
        processed_images.append(inputs["pixel_values"])
    
    batch_input = torch.cat(processed_images, dim=0)

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

    # Create framework model
    framework_model = forge.OnnxModule(f"onnx_resnet50_{variant.split('/')[-1]}", onnx_model)

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
        print(f"True Label: {label_names[sample]} | Predicted Label: {predicted_label[sample]}")



    # Clean up
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"Removed temporary ONNX file: {onnx_path}")


if __name__ == "__main__":
    run_resnet_onnx()
