# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MobileNetV2 Demo Script

import os
import torch
import onnx

import forge
from third_party.tt_forge_models.mobilenetv2.pytorch import ModelLoader, ModelVariant


def run_mobilenetv2_onnx_case(variant, opset_version=17):
    """
    Run a MobileNetV2 model using ONNX and the Forge compiler.

    Args:
        variant: ModelVariant enum specifying which variant to use
        opset_version (int): The ONNX opset version to use
    """
    print(f"Running MobileNetV2 ONNX demo with variant: {variant}")

    # Load Model and inputs using the ModelLoader
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    input_tensor = loader.load_inputs()

    # Create a temporary path for the ONNX model
    # Use variant name to create unique filename
    # Get the module name from the loader
    variant_name = variant.value.replace("/", "_").replace("-", "_")
    onnx_path = f"mobilenetv2_{variant_name}.onnx"
    module_name = loader._get_model_info().name.replace("pytorch", "onnx")

    # Export model to ONNX
    print(f"Exporting model to ONNX with opset version {opset_version}...")
    torch.onnx.export(model, input_tensor, onnx_path, opset_version=opset_version)

    # Load ONNX model
    print("Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # Compile model using Forge
    print("Compiling model using Forge...")
    compiled_model = forge.compile(onnx_model, [input_tensor], module_name=module_name)

    # Run inference on Tenstorrent device
    print("Running inference...")
    output = compiled_model(input_tensor)

    # Post-process and display results using the loader's method
    loader.print_cls_results(output)

    # Clean up temporary ONNX file
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        print(f"Removed temporary ONNX file: {onnx_path}")

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # HuggingFace variants
        ModelVariant.MOBILENET_V2_075_160_HF,
        ModelVariant.MOBILENET_V2_100_224_HF,
        # TIMM variants
        ModelVariant.MOBILENET_V2_100_TIMM,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_mobilenetv2_onnx_case(variant)
