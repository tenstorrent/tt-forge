# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MobileNetV2 Demo Script

import sys
import forge
from third_party.tt_forge_models.mobilenetv2.pytorch import ModelLoader, ModelVariant


def run_mobilenetv2_demo_case(variant):

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process and display results
    loader.print_cls_results(output)

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
        run_mobilenetv2_demo_case(variant)
