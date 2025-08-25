# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# EfficientNet Demo Script

import sys
import forge
from third_party.tt_forge_models.efficientnet_lite.pytorch import ModelLoader, ModelVariant


def run_efficientnet_lite_demo_case(variant):

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
        # Torchvision variants
        ModelVariant.TF_EFFICIENTNET_LITE0_IN1K,
        ModelVariant.TF_EFFICIENTNET_LITE1_IN1K,
        ModelVariant.TF_EFFICIENTNET_LITE2_IN1K,
        ModelVariant.TF_EFFICIENTNET_LITE3_IN1K,
        ModelVariant.TF_EFFICIENTNET_LITE4_IN1K,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_efficientnet_lite_demo_case(variant)
