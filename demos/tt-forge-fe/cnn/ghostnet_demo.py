# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Ghostnet Demo Script

import sys
import forge
from third_party.tt_forge_models.ghostnet.pytorch import ModelLoader, ModelVariant


def run_ghostnet_demo_case(variant):

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process and display results
    loader.post_processing(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # TIMM variants
        ModelVariant.GHOSTNET_100,
        ModelVariant.GHOSTNET_100_IN1K,
        ModelVariant.GHOSTNETV2_100_IN1K,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_ghostnet_demo_case(variant)
