# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# VIT Demo Script

import sys
import forge
from third_party.tt_forge_models.vit.pytorch import ModelLoader, ModelVariant


def run_vit_demo_case(variant):

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
        # HuggingFace variants
        ModelVariant.BASE,
        ModelVariant.LARGE,
        # Torchvision variants
        ModelVariant.VIT_B_16,
        ModelVariant.VIT_B_32,
        ModelVariant.VIT_L_16,
        ModelVariant.VIT_L_32,
        ModelVariant.VIT_H_14,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_vit_demo_case(variant)
