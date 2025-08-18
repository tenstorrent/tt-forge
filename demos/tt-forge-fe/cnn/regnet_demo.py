# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Regnet Demo Script

import sys
import forge
from third_party.tt_forge_models.regnet.pytorch import ModelLoader, ModelVariant


def run_regnet_demo_case(variant):

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
        ModelVariant.Y_040,
        ModelVariant.Y_064,
        ModelVariant.Y_080,
        ModelVariant.Y_120,
        ModelVariant.Y_160,
        ModelVariant.Y_320,
        # Torchvision variants
        ModelVariant.Y_400MF,
        ModelVariant.Y_800MF,
        ModelVariant.Y_1_6GF,
        ModelVariant.Y_3_2GF,
        ModelVariant.Y_8GF,
        ModelVariant.Y_16GF,
        ModelVariant.Y_32GF,
        ModelVariant.Y_128GF,
        ModelVariant.X_400MF,
        ModelVariant.X_800MF,
        ModelVariant.X_1_6GF,
        ModelVariant.X_3_2GF,
        ModelVariant.X_8GF,
        ModelVariant.X_16GF,
        ModelVariant.X_32GF,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_regnet_demo_case(variant)
