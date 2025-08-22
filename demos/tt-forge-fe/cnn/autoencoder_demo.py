# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Autoencoder Demo Script

import forge
from third_party.tt_forge_models.autoencoder.pytorch import ModelLoader, ModelVariant
import shutil

SAVE_DIR = "demos/tt-forge-fe/cnn/autoencoder"


def run_autoencoder_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process the output
    loader.post_processing(output, save_path=SAVE_DIR)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.LINEAR,
    ]

    try:
        for variant in demo_cases:
            run_autoencoder_demo_case(variant)
    finally:
        shutil.rmtree(SAVE_DIR, ignore_errors=True)
