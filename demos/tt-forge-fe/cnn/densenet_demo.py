# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Densenet Demo Script

import forge
from third_party.tt_forge_models.densenet.pytorch import ModelLoader, ModelVariant


def run_densenet_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process the output
    loader.post_process(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.DENSENET121,
        ModelVariant.DENSENET161,
        ModelVariant.DENSENET169,
        ModelVariant.DENSENET201,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_densenet_demo_case(variant)
