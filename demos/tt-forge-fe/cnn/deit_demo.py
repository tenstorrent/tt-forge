# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Deit Demo Script

import forge
from third_party.tt_forge_models.deit.pytorch import ModelLoader, ModelVariant


def run_deit_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs.pixel_values])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs.pixel_values)

    # Post-process the output
    loader.post_processing(output, model)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.BASE,
        ModelVariant.BASE_DISTILLED,
        ModelVariant.SMALL,
        ModelVariant.TINY,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_deit_demo_case(variant)
