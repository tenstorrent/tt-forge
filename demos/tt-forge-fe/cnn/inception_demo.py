# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Inception Demo Script

import sys
import forge
from third_party.tt_forge_models.inception.pytorch import ModelLoader, ModelVariant


def run_inception_demo_case(variant):

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
        # TIMM variants
        ModelVariant.INCEPTION_V4,
        ModelVariant.INCEPTION_V4_TF_IN1K,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_inception_demo_case(variant)
