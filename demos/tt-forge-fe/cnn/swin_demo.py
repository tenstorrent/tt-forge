# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Swin Demo Script

import sys
import forge
from third_party.tt_forge_models.swin.image_classification.pytorch import ModelLoader, ModelVariant


def run_swin_demo_case(variant):

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
        ModelVariant.SWIN_TINY_HF,
        # Torchvision variants - Swin V1
        ModelVariant.SWIN_T,
        ModelVariant.SWIN_S,
        ModelVariant.SWIN_B,
        # Torchvision variants - Swin V2
        ModelVariant.SWIN_V2_T,
        ModelVariant.SWIN_V2_S,
        ModelVariant.SWIN_V2_B,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_swin_demo_case(variant)
