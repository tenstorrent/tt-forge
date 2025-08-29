# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Wideresnet Demo Script

import forge
from third_party.tt_forge_models.wide_resnet.pytorch import ModelLoader, ModelVariant


def run_wideresnet_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process the output
    loader.post_processing(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.WIDE_RESNET50_2,
        ModelVariant.WIDE_RESNET101_2,
        ModelVariant.TIMM_WIDE_RESNET50_2,
        ModelVariant.TIMM_WIDE_RESNET101_2,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_wideresnet_demo_case(variant)
