# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# HRNet Demo Script

import sys
import forge
from third_party.tt_forge_models.hrnet.pytorch import ModelLoader, ModelVariant


def run_hrnet_demo_case(variant):

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
        ModelVariant.HRNET_W18_SMALL,
        ModelVariant.HRNET_W18_SMALL_V2,
        ModelVariant.HRNET_W18,
        ModelVariant.HRNET_W30,
        # OSMR variants
        ModelVariant.HRNET_W18_SMALL_V1_OSMR,
        ModelVariant.HRNET_W18_SMALL_V2_OSMR,
        ModelVariant.HRNETV2_W18_OSMR,
        ModelVariant.HRNETV2_W30_OSMR,
        ModelVariant.HRNETV2_W32_OSMR,
        ModelVariant.HRNETV2_W40_OSMR,
        ModelVariant.HRNETV2_W44_OSMR,
        ModelVariant.HRNETV2_W48_OSMR,
        ModelVariant.HRNETV2_W64_OSMR,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_hrnet_demo_case(variant)
