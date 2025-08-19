# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# VoVNet Demo Script

import sys
import forge
from third_party.tt_forge_models.vovnet.pytorch import ModelLoader, ModelVariant


def run_vovnet_demo_case(variant):

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
        # OSMR variants
        ModelVariant.VOVNET27S,
        ModelVariant.VOVNET39,
        ModelVariant.VOVNET57,
        # TorchHub variants
        ModelVariant.VOVNET39_TORCHHUB,
        ModelVariant.VOVNET57_TORCHHUB,
        # TIMM variants
        ModelVariant.TIMM_VOVNET19B_DW,
        ModelVariant.TIMM_VOVNET39B,
        ModelVariant.TIMM_VOVNET99B,
        ModelVariant.TIMM_VOVNET19B_DW_RAIN1K,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_vovnet_demo_case(variant)
