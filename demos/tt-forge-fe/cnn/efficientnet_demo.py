# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# EfficientNet Demo Script

import sys
import forge
from third_party.tt_forge_models.efficientnet.pytorch import ModelLoader, ModelVariant
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch


def run_efficientnet_demo_case(variant):

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs], compiler_cfg=compiler_cfg)

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process and display results
    loader.print_cls_results(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Torchvision variants
        ModelVariant.B0,
        ModelVariant.B1,
        ModelVariant.B2,
        ModelVariant.B3,
        ModelVariant.B4,
        ModelVariant.B5,
        ModelVariant.B6,
        ModelVariant.B7,
        # TIMM variants
        ModelVariant.TIMM_EFFICIENTNET_B0,
        ModelVariant.TIMM_EFFICIENTNET_B4,
        ModelVariant.HF_TIMM_EFFICIENTNET_B0_RA_IN1K,
        ModelVariant.HF_TIMM_EFFICIENTNET_B4_RA2_IN1K,
        ModelVariant.HF_TIMM_EFFICIENTNET_B5_IN12K_FT_IN1K,
        ModelVariant.HF_TIMM_TF_EFFICIENTNET_B0_AA_IN1K,
        ModelVariant.HF_TIMM_EFFICIENTNETV2_RW_S_RA2_IN1K,
        ModelVariant.HF_TIMM_TF_EFFICIENTNETV2_S_IN21K,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_efficientnet_demo_case(variant)
