# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# VGG Demo Script

import sys
import forge
from third_party.tt_forge_models.vgg.pytorch import ModelLoader, ModelVariant
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch


def run_vgg_demo_case(variant):

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
        # OSMR variants
        ModelVariant.VGG11,
        ModelVariant.VGG13,
        ModelVariant.VGG16,
        ModelVariant.VGG19,
        ModelVariant.VGG19_BN_OSMR,
        ModelVariant.VGG19_BNB_OSMR,
        # TorchHub variant
        ModelVariant.VGG19_BN,
        # TIMM variant
        ModelVariant.TIMM_VGG19_BN,
        # Torchvision variants
        ModelVariant.TV_VGG11,
        ModelVariant.TV_VGG11_BN,
        ModelVariant.TV_VGG13,
        ModelVariant.TV_VGG13_BN,
        ModelVariant.TV_VGG16,
        ModelVariant.TV_VGG16_BN,
        ModelVariant.TV_VGG19,
        # HuggingFace variant
        ModelVariant.HF_VGG19,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_vgg_demo_case(variant)
