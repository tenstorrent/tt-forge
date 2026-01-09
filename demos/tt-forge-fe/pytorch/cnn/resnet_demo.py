# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet Demo Script

import sys
import forge
from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant
from datasets import load_dataset
import random
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch


def run_resnet_demo_case(variant):

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

    # Post-process
    loader.output_postprocess(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # HuggingFace variant
        ModelVariant.RESNET_50_HF,
        # TIMM variant
        ModelVariant.RESNET_50_TIMM,
        # Torchvision variants
        ModelVariant.RESNET_18,
        ModelVariant.RESNET_34,
        ModelVariant.RESNET_50,
        ModelVariant.RESNET_101,
        ModelVariant.RESNET_152,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_resnet_demo_case(variant)
