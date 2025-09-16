# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNext Demo Script

import sys
import forge
from third_party.tt_forge_models.resnext.pytorch import ModelLoader, ModelVariant
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch


def run_resnext_demo_case(variant):

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
        # TORCH_HUB variants
        ModelVariant.RESNEXT50_32X4D,
        ModelVariant.RESNEXT101_32X8D,
        ModelVariant.RESNEXT101_32X8D_WSL,
        # OSMR variants
        ModelVariant.RESNEXT14_32X4D_OSMR,
        ModelVariant.RESNEXT26_32X4D_OSMR,
        ModelVariant.RESNEXT50_32X4D_OSMR,
        ModelVariant.RESNEXT101_64X4D_OSMR,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_resnext_demo_case(variant)
