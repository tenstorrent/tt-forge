# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# YOLOX Demo Script

import sys
import forge
from third_party.tt_forge_models.yolox.pytorch import ModelLoader, ModelVariant
import torch
from forge._C import DataFormat
from forge.config import CompilerConfig


def run_yolox_demo_case(variant):

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
    loader.post_processing(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.YOLOX_NANO,
        ModelVariant.YOLOX_TINY,
        ModelVariant.YOLOX_S,
        ModelVariant.YOLOX_M,
        ModelVariant.YOLOX_L,
        ModelVariant.YOLOX_DARKNET,
        ModelVariant.YOLOX_X,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_yolox_demo_case(variant)
