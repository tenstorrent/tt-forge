# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Dla Demo Script

import forge
from third_party.tt_forge_models.dla.pytorch import ModelLoader, ModelVariant
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch


def run_dla_demo_case(variant):

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
        ModelVariant.DLA34,
        ModelVariant.DLA46_C,
        ModelVariant.DLA46X_C,
        ModelVariant.DLA60,
        ModelVariant.DLA60X,
        ModelVariant.DLA60X_C,
        ModelVariant.DLA102,
        ModelVariant.DLA102X,
        ModelVariant.DLA102X2,
        ModelVariant.DLA169,
        # TIMM variants
        ModelVariant.DLA34_IN1K,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_dla_demo_case(variant)
