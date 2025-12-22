# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Alexnet Demo Script

import forge
# Add repository root to path to locate third_party modules
import sys
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.alexnet.pytorch import ModelLoader, ModelVariant
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch


def run_alexnet_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs], compiler_cfg=compiler_cfg)

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process the output
    loader.print_cls_results(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.ALEXNET_TORCH_HUB,
        ModelVariant.ALEXNET_OSMR_B,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_alexnet_demo_case(variant)
