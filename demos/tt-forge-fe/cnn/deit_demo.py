# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Deit Demo Script

import forge
# Add repository root to path to locate third_party modules
import sys
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.deit.pytorch import ModelLoader, ModelVariant
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch


def run_deit_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs.pixel_values], compiler_cfg=compiler_cfg)

    # Run inference on Tenstorrent device
    output = compiled_model(inputs.pixel_values)

    # Post-process the output
    loader.post_processing(output, model)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.BASE,
        ModelVariant.BASE_DISTILLED,
        ModelVariant.SMALL,
        ModelVariant.TINY,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_deit_demo_case(variant)
