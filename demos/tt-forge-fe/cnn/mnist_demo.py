# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MNIST Demo Script

import forge
# Add repository root to path to locate third_party modules
import sys
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.mnist.pytorch import ModelLoader
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch

# Load model and input
loader = ModelLoader()
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
