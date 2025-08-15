# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from third_party.tt_forge_models.rmbg.pytorch import ModelLoader, ModelVariant

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.verify.verify import verify

framework_model = loader.load_model(dtype_override=torch.bfloat16)
inputs = loader.load_inputs(dtype_override=torch.bfloat16)
inputs = [inputs]

data_format_override = DataFormat.Float16_b
compiler_cfg = CompilerConfig(default_df_override=data_format_override)

# Forge compile framework model
compiled_model = forge.compile(
    framework_model,
    sample_inputs=inputs,
    compiler_cfg=compiler_cfg,
)

# Model Verification
verify(inputs, framework_model, compiled_model)
