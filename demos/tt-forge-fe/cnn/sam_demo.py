# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from third_party.tt_forge_models.sam.pytorch import ModelLoader, ModelVariant

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.verify.verify import verify


class SamWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, input_points):
        return self.model(pixel_values=pixel_values, input_points=input_points).pred_masks.cpu()


# Load Model and Inputs
loader = ModelLoader(variant=ModelVariant.BASE)
framework_model = loader.load_model(dtype_override=torch.bfloat16)
framework_model = SamWrapper(framework_model)
pixel_values, input_points = loader.load_inputs(dtype_override=torch.bfloat16)
sample_inputs = [pixel_values, input_points]

data_format_override = DataFormat.Float16_b
compiler_cfg = CompilerConfig(default_df_override=data_format_override)

compiled_model = forge.compile(
    framework_model,
    sample_inputs=sample_inputs,
    compiler_cfg=compiler_cfg,
)

# Model Verification
verify(sample_inputs, framework_model, compiled_model)
