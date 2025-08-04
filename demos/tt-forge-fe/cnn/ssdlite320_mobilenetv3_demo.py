# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
from third_party.tt_forge_models.ssdlite320_mobilenetv3.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

import forge
from forge._C import DataFormat
from forge.config import CompilerConfig
from forge.verify.verify import verify


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_batch):
        outputs = self.model(input_batch)
        outputs = [outputs[0]["boxes"], outputs[0]["labels"], outputs[0]["scores"]]
        return outputs


# Load model and input using the new loader
loader = ModelLoader(variant=ModelVariant.SSDLITE320_MOBILENET_V3_LARGE)

# Load model with bfloat16 override
framework_model = loader.load_model(dtype_override=torch.bfloat16)
framework_model = Wrapper(framework_model)

# Load inputs with bfloat16 override
inputs = loader.load_inputs(dtype_override=torch.bfloat16)

data_format_override = DataFormat.Float16_b
compiler_cfg = CompilerConfig(default_df_override=data_format_override)

# Forge compile framework model
compiled_model = forge.compile(
    framework_model,
    sample_inputs=[inputs],
    compiler_cfg=compiler_cfg,
)

# Model Verification
verify([inputs], framework_model, compiled_model)
