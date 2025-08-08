# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import os

os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
import forge
from third_party.tt_forge_models.hippynn.pytorch import ModelLoader as AtomicModelLoader
from third_party.tt_forge_models.hippynn.pytorch import (
    ModelVariant as AtomicModelVariant,
)
from forge.verify.verify import verify


class HippynWrapper(torch.nn.Module):
    def __init__(self, model, output_key):
        super().__init__()
        self.model = model
        self.output_key = output_key

    def forward(self, species: torch.Tensor, positions: torch.Tensor):
        input_dict = {"Z": species, "R": positions}
        output_dict = self.model(*input_dict.values())
        output_dict = list(output_dict)
        return output_dict


# Load Models
loader = AtomicModelLoader()
framework_model, output_key = loader.load_model()
framework_model = HippynWrapper(framework_model, output_key=output_key)

# Load inputs
inputs = loader.load_inputs()

# Forge compile framework model
compiled_model = forge.compile(
    framework_model,
    sample_inputs=inputs,
)
# Model Verification
verify(inputs, framework_model, compiled_model)
