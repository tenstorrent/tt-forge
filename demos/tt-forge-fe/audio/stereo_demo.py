# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from third_party.tt_forge_models.stereo.pytorch import ModelLoader, ModelVariant

import forge
from forge.verify.verify import verify


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "decoder_input_ids": decoder_input_ids}
        output = self.model(**inputs)
        return output.logits


# Load model and inputs
loader = ModelLoader(variant=ModelVariant.SMALL)
model = loader.load_model()
framework_model = Wrapper(model)
inputs_dict = loader.load_inputs()
inputs = [inputs_dict["input_ids"], inputs_dict["attention_mask"], inputs_dict["decoder_input_ids"]]

# Forge compile framework model
compiled_model = forge.compile(framework_model, sample_inputs=inputs)

# Model Verification
verify(inputs, framework_model, compiled_model)
