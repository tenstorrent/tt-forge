# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# FLAN-T5  Demo Script

import forge
from third_party.tt_forge_models.flan_t5.pytorch import ModelLoader
import torch

# Load model and input
loader = ModelLoader()
model = loader.load_model()
model.config.return_dict = False
model.config.use_cache = False
inputs = loader.load_inputs()


class FlanT5(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, decoder_input_ids, attention_mask=None):
        inputs = {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
        }
        output = self.model(**inputs)
        return output


model = FlanT5(model)
# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs["input_ids"], inputs["decoder_input_ids"]])

# Run inference on Tenstorrent device
output = compiled_model(inputs["input_ids"], inputs["decoder_input_ids"])

# Post-process the output
loader.decode_output(output)
