# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Bert Demo Script

import forge
from third_party.tt_forge_models.bert.pytorch import ModelLoader
from transformers.modeling_outputs import QuestionAnsweringModelOutput

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Set model to return tuple outputs instead of ModelOutput dict
model.config.return_dict = False

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs["input_ids"], inputs["attention_mask"]])

# Run inference on Tenstorrent device
output = compiled_model(inputs["input_ids"], inputs["attention_mask"])

# Convert tuple output to ModelOutput for attribute access
output = QuestionAnsweringModelOutput(
    start_logits=output[0],
    end_logits=output[1],
)

# Decode the output
print("predicted answer:", loader.decode_output(output))
