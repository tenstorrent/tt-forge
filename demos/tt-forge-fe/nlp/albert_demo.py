# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Albert Demo Script

import forge
from third_party.tt_forge_models.albert.masked_lm.pytorch import ModelLoader

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs["input_ids"], inputs["attention_mask"]])

# Run inference on Tenstorrent device
output = compiled_model(inputs["input_ids"], inputs["attention_mask"])

# Decode the output
predicted_token_id = loader.decode_output(output)
print("The predicted token for the [MASK] is: ", predicted_token_id)
