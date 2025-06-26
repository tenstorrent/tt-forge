# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# GPT-Neo Demo Script

import forge
from third_party.tt_forge_models.gpt_neo.pytorch import ModelLoader

# Load model and input
loader = ModelLoader()
model = loader.load_model()
model.config.return_dict = False
model.config.use_cache = False
input_tokens = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[input_tokens["input_ids"]])

# Run inference on Tenstorrent device
output = compiled_model(input_tokens["input_ids"])

# post-process
generated_text = loader.decode_output(output[0])
print("generated_text: ", generated_text)
