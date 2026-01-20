# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Falcon Demo Script

import forge
from third_party.tt_forge_models.falcon.pytorch import ModelLoader

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs["input_ids"], inputs["attention_mask"]])

# Run inference on Tenstorrent device
output = compiled_model(inputs)

# Decode the output
loader.decode_output(output)
