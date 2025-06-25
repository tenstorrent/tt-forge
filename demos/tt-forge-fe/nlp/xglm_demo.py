# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# XGLM Demo Script

import forge
from ....third_party.tt_forge_models.xglm.pytorch import ModelLoader

# Load model and input
model = ModelLoader.load_model()
input_tokens = ModelLoader.load_inputs()

# Prepare input
inputs = [input_tokens["input_ids"], input_tokens["attention_mask"]]

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=inputs)

# Run inference on Tenstorrent device
output = compiled_model(inputs)
