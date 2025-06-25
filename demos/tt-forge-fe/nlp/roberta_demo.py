# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# RoBERTa Demo Script

import forge
from ....third_party.tt_forge_models.roberta.pytorch import ModelLoader

# Load model and input
model = ModelLoader.load_model()
inputs = ModelLoader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs])

# Run inference on Tenstorrent device
output = compiled_model(inputs)
