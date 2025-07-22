# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# FLAN-T5  Demo Script

import forge
from third_party.tt_forge_models.flan_t5.pytorch import ModelLoader

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs["input_ids"], inputs["decoder_input_ids"]])

# Run inference on Tenstorrent device
output = compiled_model(inputs["input_ids"], inputs["decoder_input_ids"])

# Post-process the output
loader.decode_output(output)
