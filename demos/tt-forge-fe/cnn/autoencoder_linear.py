# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Autoencoder-linear Demo Script

import os
import forge
from third_party.tt_forge_models.autoencoder_linear.pytorch import ModelLoader

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs])

# Run inference on Tenstorrent device
output = compiled_model(inputs)

# Post-process the output
output_dir = "demos/tt-forge-fe/cnn/autoencoder_linear"
os.makedirs(output_dir, exist_ok=True)
loader.post_processing(output, output_dir)
