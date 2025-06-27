# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MonoDepthV2 Demo Script

import forge
from third_party.tt_forge_models.monodepth2.pytorch import ModelLoader

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs])

# Run inference on Tenstorrent device
output = compiled_model(inputs)

# Post-process the output
loader.postprocess_and_save_disparity_map(output)
