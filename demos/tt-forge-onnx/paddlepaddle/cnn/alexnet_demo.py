# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Alexnet Demo Script

import forge
from third_party.tt_forge_models.alexnet.image_classification.paddlepaddle import ModelLoader

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, inputs)

# Run inference on Tenstorrent device
output = compiled_model(*inputs)

# Post-process the output
loader.print_results(output)
