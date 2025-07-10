# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Yolox Demo Script
import os
import forge
from third_party.tt_forge_models.yolox.pytorch import ModelLoader
from third_party.tt_forge_models.tools.utils import get_file

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
compiled_model = forge.compile(model, sample_inputs=[inputs])

# Run inference on Tenstorrent device
output = compiled_model(inputs)

# Post-process the output
loader.post_processing(output)

# Remove the cached .pth weight file
model_name = loader.model_variant.replace("-", "_")
weight_url = f"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model_name}.pth"
weight_path = get_file(weight_url)

# Clean up the downloaded model file
if weight_path.exists():
    os.remove(weight_path)
    print(f"Removed downloaded weight file: {weight_path}")
