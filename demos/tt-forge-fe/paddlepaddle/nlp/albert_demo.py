# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ALBERT Demo Script

import forge
from third_party.tt_forge_models.albert.masked_lm.paddlepaddle import ModelLoader
from forge.tvm_calls.forge_utils import paddle_trace

# Load model and input
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()

# Compile the model using Forge
framework_model, _ = paddle_trace(model, inputs=inputs)
compiled_model = forge.compile(framework_model, inputs)

# Run inference on Tenstorrent device
output = compiled_model(*inputs)
