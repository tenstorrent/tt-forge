# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
from tt_xla.python_package.tt_torch.backend import xla_backend #FIXME: change to proper import path
from transformers import ResNetForImageClassification, AutoImageProcessor
from PIL import Image


model_name = "microsoft/resnet-50"

# Load the model and processor
model = ResNetForImageClassification.from_pretrained(model_name)
model = model.to(torch.bfloat16)
model = model.eval()
processor = AutoImageProcessor.from_pretrained(model_name)

# Load the inputs
image = Image.open("demos/tt-torch/000000039769.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt").pixel_values

compiled_model = model.compile(backend=xla_backend)

device = xm.xla_device(0)
compiled_model = compiled_model.to(device)

def attempt_to_device(x):
    if hasattr(x, "to"):
        return x.to(device)
    return x

inputs = tree_map(attempt_to_device, inputs)

ouputs = compiled_model(*inputs)