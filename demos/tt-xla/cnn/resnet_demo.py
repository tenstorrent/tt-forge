# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet Demo Script

import sys
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Add repository root to path to locate third_party modules
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.resnet.pytorch import ModelLoader, ModelVariant
from datasets import load_dataset
import random


def run_resnet_demo_case(variant):

    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # Compile the model using XLA
    compiled_model = torch.compile(model, backend=xla_backend)

    # Move model and inputs to the TT device
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)
    # Keep framework model on CPU for post-processing comparison
    def attempt_to_device(x):
        if hasattr(x, "to"):
            return x.to(device)
        return x

    inputs = tree_map(attempt_to_device, inputs)

    # Run inference on Tenstorrent device
    with torch.no_grad():
        output = compiled_model(inputs)

    if not isinstance(output, (list, tuple)):
        output = [output]
    loader.output_postprocess(co_out=output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.RESNET_50_TIMM,
        ModelVariant.RESNET_18,
        ModelVariant.RESNET_34,
        ModelVariant.RESNET_50,
        ModelVariant.RESNET_101,
        ModelVariant.RESNET_152,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_resnet_demo_case(variant)
