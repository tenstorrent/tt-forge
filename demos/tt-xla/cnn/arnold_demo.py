# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Arnold Demo Script

import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend

# Add repository root to path to locate third_party modules
import sys
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.arnold.pytorch import ModelLoader, ModelVariant


def run_arnold_demo_case(variant):

    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    screens, variables = loader.load_inputs(dtype_override=torch.bfloat16)

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

    screens = tree_map(attempt_to_device, screens)
    variables = tree_map(attempt_to_device, variables)

    # Run inference on Tenstorrent device
    with torch.no_grad():
        output = compiled_model(screens, variables)

    # Extract tensor from output if it's a list/tuple
    if isinstance(output, (list, tuple)):
        output_tensor = output[0]
    else:
        output_tensor = output

    # Post-process the output
    loader.post_process(output_tensor, return_q_values=True)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.DEFEND_THE_CENTER_FF,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_arnold_demo_case(variant)
