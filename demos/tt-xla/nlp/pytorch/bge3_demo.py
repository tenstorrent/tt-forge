# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BGE-M3 Demo Script

import sys
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend
from third_party.tt_forge_models.bge_m3.pytorch import (
    ModelLoader as BGE3Loader,
    ModelVariant as BGE3Variant,
)


def run_bge_m3_demo_case(variant):

    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load Model and inputs
    loader = BGE3Loader(variant=variant)
    framework_model = loader.load_model()
    input_dict = loader.load_inputs()

    # Compile the model using XLA
    compiled_model = torch.compile(framework_model, backend=xla_backend)

    # Move model and inputs to the TT device
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)

    def attempt_to_device(x):
        if hasattr(x, "to"):
            return x.to(device)
        return x

    # Move inputs to device
    inputs = tree_map(attempt_to_device, input_dict)

    # Run inference on Tenstorrent device
    with torch.no_grad():
        output = compiled_model(**inputs)

    # Display results
    print(f"Input texts: {input_dict['text_input']['input_ids'].shape}")
    print(f"Dense embeddings shape: {output['dense_vecs'].shape}")
    print(f"Sparse embeddings shape: {output['sparse_vecs'].shape}")
    print(f"ColBERT embeddings shape: {output['colbert_vecs'].shape}")


if __name__ == "__main__":

    demo_cases = [
        # BGE-M3 model variants
        BGE3Variant.BASE,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_bge_m3_demo_case(variant)
