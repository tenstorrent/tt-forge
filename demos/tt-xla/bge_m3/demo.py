# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from FlagEmbedding import BGEM3FlagModel

#FIXME: this shouldn't be needed in CI - Handles potential duplicate backend registration
from torch._dynamo.backends.registry import _COMPILER_FNS
if "tt" in _COMPILER_FNS:
    # Backend already registered, clear it first
    del _COMPILER_FNS["tt"]

#FIXME: change to proper import path for CI (from tt_xla.python_package.tt_torch.backend import xla_backend) ?
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../third_party/tt-xla'))
from python_package.tt_torch.backend.backend import xla_backend



def main():
    # Set the XLA runtime device to TT 
    xr.set_device_type("TT")

    # Load the model
    model = BGEM3FlagModel("BAAI/bge-m3").model
    model = model.eval()

    # Load the inputs
    sentences = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]
    tokens = model.tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True
    )
    inputs = {
        "text_input": tokens,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
    }

    # Compile the model with tt backend
    compiled_model = torch.compile(model, backend=xla_backend)

    def attempt_to_device(x):
        if hasattr(x, "to"):
            return x.to(device)
        return x

    # Move model and inputs to the TT device
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)
    inputs = tree_map(attempt_to_device, inputs)

    # Run the model and print top predictions
    with torch.no_grad():
        outputs = compiled_model(**inputs)       

       
if __name__ == "__main__":
    main()
