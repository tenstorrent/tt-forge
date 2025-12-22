# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BGE-M3 Encode Demo (compiling the third-party encode function)

import torch
import numpy as np
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


from third_party.tt_forge_models.bge_m3.encode.pytorch import (
    ModelLoader as BGE3EncodeLoader,
    ModelVariant as BGE3EncodeVariant,
)

from third_party.tt_forge_models.bge_m3.pytorch import (
    ModelLoader as BGE3Loader,
    ModelVariant as BGE3Variant,
)


def run_bge_m3_encode_compiled_fn_demo_case(variant):
    xr.set_device_type("TT")

    loader_encode = BGE3EncodeLoader(variant=variant)
    encode_fn = loader_encode.load_model()
    inputs = loader_encode.load_inputs()

    loader_model = BGE3Loader(variant=BGE3Variant.BASE)
    torch_model = loader_model.load_model()

    compiled_torch_model = torch.compile(torch_model, backend=xla_backend)

    device = xm.xla_device()
    compiled_torch_model = compiled_torch_model.to(device)

    bge_m3 = loader_encode.model
    bge_m3.model = compiled_torch_model
    bge_m3.target_devices = [str(device)]

    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]

    with torch.no_grad():
        output_1 = encode_fn(
            sentences=sentences_1,
            return_dense=inputs["return_dense"],
            return_sparse=inputs["return_sparse"],
            return_colbert_vecs=inputs["return_colbert_vecs"],
            device=str(device),
        )
        output_2 = encode_fn(
            sentences=sentences_2,
            return_dense=inputs["return_dense"],
            return_sparse=inputs["return_sparse"],
            return_colbert_vecs=inputs["return_colbert_vecs"],
            device=str(device),
        )

    print(f"\nModel Variant: {variant}")

    dense_1 = output_1["dense_vecs"]
    dense_2 = output_2["dense_vecs"]
    print("\n=== BGE-M3 Dense Embedding Demo ===")
    print(f"Dense embedding shape for sentences_1: {dense_1.shape}")
    print(f"Dense embedding shape for sentences_2: {dense_2.shape}")
    sim = dense_1 @ dense_2.T
    print("\nSimilarity matrix (sentences_1 x sentences_2):")
    print(sim)
    print("\nDetailed similarities:")
    print(f"  '{sentences_1[0]}' vs '{sentences_2[0][:50]}...': {sim[0, 0]:.4f}")
    print(f"  '{sentences_1[0]}' vs '{sentences_2[1][:50]}...': {sim[0, 1]:.4f}")
    print(f"  '{sentences_1[1]}' vs '{sentences_2[0][:50]}...': {sim[1, 0]:.4f}")
    print(f"  '{sentences_1[1]}' vs '{sentences_2[1][:50]}...': {sim[1, 1]:.4f}")

    print("\n=== BGE-M3 Sparse Embedding (Lexical Weight) Demo ===")
    print(bge_m3.convert_id_to_token(output_1["lexical_weights"]))
    print(bge_m3.compute_lexical_matching_score(output_1["lexical_weights"][0], output_2["lexical_weights"][0]))
    print(bge_m3.compute_lexical_matching_score(output_1["lexical_weights"][0], output_1["lexical_weights"][1]))

    print("\n=== BGE-M3 Multi-Vector (ColBERT) Demo ===")
    print(bge_m3.colbert_score(output_1["colbert_vecs"][0], output_2["colbert_vecs"][0]))
    print(bge_m3.colbert_score(output_1["colbert_vecs"][0], output_2["colbert_vecs"][1]))


if __name__ == "__main__":
    demo_cases = [BGE3EncodeVariant.BASE]
    for variant in demo_cases:
        run_bge_m3_encode_compiled_fn_demo_case(variant)
