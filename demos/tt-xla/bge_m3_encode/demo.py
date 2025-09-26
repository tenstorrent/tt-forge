# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend
from typing import List, Union, Optional, Dict, Literal
import numpy as np
from collections import defaultdict
from FlagEmbedding import BGEM3FlagModel

# encode function from https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py
# Modified to remove unnecessary overhead handled by the tt backend
def encode(
    model,
    sentences: Union[List[str], str],
    return_dense: Optional[bool] = None,
    return_sparse: Optional[bool] = None,
    return_colbert_vecs: Optional[bool] = None,
    device: Optional[str] = None,
) -> Dict[
    Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
    Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]],
]:
    # Tokenize the input sentences and move to device
    text_input = model.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    text_input.to(device)
    inputs = {
        "text_input": text_input,
        "return_dense": return_dense,
        "return_sparse": return_sparse,
        "return_colbert_vecs": return_colbert_vecs,
    }

    # Move model to device and run inference
    model = model.to(device)
    outputs = model(**inputs)

    def _process_token_weights(token_weights: np.ndarray, input_ids: list):
        result = defaultdict(int)
        unused_tokens = set()
        for _token in ["cls_token", "eos_token", "pad_token", "unk_token"]:
            if _token in model.tokenizer.special_tokens_map:
                _token_id = model.tokenizer.convert_tokens_to_ids(model.tokenizer.special_tokens_map[_token])
                unused_tokens.add(_token_id)
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx = str(idx)
                if w > result[idx]:
                    result[idx] = w
        return result

    def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
        tokens_num = np.sum(attention_mask)
        return colbert_vecs[: tokens_num - 1]

    all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []

    batch_size = text_input["input_ids"].shape[0]
    length_sorted_idx = np.argsort([-len(text_input["input_ids"][i]) for i in range(batch_size)])

    all_dense_embeddings.append(outputs["dense_vecs"].cpu().detach())

    token_weights = outputs["sparse_vecs"].squeeze(-1)
    all_lexical_weights.extend(
        list(
            map(
                _process_token_weights,
                token_weights.cpu().detach().numpy(),
                text_input["input_ids"].cpu().detach().numpy().tolist(),
            )
        )
    )

    all_colbert_vecs.extend(
        list(
            map(
                _process_colbert_vecs,
                outputs["colbert_vecs"].cpu().detach().numpy(),
                text_input["attention_mask"].cpu().detach().numpy(),
            )
        )
    )

    all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
    all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]

    all_lexical_weights = [all_lexical_weights[i] for i in np.argsort(length_sorted_idx)]

    all_colbert_vecs = [all_colbert_vecs[i] for i in np.argsort(length_sorted_idx)]

    # return the embeddings
    return {
        "dense_vecs": all_dense_embeddings,
        "lexical_weights": all_lexical_weights,
        "colbert_vecs": all_colbert_vecs,
    }


def main():
    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load the model
    bge_m3 = BGEM3FlagModel("BAAI/bge-m3")
    model = bge_m3.model
    model = model.eval()

    # Load the inputs
    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]
    encode_kwargs = {
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
        "device": "xla",
    }

    # Compile the model with tt backend
    compiled_model = torch.compile(model, backend=xla_backend)

    # Initialize the tt device
    device = xm.xla_device()

    # Run the model
    output_1 = encode(compiled_model, sentences_1, **encode_kwargs)
    output_2 = encode(compiled_model, sentences_2, **encode_kwargs)

    # Print Dense Embedding results
    print("\n=== BGE-M3 Dense Embedding Demo ===")
    # Extract dense embeddings
    embeddings_1 = output_1["dense_vecs"]
    embeddings_2 = output_2["dense_vecs"]

    # Compute similarity matrix (dot product between normalized embeddings)
    # Note: BGE-M3 embeddings are already normalized, so we can directly compute dot product
    similarity = embeddings_1 @ embeddings_2.T

    print(f"Dense embedding shape for sentences_1: {embeddings_1.shape}")
    print(f"Dense embedding shape for sentences_2: {embeddings_2.shape}")
    print(f"\nSimilarity matrix (sentences_1 x sentences_2):")
    print(similarity)
    print(f"\nDetailed similarities:")
    print(f"  '{sentences_1[0]}' vs '{sentences_2[0][:50]}...': {similarity[0, 0]:.4f}")
    print(f"  '{sentences_1[0]}' vs '{sentences_2[1][:50]}...': {similarity[0, 1]:.4f}")
    print(f"  '{sentences_1[1]}' vs '{sentences_2[0][:50]}...': {similarity[1, 0]:.4f}")
    print(f"  '{sentences_1[1]}' vs '{sentences_2[1][:50]}...': {similarity[1, 1]:.4f}")

    # Print results as done in Sparse Embedding and Multi-Vector Hugging Face examples
    print("\n=== BGE-M3 Sparse Embedding (Lexical Weight) Demo ===")
    print(bge_m3.convert_id_to_token(output_1["lexical_weights"]))
    print(bge_m3.compute_lexical_matching_score(output_1["lexical_weights"][0], output_2["lexical_weights"][0]))
    print(bge_m3.compute_lexical_matching_score(output_1["lexical_weights"][0], output_1["lexical_weights"][1]))
    print("\n=== BGE-M3 Multi-Vector (ColBERT) Demo ===")
    print(bge_m3.colbert_score(output_1["colbert_vecs"][0], output_2["colbert_vecs"][0]))
    print(bge_m3.colbert_score(output_1["colbert_vecs"][0], output_2["colbert_vecs"][1]))


if __name__ == "__main__":
    main()
