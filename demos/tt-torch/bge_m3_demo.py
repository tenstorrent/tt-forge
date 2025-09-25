# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import backend, BackendOptions
from tt_torch.dynamo.experimental.xla_backend import xla_backend
from tt_torch.tools.device_manager import DeviceManager
from FlagEmbedding import BGEM3FlagModel


def clear_dynamo_cache():
    # taken from/ inspired by: https://github.com/pytorch/pytorch/issues/107444
    import torch._dynamo as dynamo
    import gc

    dynamo.reset()  # clear cache
    gc.collect()


def main():
    # Load the model
    bge_m3 = BGEM3FlagModel("BAAI/bge-m3")
    model = bge_m3.encode

    # Create tokenized inputs for the model
    sentences_1 = ["What is BGE M3?", "Defination of BM25"]
    sentences_2 = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]
    input_args_1 = {
        "sentences": sentences_1,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
    }
    input_args_2 = {
        "sentences": sentences_2,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
    }

    # Set up compiler configuration and backend options
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc

    # Compile and run the model, passing in the custom backend and options
    compiled_model = torch.compile(model, backend=xla_backend, dynamic=False, options=options)
    output_1 = compiled_model(**input_args_1)
    output_2 = compiled_model(**input_args_2)

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
    clear_dynamo_cache()
    main()
    clear_dynamo_cache()
