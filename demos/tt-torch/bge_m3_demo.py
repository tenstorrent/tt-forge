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
    # Load the model and set to evaluation mode
    model = BGEM3FlagModel("BAAI/bge-m3").model
    model = model.eval()

    # Create tokenized inputs for the model 
    sentences = [
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document",
    ]
    text_input = model.tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True
    )
    input_args = {
        "text_input": text_input,
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
    out = compiled_model(**input_args)
    
    # Print output in a nicer format
    print("=== BGE-M3 Model Output ===")
    print(f"Dense embeddings shape: {out['dense_vecs'].shape}")
    print(f"Sparse embeddings shape: {out['sparse_vecs'].shape}")
    print(f"ColBERT embeddings shape: {out['colbert_vecs'].shape}")



if __name__ == "__main__":
    clear_dynamo_cache()
    main()
    clear_dynamo_cache()
