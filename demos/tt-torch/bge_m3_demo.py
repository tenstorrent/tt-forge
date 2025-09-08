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
    
    # Print output in a nicer format
    print("\n=== BGE-M3 ColBERT Score Demo ===")
    print(bge_m3.colbert_score(
        output_1['colbert_vecs'][0], 
        output_2['colbert_vecs'][0]
    ))
    print(bge_m3.colbert_score(
        output_1['colbert_vecs'][0], 
        output_2['colbert_vecs'][1]
    ))
    print("\n=== BGE-M3 Lexical Weights Demo ===")
    print(bge_m3.convert_id_to_token(output_1['lexical_weights']))
    lexical_scores = bge_m3.compute_lexical_matching_score(output_1['lexical_weights'][0], output_2['lexical_weights'][0])
    print(lexical_scores)
    print(bge_m3.compute_lexical_matching_score(output_1['lexical_weights'][0], output_1['lexical_weights'][1]))



if __name__ == "__main__":
    clear_dynamo_cache()
    main()
    clear_dynamo_cache()
