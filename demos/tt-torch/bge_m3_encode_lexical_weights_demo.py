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
    sentences = ["What is BGE M3?", "Defination of BM25"]
    input_args = {
        "sentences": sentences,
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": False,
    }
   
    # Set up compiler configuration and backend options
    cc = CompilerConfig()
    cc.enable_consteval = True
    cc.consteval_parameters = True

    options = BackendOptions()
    options.compiler_config = cc

    # Compile and run the model, passing in the custom backend and options
    compiled_model = torch.compile(model, backend=xla_backend, dynamic=False, options=options)
    output = compiled_model(**input_args)
    
    print("\n=== BGE-M3 Lexical Weights Demo ===")
    print(bge_m3.convert_id_to_token(output['lexical_weights']))



if __name__ == "__main__":
    clear_dynamo_cache()
    main()
    clear_dynamo_cache()
