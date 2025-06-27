# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StaticCache,
)

_global_max_cache_len = 64 + 64


def clear_dynamo_cache():
    # taken from/ inspired by: https://github.com/pytorch/pytorch/issues/107444
    import torch._dynamo as dynamo
    import gc

    dynamo.reset()  # clear cache
    gc.collect()


def load_model(model_name="meta-llama/Llama-3.2-3B"):
    # set up the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def load_inputs(
    model,
    tokenizer,
    test_input="This is a sample text from ",
    max_cache_len=_global_max_cache_len,
):
    batch_size = 1
    inputs = tokenizer.encode_plus(
        test_input,
        return_tensors="pt",
        truncation=True,
    )

    # set up static cache
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=model.device,
        dtype=model.dtype,
    )

    cache_position = torch.arange(0, inputs.input_ids.shape[1])

    args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "use_cache": True,
        "cache_position": cache_position,
    }
    return args


@torch.inference_mode()
def main():
    model, tokenizer = load_model()
    input_args = load_inputs(model, tokenizer)
    generated_ids = input_args["input_ids"]
    print(tokenizer.decode(generated_ids[0].tolist()), end="", flush=True)

    clear_dynamo_cache()
    cc = CompilerConfig()
    cc.enable_consteval = False
    cc.consteval_parameters = False

    options = BackendOptions()
    options.compiler_config = cc

    device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
    options.devices = [device]

    buffer_cache = {}
    options.buffer_cache = buffer_cache

    constant_cache = {}
    options.constant_cache = constant_cache

    compiled_model = torch.compile(model, backend=backend, dynamic=False, options=options)

    # up to _global_max_cache_len - input_args["input_ids"].shape[1]
    tokens_to_generate = 32

    for i in range(tokens_to_generate):
        outputs = compiled_model(**input_args)
        next_token_ids = outputs.logits[:, -1:].argmax(dim=-1)
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        print(tokenizer.decode(next_token_ids[0].tolist()), end="", flush=True)

        cache_position = input_args["cache_position"][-1:] + 1
        input_args = {
            "input_ids": next_token_ids.to(dtype=torch.int32),
            "past_key_values": input_args["past_key_values"],  # updated in place
            "cache_position": cache_position,
            "use_cache": True,
        }
    print()  # Add a newline at the end of the output
    DeviceManager.release_parent_device(device)
    clear_dynamo_cache()


if __name__ == "__main__":
    main()
