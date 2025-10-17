# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List
import time

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

# Common constants
OPTIMIZER_ENABLED = False
PROGRAM_CACHE_ENABLED = True
MEMORY_LAYOUT_ANALYSIS_ENABLED = False
TRACE_ENABLED = False

if PROGRAM_CACHE_ENABLED:
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"


def setup_model_and_tokenizer(
    model_name: str,
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    model = model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def construct_inputs(
    input_prompt: str,
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
) -> dict:
    """
    Construct inputs including static cache.

    Args:
        input_prompt: Input text prompt
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length

    Returns:
        Dictionary containing input_ids, past_key_values, cache_position, and use_cache
    """
    inputs = tokenizer.encode_plus(
        input_prompt,
        return_tensors="pt",
        truncation=True,
    )

    # Static cache should be initialized on CPU and separately transferred to device
    # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
    static_cache: StaticCache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    return input_args


def transfer_to_device(model: torch.nn.Module, input_args: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """
    Transfer model and inputs to device.

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        Tuple of (model, input_args) on device
    """
    input_args["past_key_values"].key_cache = [k.to(device) for k in input_args["past_key_values"].key_cache]
    input_args["past_key_values"].value_cache = [v.to(device) for v in input_args["past_key_values"].value_cache]
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)

    model = model.to(device)

    return model, input_args


def generate_and_benchmark(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_tokens_to_generate: int = 128,
):
    """
    Run the generation loop.

    Args:
        compiled_model: Compiled model instance
        input_args: Input arguments dictionary
        tokenizer: Tokenizer instance
        device: Device
        mesh: Device mesh for SPMD operations (optional)
        is_spmd: Whether SPMD mode is enabled
        max_tokens_to_generate: Maximum number of tokens to generate
    """
    output_tokens: List[str] = []
    itteration_times: List[float] = []
    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            # Run forward pass
            start = time.perf_counter_ns()

            output: CausalLMOutputWithPast = compiled_model(**input_args)
            output_logits: torch.Tensor = output.logits.to("cpu")
            next_token_id = output_logits[:, -1].argmax(dim=-1)
            output_text = tokenizer.decode(next_token_id)
            output_tokens.append(output_text)
            print(output_text, end="", flush=True)

            # Check for EOS token and early exit
            if next_token_id.item() == tokenizer.eos_token_id:
                print()  # Add newline after generation completes
                end = time.perf_counter_ns()
                itteration_times.append(end - start)
                print(" | Step time (ms): ", (end - start) / 1e6)
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

            end = time.perf_counter_ns()
            itteration_times.append(end - start)
            print(" | Step time (ms): ", (end - start) / 1e6)
    print("Output tokens:", output_tokens)

    total_time = sum(itteration_times)
    tps = 1e9 * (len(itteration_times) / total_time)
    print(f"Total generation time (ms): {total_time / 1e6:.2f}")
    print(f"Tokens per second (TPS): {tps:.2f}")


def check_transformers_version():
    """
    Check that transformers version is <= 4.52.4.
    Raises RuntimeError if version is incompatible.

    This is because transformers SDPA implementation changed in later versions,
    which causes dynamo trace to fail.

    See https://github.com/tenstorrent/tt-xla/issues/1020
    """
    import packaging.version

    current_version = packaging.version.parse(transformers.__version__)
    max_version = packaging.version.parse("4.52.4")

    if current_version > max_version:
        raise RuntimeError(
            f"Transformers version {transformers.__version__} is not supported. " f"Please use version <= 4.52.4"
        )


def test_llama_torch_xla():
    # Check transformers version
    check_transformers_version()

    # Set up config variables.
    batch_size: int = 1
    max_cache_len: int = 128
    input_prompt: str = "I like taking walks in the"
    model_name: str = "meta-llama/Llama-3.2-3B"

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Construct inputs, including static cache
    input_args = construct_inputs(input_prompt, tokenizer, model.config, batch_size, max_cache_len)

    # Limit maximum generation count to fit within preallocated static cache
    max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

    # Transfer model and inputs to device
    model, input_args = transfer_to_device(model, input_args, device)

    # Set XLA compilation options
    options = {
        "enable_optimizer": OPTIMIZER_ENABLED,
        "enable_memory_layout_analysis": MEMORY_LAYOUT_ANALYSIS_ENABLED,
        "enable_l1_interleaved": False,
        "enable_fusing_conv2d_with_multiply_pattern": True,
    }

    torch_xla.set_custom_compile_options(options)

    # Compile model
    compiled_model = torch.compile(model, backend="tt")

    # Warmp-up run
    print("Warming up...")
    generate_and_benchmark(
        compiled_model,
        input_args,
        tokenizer,
        device,
        16,
    )

    generate_and_benchmark(
        compiled_model,
        input_args,
        tokenizer,
        device,
        max_tokens_to_generate,
    )


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    test_llama_torch_xla()
