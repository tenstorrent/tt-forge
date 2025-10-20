# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import sys
from typing import List
import time
import pytest
import socket

# Third-party modules
import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import tt_torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from benchmark.utils import get_xla_device_arch
from utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants
OPTIMIZER_ENABLED = False
PROGRAM_CACHE_ENABLED = True
MEMORY_LAYOUT_ANALYSIS_ENABLED = False
TRACE_ENABLED = False

if PROGRAM_CACHE_ENABLED:
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

# Default input prompt
DEFAULT_INPUT_PROMPT = "I like taking walks in the"


def setup_model_and_tokenizer(model_loader) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_loader.get_model_info().name}...")

    model = model_loader.load_model(dtype_override=torch.bfloat16)
    model = model.eval()
    tokenizer = model_loader.tokenizer

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


def transfer_to_device(input_args: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
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

    return input_args


def generate_and_benchmark(
    model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    max_tokens_to_generate: int,
    verbose: bool = True,
):
    """
    Run the generation loop and measure time.

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        tokenizer: Tokenizer instance
        device: Device
        max_tokens_to_generate: Maximum number of tokens to generate
        verbose: Whether to print generation output

    Returns:
        Tuple of (output_logits, iteration_times)
    """
    output_tokens: List[str] = []
    output_logits: List[torch.Tensor] = []
    iteration_times: List[float] = []
    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            start = time.perf_counter_ns()

            # Run forward pass
            output: CausalLMOutputWithPast = model(**input_args)
            logits: torch.Tensor = output.logits.to("cpu")
            output_logits.append(logits)
            next_token_id = logits[:, -1].argmax(dim=-1)
            output_text = tokenizer.decode(next_token_id)
            output_tokens.append(output_text)

            # Check for EOS token and early exit
            if next_token_id.item() == tokenizer.eos_token_id:
                if verbose:
                    print()  # Add newline after generation completes
                end = time.perf_counter_ns()
                iteration_times.append(end - start)
                if verbose:
                    print(f"Iteration\t{step}/{max_tokens_to_generate}\ttook {iteration_times[-1] / 1e6:.04} ms")
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

            end = time.perf_counter_ns()
            iteration_times.append(end - start)
            if verbose:
                print(f"Iteration\t{step}/{max_tokens_to_generate}\ttook {iteration_times[-1] / 1e6:.04} ms")

    if verbose:
        print("Output tokens:", output_tokens)

    return output_logits, iteration_times


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


def benchmark_llm_torch_xla(
    training, batch_size, loop_count, task, data_format, measure_cpu, input_sequence_length, model_loader
):
    """
    This function creates an LLM based model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

        # Enforce bfloat16 only
    if data_format != "bfloat16":
        raise ValueError(
            f"Only bfloat16 data format is supported for llm benchmark. Got: {data_format}. " "Please use -df bfloat16"
        )

    if not model_loader:
        raise ValueError("Model loader must be specified for benchmark. ")

    if loop_count != 1:
        raise ValueError(
            f"Loop count must be 1 for llm benchmark (not yet supported). Got: {loop_count}. " "Please use -lp 1"
        )

    if not input_sequence_length or input_sequence_length <= 0:
        raise ValueError(
            f"Input sequence length must be a positive integer for llm benchmark. Got: {input_sequence_length}. "
            "Please use -isl <length> (e.g., -isl 128)"
        )

    if task != "text-generation":
        raise ValueError(
            f"Only 'text-generation' task is supported for llm benchmark. Got: {task}. " "Please use -t text-generation"
        )

    # Check transformers version
    check_transformers_version()

    xr.set_device_type("TT")

    # Set up config variables
    input_prompt: str = DEFAULT_INPUT_PROMPT
    max_cache_len: int = input_sequence_length

    # Connect the device
    device: torch.device = xm.xla_device()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_loader)

    # Construct inputs, including static cache
    input_args = construct_inputs(input_prompt, tokenizer, model.config, batch_size, max_cache_len)

    # Limit maximum generation count to fit within preallocated static cache
    max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

    # Get CPU result
    cpu_logits, _ = generate_and_benchmark(
        model,
        input_args,
        tokenizer,
        torch.device("cpu"),
        1,
        verbose=False,
    )
    # Only one output makes sense to compare.
    cpu_logits = cpu_logits[0]

    if measure_cpu:
        print("Measuring CPU performance...")
        # Measure CPU performance by taking the best of each token over 256 iterations
        min_time_ns = sys.maxsize
        for i in range(16):
            input_args = construct_inputs(input_prompt, tokenizer, model.config, batch_size, max_cache_len)
            _, iteration_times = generate_and_benchmark(
                model,
                input_args,
                tokenizer,
                torch.device("cpu"),
                max_tokens_to_generate,
                verbose=False,
            )
            min_time_ns = min(min_time_ns, *iteration_times)

        cpu_tokens_per_second = 1e9 / min_time_ns
        print(f"CPU tokens per second: {cpu_tokens_per_second:.2f}")
    else:
        cpu_tokens_per_second = -1.0

    # Transfer model and inputs to device
    input_args = construct_inputs(input_prompt, tokenizer, model.config, batch_size, max_cache_len)
    input_args = transfer_to_device(input_args, device)
    model = model.to(device, dtype=torch.bfloat16)

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

    # Warmup run
    print("Warming up...")
    warmup_tokens = min(16, max_tokens_to_generate)
    _, _ = generate_and_benchmark(
        compiled_model,
        input_args,
        tokenizer,
        device,
        warmup_tokens,
        verbose=False,
    )

    # Reconstruct inputs for the actual benchmark run
    input_args = construct_inputs(input_prompt, tokenizer, model.config, batch_size, max_cache_len)
    input_args = transfer_to_device(input_args, device)

    # Run benchmark once
    print(f"\nStarting benchmark...")
    output_logits, iteration_times = generate_and_benchmark(
        compiled_model,
        input_args,
        tokenizer,
        device,
        max_tokens_to_generate,
        verbose=True,
    )

    total_time_ns = sum(iteration_times)
    total_time = total_time_ns / 1e9

    # Calculate metrics
    total_tokens = len(output_logits)
    tokens_per_second = total_tokens / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_loader.get_model_info().name
    model_type = "text-generation"
    dataset_name = "Random Data"

    # Extract number of layers from model config if available
    num_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else -1

    evaluation_score = 0.0
    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_tokens_per_second,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_tokens,
        samples_per_sec=tokens_per_second,
        cpu_samples_per_sec=cpu_tokens_per_second,
        evaluation_score=evaluation_score,
        batch_size=batch_size,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
    )

    # Check PCC
    pcc_value = compute_pcc(output_logits[0], cpu_logits, required_pcc=0.99)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=(input_sequence_length,),
        loop_count=loop_count,
        data_format=data_format,
        training=training,
        total_time=total_time,
        total_samples=total_tokens,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimizer_enabled=OPTIMIZER_ENABLED,
        program_cache_enabled=PROGRAM_CACHE_ENABLED,
        memory_layout_analysis_enabled=MEMORY_LAYOUT_ANALYSIS_ENABLED,
        trace_enabled=TRACE_ENABLED,
        model_info=full_model_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        input_is_image=False,
        input_sequence_length=input_sequence_length,
    )

    return result
