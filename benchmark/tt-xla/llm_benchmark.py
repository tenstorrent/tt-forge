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
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
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

xr.set_device_type("TT")

MIN_STEPS = 16

# Default input prompt
DEFAULT_INPUT_PROMPT = "Here is an exaustive list of the best practices for writing clean code:"

MODULE_EXPORT_PATH = "modules"


def setup_model_and_tokenizer(model_loader, model_variant) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_loader: Loader of the HuggingFace model.
        model_variant: Specific variant of the model.

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model {model_loader.get_model_info(variant=model_variant).name}...")

    model = model_loader.load_model(dtype_override=torch.bfloat16)
    model = model.eval()
    tokenizer = model_loader.tokenizer

    return model, tokenizer


def construct_inputs(
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
    input_prompt = DEFAULT_INPUT_PROMPT
    input_prompt = [input_prompt] * batch_size

    # TODO: Only works on same length inputs for now
    prompt_len = len(input_prompt[0])
    assert all(len(prompt) == prompt_len for prompt in input_prompt), "All input prompts must have the same length"

    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        max_length=max_cache_len,
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
    Transfer inputs to device.

    Args:
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        Tuple input_args on device
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
    read_logits_fn: callable,
    verbose: bool = True,
    is_multichip: bool = False,
    mesh: Mesh = None,
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
            output = model(**input_args)

            logits = read_logits_fn(output).to("cpu")
            output_logits.append(logits)
            next_token_ids = logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(token_id) for token_id in next_token_ids]
            output_tokens.append(output_text)

            # Check for EOS token and early exit
            if torch.all(next_token_ids == tokenizer.eos_token_id):
                if verbose:
                    print()  # Add newline after generation completes
                end = time.perf_counter_ns()
                iteration_times.append(end - start)
                if verbose:
                    print(f"Iteration\t{step}/{max_tokens_to_generate}\ttook {iteration_times[-1] / 1e6:.04} ms")
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_ids.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)
            # Reapply shardings for static cache if SPMD is enabled
            # See https://github.com/tenstorrent/tt-xla/issues/1641
            if is_multichip:
                for i, (key, value) in enumerate(
                    zip(
                        input_args["past_key_values"].key_cache,
                        input_args["past_key_values"].value_cache,
                    )
                ):
                    xs.mark_sharding(key, mesh, (None, "model", None, None))
                    xs.mark_sharding(value, mesh, (None, "model", None, None))

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
    model_loader,
    model_variant,
    optimization_level,
    trace_enabled,
    training,
    batch_size,
    loop_count,
    task,
    data_format,
    measure_cpu,
    input_sequence_length,
    experimental_compile,
    enable_weight_bfp8_conversion,
    experimental_enable_permute_matmul_fusion,
    ttnn_perf_metrics_output_file,
    read_logits_fn,
    mesh,
    shard_spec_fn,
):
    """
    Benchmark an LLM (Large Language Model) using PyTorch and torch-xla.

    This function loads an LLM, compiles it with torch-xla for the Tenstorrent backend,
    and measures its text generation performance. It performs warmup runs, collects token
    generation metrics, and validates output correctness via PCC (Pearson Correlation Coefficient).
    The benchmark measures tokens per second on both CPU and device backends.

    Args:
        model_loader: Model loader instance for loading the LLM
        model_variant: Specific variant/version of the model to benchmark
        optimization_level: tt-mlir optimization level for compilation
        training: Whether to run in training mode (not supported)
        batch_size: Batch size for text generation
        loop_count: Number of inference iterations
        task: Task type
        data_format: Data precision format
        measure_cpu: Whether to measure CPU baseline performance
        input_sequence_length: Length of input sequence for generation context
        experimental_compile: Whether to use experimental compilation features
        enable_weight_bfp8_conversion: Whether to enable bfp8 weight conversion
        experimental_enable_permute_matmul_fusion: Whether to enable permute matmul fusion optimization
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        read_logits_fn: Callback function to extract logits from model output

    Returns:
        Benchmark result containing token generation performance metrics and model information
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

    # Set up for multi-chip if applicable
    if mesh is not None and shard_spec_fn is not None:
        is_multichip = len(mesh.device_ids) > 1
        if is_multichip:
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            xr.use_spmd()
    else:
        is_multichip = False

    # Set up config variables
    max_cache_len: int = input_sequence_length

    # Connect the device
    device: torch.device = torch_xla.device()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_loader, model_variant)

    # Construct inputs, including static cache
    input_args = construct_inputs(tokenizer, model.config, batch_size, max_cache_len)

    # Limit maximum generation count to fit within preallocated static cache
    max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

    # Get CPU result
    cpu_logits, _ = generate_and_benchmark(
        model,
        input_args,
        tokenizer,
        torch.device("cpu"),
        1,
        read_logits_fn=read_logits_fn,
        verbose=False,
    )
    # Only one output makes sense to compare.
    cpu_logits = cpu_logits[0]

    if measure_cpu:
        print("Measuring CPU performance...")
        # Measure CPU performance by taking the best of each token over 256 iterations
        min_time_ns = sys.maxsize
        for i in range(MIN_STEPS):
            input_args = construct_inputs(tokenizer, model.config, batch_size, max_cache_len)
            _, iteration_times = generate_and_benchmark(
                model,
                input_args,
                tokenizer,
                torch.device("cpu"),
                max_tokens_to_generate,
                read_logits_fn=read_logits_fn,
                verbose=False,
            )
            min_time_ns = min(min_time_ns, *iteration_times)

        cpu_tokens_per_second = 1e9 / min_time_ns
        print(f"CPU tokens per second: {cpu_tokens_per_second:.2f}")
    else:
        cpu_tokens_per_second = -1.0

    # Transfer model and inputs to device
    input_args = construct_inputs(tokenizer, model.config, batch_size, max_cache_len)
    input_args = transfer_to_device(input_args, device)
    model = model.to(device, dtype=torch.bfloat16)

    # Shard model if shard spec function is provided
    if is_multichip:
        shard_specs = shard_spec_fn(model_loader, model)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

        # Also shard KV cache tensors created in input_args
        for key, value in zip(
            input_args["past_key_values"].key_cache,
            input_args["past_key_values"].value_cache,
        ):
            xs.mark_sharding(key, mesh, (None, "model", None, None))
            xs.mark_sharding(value, mesh, (None, "model", None, None))

    # Set XLA compilation options
    options = {
        "optimization_level": optimization_level,
        "enable_trace": trace_enabled,
        "export_path": MODULE_EXPORT_PATH,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "experimental_enable_weight_bfp8_conversion": enable_weight_bfp8_conversion,
        "experimental_enable_permute_matmul_fusion": experimental_enable_permute_matmul_fusion,
    }

    torch_xla.set_custom_compile_options(options)

    # Compile model
    compiled_model = torch.compile(model, backend="tt", options={"tt_experimental_compile": experimental_compile})

    # Warmup run
    print("Warming up...")
    warmup_tokens = min(MIN_STEPS, max_tokens_to_generate)
    _, _ = generate_and_benchmark(
        compiled_model,
        input_args,
        tokenizer,
        device,
        warmup_tokens,
        read_logits_fn=read_logits_fn,
        verbose=False,
        is_multichip=is_multichip,
        mesh=mesh,
    )

    # Reconstruct inputs for the actual benchmark run
    input_args = construct_inputs(tokenizer, model.config, batch_size, max_cache_len)
    input_args = transfer_to_device(input_args, device)

    # Re-shard KV cache tensors created in input_args
    if is_multichip:
        for key, value in zip(
            input_args["past_key_values"].key_cache,
            input_args["past_key_values"].value_cache,
        ):
            xs.mark_sharding(key, mesh, (None, "model", None, None))
            xs.mark_sharding(value, mesh, (None, "model", None, None))

    # Run benchmark once
    print(f"\nStarting benchmark...")
    output_logits, iteration_times = generate_and_benchmark(
        compiled_model,
        input_args,
        tokenizer,
        device,
        max_tokens_to_generate,
        read_logits_fn=read_logits_fn,
        verbose=True,
        is_multichip=is_multichip,
        mesh=mesh,
    )

    if len(iteration_times) < 10:
        raise RuntimeError("LLM benchmark failed: insufficient number of iterations completed.")

    ttft_ns = iteration_times[0]
    ttft_ms = ttft_ns / 1e6

    decode_iteration_times = iteration_times[1:]
    decode_total_time_ns = sum(decode_iteration_times)
    decode_total_time = decode_total_time_ns / 1e9

    # Calculate metrics (ignore first iteration for samples/sec)
    decode_total_tokens = len(decode_iteration_times)
    tokens_per_second = (decode_total_tokens / decode_total_time) if decode_total_time > 0 else 0.0

    metadata = get_benchmark_metadata()

    full_model_name = model_loader.get_model_info(variant=model_variant).name
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
        },
        {
            "measurement_name": "ttft",
            "value": ttft_ms,
            "target": -1,
        },
    ]

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=decode_total_time,
        total_samples=decode_total_tokens,
        samples_per_sec=tokens_per_second,
        cpu_samples_per_sec=cpu_tokens_per_second,
        evaluation_score=evaluation_score,
        batch_size=batch_size,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
        ttft_ms=ttft_ms,
    )

    # Check PCC
    pcc_value = compute_pcc(output_logits[0][0], cpu_logits[0], required_pcc=0.95)
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
        total_time=decode_total_time,
        total_samples=decode_total_tokens,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        model_info=full_model_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        input_is_image=False,
        input_sequence_length=input_sequence_length,
    )

    return result
