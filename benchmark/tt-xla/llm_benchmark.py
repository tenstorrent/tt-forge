# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import sys
from typing import List, Optional
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
    build_xla_export_name,
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
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    model = model.eval()
    tokenizer = model_loader.tokenizer

    return model, tokenizer


def construct_inputs(
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
    past_key_values: Optional[StaticCache] = None,
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

    if past_key_values is None:
        if hasattr(model_config, "head_dim") and model_config.head_dim:
            head_dim = model_config.head_dim
        else:
            head_dim = model_config.hidden_size // model_config.num_attention_heads
        num_key_value_heads = getattr(model_config, "num_key_value_heads", model_config.num_attention_heads)

        # Static cache should be initialized on CPU and separately transferred to device
        # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
        static_cache: StaticCache = StaticCache(
            config=model_config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
        static_cache.early_initialization(
            batch_size=batch_size,
            num_heads=num_key_value_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )
    else:
        static_cache = past_key_values
    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    return input_args


def get_mesh(model_loader, mesh_config_fn):
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_name = mesh_config_fn(model_loader, num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, mesh_name)


def transfer_to_device(input_args: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """
    Transfer inputs to device.

    Args:
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        Tuple input_args on device
    """
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
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

            end = time.perf_counter_ns()
            iteration_times.append(end - start)
            if verbose:
                print(f"Iteration\t{step}/{max_tokens_to_generate}\ttook {iteration_times[-1] / 1e6:.04} ms")

    if verbose:
        print("Output tokens:", output_tokens)

    return output_logits, iteration_times


def check_transformers_version():
    """
    Check that transformers version is <= 4.57.1.
    Raises RuntimeError if version is incompatible.
    """
    import packaging.version

    current_version = packaging.version.parse(transformers.__version__)
    max_version = packaging.version.parse("4.57.1")

    if current_version > max_version:
        raise RuntimeError(
            f"Transformers version {transformers.__version__} is not supported. " f"Please use version <= 4.57.1"
        )


def benchmark_llm_torch_xla(
    model_loader,
    model_variant,
    display_name,
    optimization_level,
    trace_enabled,
    batch_size,
    loop_count,
    task,
    data_format,
    input_sequence_length,
    experimental_compile,
    enable_weight_bfp8_conversion,
    experimental_enable_permute_matmul_fusion,
    ttnn_perf_metrics_output_file,
    read_logits_fn,
    mesh_config_fn,
    shard_spec_fn,
    arch,
    required_pcc,
):
    """
    Benchmark an LLM (Large Language Model) using PyTorch and torch-xla.

    This function loads an LLM, compiles it with torch-xla for the Tenstorrent backend,
    and measures its text generation performance. It performs warmup runs, collects token
    generation metrics, and validates output correctness via PCC (Pearson Correlation Coefficient).
    The benchmark measures tokens per second on the device backend.

    Args:
        model_loader: Model loader instance for loading the LLM
        model_variant: Specific variant/version of the model to benchmark
        optimization_level: tt-mlir optimization level for compilation
        batch_size: Batch size for text generation
        loop_count: Number of inference iterations
        task: Task type
        data_format: Data precision format
        input_sequence_length: Length of input sequence for generation context
        experimental_compile: Whether to use experimental compilation features
        enable_weight_bfp8_conversion: Whether to enable bfp8 weight conversion
        experimental_enable_permute_matmul_fusion: Whether to enable permute matmul fusion optimization
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        read_logits_fn: Callback function to extract logits from model output
        required_pcc: Required PCC threshold for validation

    Returns:
        Benchmark result containing token generation performance metrics and model information
    """

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
    if mesh_config_fn is not None and shard_spec_fn is not None:
        is_multichip = xr.global_runtime_device_count() > 1
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
    full_model_name = model_loader.get_model_info(variant=model_variant).name

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

    # Transfer model and inputs to device
    input_args = construct_inputs(tokenizer, model.config, batch_size, max_cache_len)
    input_args = transfer_to_device(input_args, device)
    model = model.to(device, dtype=torch.bfloat16)

    # Shard model if shard spec function is provided
    mesh = None
    if is_multichip:
        shard_specs = shard_spec_fn(model_loader, model)
        mesh = get_mesh(model_loader, mesh_config_fn)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

        # Also shard KV cache tensors created in input_args
        for layer in input_args["past_key_values"].layers:
            xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
            xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    # Set XLA compilation options
    num_layers_override = getattr(model_loader, "num_layers", None)
    export_model_name = build_xla_export_name(
        model_name=display_name,
        num_layers=num_layers_override,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
    )
    options = {
        "optimization_level": optimization_level,
        "enable_trace": trace_enabled,
        "export_path": MODULE_EXPORT_PATH,
        "export_model_name": export_model_name,
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
    input_args = construct_inputs(
        tokenizer, model.config, batch_size, max_cache_len, past_key_values=input_args["past_key_values"]
    )
    input_args = transfer_to_device(input_args, device)

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

    model_type = "text-generation"
    dataset_name = "Random Data"

    # Extract number of layers from model config if available
    num_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else -1

    evaluation_score = 0.0
    custom_measurements = [
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
        evaluation_score=evaluation_score,
        batch_size=batch_size,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
        ttft_ms=ttft_ms,
    )

    # Check PCC
    pcc_value = compute_pcc(output_logits[0][0], cpu_logits[0], required_pcc=required_pcc)
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
        total_time=decode_total_time,
        total_samples=decode_total_tokens,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        model_info=full_model_name,
        display_name=display_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=arch or get_xla_device_arch(),
        input_is_image=False,
        input_sequence_length=input_sequence_length,
    )

    return result
