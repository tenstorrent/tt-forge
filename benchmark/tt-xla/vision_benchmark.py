# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import socket
import pytest
import time

# Third-party modules
import torch
import torch_xla
import torch_xla.runtime as xr
import torch.nn as nn

from benchmark.utils import get_xla_device_arch
from utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
    move_to_cpu,
)

xr.set_device_type("TT")

MIN_STEPS = 16

MODULE_EXPORT_PATH = "modules"


def warmup_vision_model(model, load_inputs_fn, batch_size, data_format, device, loop_count, extract_output_tensor_fn):
    """
    Warmup the model for a given number of loop_count.

    Parameters:
    ----------
    model: Callable
        The model to warmup.
    load_inputs_fn: Callable
        Function to load a single batch of preprocessed inputs.
        Signature: fn(batch_size, data_format) -> Tensor
    batch_size: int
        Batch size for inputs.
    data_format: str
        Data format (bfloat16 or float32).
    device: torch.device
        The device to run the warmup on.
    loop_count: int
        The number of iterations to warmup the model.
    extract_output_tensor_fn: Callable
        Function to extract tensor from model output (e.g. get .logits from HF output).
    """
    print("Warming up the device...")

    with torch.no_grad():
        for i in range(loop_count):
            # Load and preprocess input
            input_tensor = load_inputs_fn(batch_size, data_format)
            # Move input to device
            device_input = input_tensor.to(device)
            # Model forward, non blocking.
            output = model(device_input)
            # Extract output tensor and move to CPU
            _ = move_to_cpu(extract_output_tensor_fn(output))

    print("Warming up completed.")


def measure_fps_vision_model(
    model, load_inputs_fn, batch_size, data_format, device, loop_count, extract_output_tensor_fn
):
    """
    Benchmark the model for a given number of loop_count.

    Parameters:
    ----------
    model: Callable
        The model to benchmark.
    load_inputs_fn: Callable
        Function to load a single batch of preprocessed inputs.
        Signature: fn(batch_size, data_format) -> Tensor
    batch_size: int
        Batch size for inputs.
    data_format: str
        Data format (bfloat16 or float32).
    device: torch.device
        The device to run the benchmark on.
    loop_count: int
        Number of batches to process.
    extract_output_tensor_fn: Callable
        Function to extract tensor from model output (e.g. get .logits from HF output).

    Returns:
    -------
    predictions: list of torch.Tensor
        The predictions made by the model (on CPU).
    total_time: float
        The total time taken to process the inputs in seconds.
    """
    print("Starting benchmark loop...")

    predictions = []
    iteration_times = []
    inputs = [load_inputs_fn(batch_size, data_format) for _ in range(loop_count)]
    with torch.no_grad():
        outputs = []
        for i in range(loop_count):
            # Load and preprocess input
            start_time = time.perf_counter_ns()

            # Move input to device
            device_input = inputs[i].to(device)

            # Model forward, non blocking.
            output = model(device_input)

            # Extract output tensor
            output = extract_output_tensor_fn(output)
            outputs.append(output)

            end_time = time.perf_counter_ns()
            iteration_times.append(end_time - start_time)

            print(f"Iteration\t{i+1}/{loop_count}\ttook {iteration_times[-1] / 1e6:.04} ms")

        output_start = time.perf_counter_ns()
        # Move all outputs to CPU
        for output in outputs:
            output_cpu = move_to_cpu(output)
            predictions.append(output_cpu)
        output_end = time.perf_counter_ns()
        output_time = output_end - output_start
        print(f"Moving all outputs to CPU took {output_time / 1e6:.04} ms")

    total_time_iterations = sum(iteration_times)
    total_time = total_time_iterations + output_time

    # Convert to seconds
    total_time /= 1e9
    return predictions, total_time


def benchmark_vision_torch_xla(
    model,
    model_info_name,
    optimization_level,
    trace_enabled,
    batch_size,
    loop_count,
    input_size,
    channel_size,
    data_format,
    experimental_compile,
    ttnn_perf_metrics_output_file,
    load_inputs_fn,
    extract_output_tensor_fn,
    required_pcc=0.97,
):
    """
    Benchmark a vision model using PyTorch and torch-xla.

    This function compiles a vision model with torch-xla for the Tenstorrent backend,
    and measures its inference performance. It performs warmup runs, collects inference metrics,
    and validates output correctness via PCC (Pearson Correlation Coefficient).

    Args:
        model: Loaded model instance in eval mode
        model_info_name: Model name for identification and reporting
        optimization_level: tt-mlir optimization level for compilation
        trace_enabled: Whether to enable tracing
        batch_size: Batch size for inference
        loop_count: Number of inference iterations to benchmark
        input_size: Tuple of (height, width) for model inputs
        channel_size: Number of input channels
        data_format: Data precision format
        experimental_compile: Whether to use experimental compilation features
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        load_inputs_fn: Function to load a single batch of preprocessed inputs.
            Signature: fn(batch_size, data_format) -> Tensor
        extract_output_tensor_fn: Function to extract tensor from model outputs (e.g. get .logits).
        required_pcc: Minimum PCC threshold for output validation

    Returns:
        Benchmark result containing performance metrics and model information
    """

    framework_model = model

    # Generate golden output for PCC calculation (run on CPU)
    golden_input = load_inputs_fn(batch_size, data_format)
    with torch.no_grad():
        golden_output = framework_model(golden_input)
        golden_output = extract_output_tensor_fn(golden_output)

    # Set XLA compilation options
    options = {
        "optimization_level": optimization_level,
        "export_path": MODULE_EXPORT_PATH,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "enable_trace": trace_enabled,
    }

    torch_xla.set_custom_compile_options(options)

    # Compile model
    framework_model.compile(backend="tt", options={"tt_experimental_compile": experimental_compile})

    device = torch_xla.device()

    # Clear num_batches_tracked on BatchNorm layers to avoid creating an extra
    # XLA graph for these unused buffers. In eval mode, num_batches_tracked is
    # never used, but if left as a tensor it gets transferred to the XLA device
    # and creates a separate constant sync graph.
    for m in framework_model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.num_batches_tracked = None

    if data_format == "bfloat16":
        framework_model = framework_model.to(device, dtype=torch.bfloat16)
    else:
        framework_model = framework_model.to(device)

    # Warmup
    warmup_vision_model(
        model=framework_model,
        load_inputs_fn=load_inputs_fn,
        batch_size=batch_size,
        data_format=data_format,
        device=device,
        loop_count=loop_count,
        extract_output_tensor_fn=extract_output_tensor_fn,
    )

    # Benchmark
    predictions, total_time = measure_fps_vision_model(
        model=framework_model,
        load_inputs_fn=load_inputs_fn,
        batch_size=batch_size,
        data_format=data_format,
        device=device,
        loop_count=loop_count,
        extract_output_tensor_fn=extract_output_tensor_fn,
    )

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_info_name
    model_type = "Vision, Random Input Data"
    dataset_name = "Random Data"
    num_layers = -1

    evaluation_score = 0.0
    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        evaluation_score=evaluation_score,
        batch_size=batch_size,
        data_format=data_format,
        input_size=input_size,
        channel_size=channel_size,
    )

    # Evaluate PCC
    pcc_value = compute_pcc(predictions[0], golden_output, required_pcc=required_pcc)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
        total_time=total_time,
        total_samples=total_samples,
        evaluation_score=evaluation_score,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        model_info=model_info_name,
        torch_xla_enabled=True,
        backend="tt",
        channel_size=channel_size,
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
    )

    return result
