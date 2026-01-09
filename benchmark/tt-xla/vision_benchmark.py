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

from benchmark.utils import measure_cpu_fps, get_xla_device_arch
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


def warmup_vision_model(model, inputs, device, loop_count, preprocess_fn, output_processor_fn, data_format):
    """
    Warmup the model for a given number of loop_count.

    Parameters:
    ----------
    model: Callable
        The model to warmup.
    inputs: list
        List of input tensors for the model.
    device: torch.device
        The device to run the warmup on.
    loop_count: int
        The number of loop_count to warmup the model.
    preprocess_fn: Callable
        Function to preprocess input (dtype conversion + device placement).
        Signature: fn(input_tensor, device, data_format) -> tensor on device.
    output_processor_fn: Callable
        Function to process model output (e.g. extract logits).
    data_format: str
        Data format (bfloat16 or float32).
    """
    print("Warming up the device...")

    if len(inputs) != loop_count:
        raise ValueError("Number of inputs must be equal to loop count.")

    with torch.no_grad():
        for i in range(loop_count):
            # Preprocess input (dtype conversion + device placement)
            device_input = preprocess_fn(inputs[i], device, data_format)
            # Model forward, non blocking.
            output = model(device_input)
            # Process output (extract logits + move to CPU)
            _ = move_to_cpu(output_processor_fn(output))

    print("Warming up completed.")


def measure_fps_vision_model(model, inputs, device, loop_count, preprocess_fn, output_processor_fn, data_format):
    """
    Benchmark the model for a given number of loop_count.

    Parameters:
    ----------
    model: Callable
        The model to benchmark.
    inputs: list
        List of input tensors for benchmarking.
    device: torch.device
        The device to run the benchmark on.
    loop_count: int
        Number of batches to process.
    preprocess_fn: Callable
        Function to preprocess input (dtype conversion + device placement).
        Signature: fn(input_tensor, device, data_format) -> tensor on device.
    output_processor_fn: Callable
        Function to process model output (e.g. extract logits).
    data_format: str
        Data format (bfloat16 or float32).

    Returns:
    -------
    predictions: list of torch.Tensor
        The predictions made by the model (on CPU).
    total_time: float
        The total time taken to process the inputs in seconds.
    """
    if len(inputs) != loop_count:
        raise ValueError("Number of inputs must be equal to loop count.")

    print("Starting benchmark loop...")

    predictions = []
    iteration_times = []
    with torch.no_grad():
        for i in range(loop_count):
            start_time = time.perf_counter_ns()

            # Preprocess input (dtype conversion + device placement)
            device_input = preprocess_fn(inputs[i], device, data_format)

            # Model forward, non blocking.
            output = model(device_input)

            # Process output (extract logits + move to CPU)
            output_cpu = move_to_cpu(output_processor_fn(output))
            predictions.append(output_cpu)

            end_time = time.perf_counter_ns()
            iteration_times.append(end_time - start_time)

            print(f"Iteration\t{i+1}/{loop_count}\ttook {iteration_times[-1] / 1e6:.04} ms")

    total_time = sum(iteration_times)

    # Convert to seconds
    total_time /= 1e9
    return predictions, total_time


def benchmark_vision_torch_xla(
    model,
    model_info_name,
    optimization_level,
    trace_enabled,
    training,
    batch_size,
    loop_count,
    input_size,
    channel_size,
    data_format,
    measure_cpu,
    experimental_compile,
    ttnn_perf_metrics_output_file,
    load_inputs_fn,
    preprocess_fn,
    output_processor_fn,
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
        training: Whether to run in training mode (not supported)
        batch_size: Batch size for inference
        loop_count: Number of inference iterations to benchmark
        input_size: Tuple of (height, width) for model inputs
        channel_size: Number of input channels
        data_format: Data precision format
        measure_cpu: Whether to measure CPU baseline performance
        experimental_compile: Whether to use experimental compilation features
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        load_inputs_fn: Function to load raw inputs for the model.
            Signature: fn(batch_size, loop_count, channel_size, input_size) -> List[Tensor]
        preprocess_fn: Function to preprocess inputs (dtype conversion + device placement).
            Signature: fn(input_tensor, device, data_format) -> tensor on device
        output_processor_fn: Function to process model outputs (e.g. extract logits).
        required_pcc: Minimum PCC threshold for output validation

    Returns:
        Benchmark result containing performance metrics and model information
    """

    if training:
        pytest.skip("Training is not supported")

    framework_model = model

    # Load inputs using provided function
    inputs = load_inputs_fn(batch_size, loop_count, channel_size, input_size)
    warmup_inputs = load_inputs_fn(batch_size, loop_count, channel_size, input_size)

    # Measure CPU performance
    if measure_cpu:
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    # Generate golden output for PCC calculation (run on CPU)
    golden_input = preprocess_fn(inputs[0], device="cpu", data_format=data_format)
    with torch.no_grad():
        golden_output = framework_model(golden_input)
        golden_output = output_processor_fn(golden_output)

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
        inputs=warmup_inputs,
        device=device,
        loop_count=loop_count,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        data_format=data_format,
    )

    # Benchmark
    predictions, total_time = measure_fps_vision_model(
        model=framework_model,
        inputs=inputs,
        device=device,
        loop_count=loop_count,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        data_format=data_format,
    )

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_info_name
    model_type = "Vision, Random Input Data"
    dataset_name = "Random Data"
    num_layers = -1

    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

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
        cpu_samples_per_sec=cpu_fps,
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
        training=training,
        total_time=total_time,
        total_samples=total_samples,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
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
