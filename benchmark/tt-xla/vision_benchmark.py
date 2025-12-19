# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import socket
import pytest
import time

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from benchmark.utils import measure_cpu_fps, get_xla_device_arch
from utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
)

xr.set_device_type("TT")

MIN_STEPS = 16

MODULE_EXPORT_PATH = "modules"


def setup_model(model_loader, model_variant=None, data_format="bfloat16") -> tuple[torch.nn.Module, str]:
    """
    Instantiate model.

    Args:
        model_loader: Loader of the model.
        model_variant: Specific variant of the model (optional).
        data_format: Data format (bfloat16 or float32).

    Returns:
        Tuple of (model, model_info_name)
    """
    if model_variant:
        print(f"Loading model {model_loader.get_model_info(variant=model_variant).name}...")
        model = model_loader.load_model()
        model_info = model_loader.get_model_info(model_variant).name
    else:
        print(f"Loading model {model_loader.get_model_info().name}...")
        model = model_loader.load_model()
        model_info = model_loader.get_model_info().name

    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    elif data_format == "float32":
        model = model.to(torch.float32)

    model = model.eval()

    return model, model_info


def construct_inputs(
    batch_size: int,
    channel_size: int,
    input_size: tuple,
    loop_count: int,
    data_format: str = "bfloat16",
) -> list:
    """
    Construct random inputs for the model.

    Args:
        batch_size: Batch size
        channel_size: Channel size
        input_size: Input size tuple (height, width)
        loop_count: Number of loops
        data_format: Data format (bfloat16 or float32)

    Returns:
        List of input tensors
    """
    torch.manual_seed(1)
    inputs = []
    for _ in range(loop_count):
        inputs.append(torch.randn(batch_size, channel_size, *input_size))

    if data_format == "bfloat16":
        inputs = [item.to(torch.bfloat16) for item in inputs]
    elif data_format == "float32":
        inputs = [item.to(torch.float32) for item in inputs]

    return inputs


def warmup_vision_model(model, inputs, device, loop_count):
    """
    Warmup the model for a given number of loop_count.

    Parameters:
    ----------
    model: Callable
        The model to warmup.
    input: Any
        The input to the model.
    device: torch.device
        The device to run the warmup on.
    loop_count: int
        The number of loop_count to warmup the model.
    """
    print("Warming up the device...")

    if len(inputs) != loop_count:
        raise ValueError("Number of inputs must be equal to loop count.")

    with torch.no_grad():
        for i in range(loop_count):
            # Move input to device.
            device_input = inputs[i].to(device)
            # Model forward, non blocking.
            output = model(device_input)
            if hasattr(output, "logits"):
                output = output.logits

            if type(output) is torch.Tensor:
                output.to("cpu")
            elif type(output) is tuple:
                for out in output:
                    out.to("cpu")
            else:
                raise ValueError(f"Unsupported output type: {type(output)}. Supported types are: torch.Tensor, tuple.")
    print("Warming up completed.")


def measure_fps_vision_model(model, inputs, device, loop_count):
    """
    Benchmark the model for a given number of loop_count.

    Parameters:
    ----------
    model: Callable
        The model to benchmark.
    inputs: Any
        The input data for benchmarking.
    device: torch.device
        The device to run the benchmark on.
    loop_count: int
        Number of batches to process.

    Returns:
    -------
    predictions: list of Any
        The predictions made by the model.
    total_time: float
        The total time taken to process the inputs in seconds.
    """
    if len(inputs) != loop_count:
        raise ValueError("Number of inputs must be equal to loop count.")

    print("Starting benchmark loop...")

    predictions = []
    itteration_times = []
    with torch.no_grad():
        outputs = []
        for i in range(loop_count):
            start_time = time.perf_counter_ns()

            # Move input to device.
            device_input = inputs[i].to(device)

            # Model forward, non blocking.
            output = model(device_input)

            if hasattr(output, "logits"):
                output = output.logits
            outputs.append(output)

            end_time = time.perf_counter_ns()
            itteration_times.append(end_time - start_time)

            print(f"Iteration\t{i+1}/{loop_count}\ttook {itteration_times[-1] / 1e6:.04} ms")

        # Move all outputs to CPU, waits for model execution to finish.
        output_start = time.perf_counter_ns()
        for output in outputs:
            if type(output) is torch.Tensor:
                cpu_output = output.to("cpu")
                predictions.append(cpu_output)
            elif type(output) is tuple:
                cpu_output = tuple(out.to("cpu") for out in output)
                predictions.append(cpu_output)
            else:
                raise ValueError(f"Unsupported output type: {type(output)}. Supported types are: torch.Tensor, tuple.")
        output_end = time.perf_counter_ns()

        output_time = output_end - output_start
        print(f"Moving all outputs to CPU took {output_time / 1e6:.04} ms")

    total_time_itterations = sum(itteration_times)
    total_time = total_time_itterations + output_time

    # Convert to seconds
    total_time /= 1e9
    return predictions, total_time


def benchmark_vision_torch_xla(
    model_loader,
    model_variant,
    optimization_level,
    trace_enabled,
    training,
    batch_size,
    input_size,
    channel_size,
    loop_count,
    data_format,
    measure_cpu,
    experimental_compile,
    ttnn_perf_metrics_output_file,
    read_logits_fn,
    required_pcc=0.97,
):
    """
    Benchmark a vision model using PyTorch and torch-xla.

    This function loads a vision model, compiles it with torch-xla for the Tenstorrent backend,
    and measures its inference performance. It performs warmup runs, collects inference metrics,
    and validates output correctness via PCC (Pearson Correlation Coefficient).

    Args:
        model_loader: Model loader instance for loading the vision model
        model_variant: Specific variant/version of the model to benchmark
        optimization_level: tt-mlir optimization level for compilation
        training: Whether to run in training mode (not supported)
        batch_size: Batch size for inference
        input_size: Tuple of (height, width) for model inputs
        channel_size: Number of input channels
        loop_count: Number of inference iterations to benchmark
        data_format: Data precision format
        measure_cpu: Whether to measure CPU baseline performance
        experimental_compile: Whether to use experimental compilation features
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        required_pcc: Minimum PCC threshold for output validation
        read_logits_fn: Callback function to extract logits from model output

    Returns:
        Benchmark result containing performance metrics and model information
    """

    if training:
        pytest.skip("Training is not supported")

    # Construct inputs
    inputs = construct_inputs(
        batch_size=batch_size,
        channel_size=channel_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
    )

    warmup_inputs = construct_inputs(
        batch_size=batch_size,
        channel_size=channel_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
    )

    # Load model
    framework_model, model_info = setup_model(model_loader, model_variant, data_format)

    # Measure CPU performance
    if measure_cpu:
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    # Generate golden output for PCC calculation
    golden_input = inputs[0]
    with torch.no_grad():
        golden_output = framework_model(golden_input)
        golden_output = read_logits_fn(golden_output)

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

    if data_format == "bfloat16":
        framework_model = framework_model.to(device, dtype=torch.bfloat16)
    else:
        framework_model = framework_model.to(device)

    # Warmup
    warmup_vision_model(model=framework_model, inputs=warmup_inputs, device=device, loop_count=loop_count)

    # Benchmark
    predictions, total_time = measure_fps_vision_model(
        model=framework_model, inputs=inputs, device=device, loop_count=loop_count
    )

    # Evaluate PCC
    pcc_value = compute_pcc(predictions[0], golden_output, required_pcc=required_pcc)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")
    evaluation_score = 0.0

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = f"{model_info}"
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
        model_info=model_info,
        torch_xla_enabled=True,
        backend="tt",
        channel_size=channel_size,
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
    )

    return result
