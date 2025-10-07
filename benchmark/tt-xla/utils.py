# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import socket
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections.abc import Sequence
import torch
import torch_xla.runtime as xr
from tt_torch import parse_compiled_artifacts_from_cache_to_disk

xr.set_device_type("TT")
cache_dir = f"{os.getcwd()}/cachedir"
xr.initialize_cache(cache_dir)


def serialize_modules(output_prefix: str) -> None:
    """
    Serialize TT modules from in-memory cache to disk.
    Modules will be saved as {output_prefix}_ttir.mlir, {output_prefix}_ttnn.mlir and
    {output_prefix}.ttnn.

    """
    parse_compiled_artifacts_from_cache_to_disk(cache_dir, output_prefix)


def _compute_pcc_single(golden_flat: torch.Tensor, device_flat: torch.Tensor) -> float:
    """Helper to compute PCC between two flattened tensors."""
    golden_centered = golden_flat - golden_flat.mean()
    device_centered = device_flat - device_flat.mean()
    denom = golden_centered.norm() * device_centered.norm()

    if denom == 0:
        if torch.allclose(golden_flat, device_flat, rtol=1e-2, atol=1e-2):
            return 1.0
        raise AssertionError("PCC computation failed: denominator is zero but tensors are not close")

    pcc = ((golden_centered @ device_centered) / denom).item()
    # Clamp to [-1, 1] to handle floating-point precision errors
    return max(-1.0, min(1.0, pcc))


def compute_pcc(golden_output, device_output, required_pcc: float = 0.99) -> float:
    """
    Compute Pearson Correlation Coefficient between golden and device output.

    Supports single tensors or collections of tensors (e.g., YOLO multi-scale outputs).
    For collections, computes PCC for each element individually, then computes the overall
    PCC by concatenating all tensors into a single flattened tensor before comparison.

    Args:
        golden_output: Golden output tensor or collection of tensors
        device_output: Device output tensor or collection of tensors
        required_pcc: Minimum required PCC threshold

    Returns:
        Overall PCC value (computed across all concatenated tensor elements).

    Raises:
        AssertionError: If computed PCC is below required_pcc threshold
    """
    # Normalize inputs to iterables for uniform processing
    is_collection = isinstance(golden_output, Sequence) and not isinstance(golden_output, torch.Tensor)
    golden_iter = golden_output if is_collection else (golden_output,)
    device_iter = device_output if is_collection else (device_output,)

    assert len(golden_iter) == len(device_iter), (
        f"Output length mismatch: golden has {len(golden_iter)} elements, " f"device has {len(device_iter)} elements"
    )

    # Compute PCC per scale
    scale_pccs = []
    for i, (golden, device) in enumerate(zip(golden_iter, device_iter)):
        golden_flat = golden.to(torch.float32).flatten()
        device_flat = device.to(torch.float32).flatten()
        scale_pcc = _compute_pcc_single(golden_flat, device_flat)
        scale_pccs.append(scale_pcc)

        if is_collection:
            print(f"  Scale {i} (shape {golden.shape}): PCC={scale_pcc:.6f}")

    # Compute overall PCC
    golden_all = torch.cat([g.to(torch.float32).flatten() for g in golden_iter])
    device_all = torch.cat([d.to(torch.float32).flatten() for d in device_iter])
    pcc_value = _compute_pcc_single(golden_all, device_all)

    # Print results
    if is_collection:
        print(f"PCC check: Computing PCC for {len(golden_iter)} output tensors (multi-scale)")
        print(
            f"PCC check: Overall PCC={pcc_value:.6f}, Min scale PCC={min(scale_pccs):.6f}, Required PCC={required_pcc}"
        )
    else:
        print(f"PCC check: Calculated PCC={pcc_value:.6f}, Required PCC={required_pcc}")

    # Validate
    if is_collection:
        assert pcc_value >= required_pcc, (
            f"PCC comparison failed. Overall PCC={pcc_value:.6f}, "
            f"Min scale PCC={min(scale_pccs):.6f}. Required: pcc={required_pcc}"
        )
    else:
        assert (
            pcc_value >= required_pcc
        ), f"PCC comparison failed. Calculated: pcc={pcc_value:.6f}. Required: pcc={required_pcc}"

    return pcc_value


def get_benchmark_metadata() -> Dict[str, str]:
    """Get common benchmark metadata."""
    return {
        "date": datetime.now().strftime("%d-%m-%Y"),
        "machine_name": socket.gethostname(),
    }


def determine_model_type_and_dataset(task: str, full_model_name: str) -> tuple[str, str]:
    """Determine model type and dataset name based on task."""
    model_type = "Classification"

    if task == "classification":
        model_type += ", ImageNet-1K"
        dataset_name = "ImageNet-1K"
    elif task == "na":
        model_type += ", Random Input Data"
        dataset_name = full_model_name + ", Random Data"
    else:
        raise ValueError(f"Unsupported task: {task}.")

    return model_type, dataset_name


def print_benchmark_results(
    model_title: str,
    full_model_name: str,
    model_type: str,
    dataset_name: str,
    date: str,
    machine_name: str,
    total_time: float,
    total_samples: int,
    samples_per_sec: float,
    cpu_samples_per_sec: Optional[float] = None,
    evaluation_score: Optional[float] = None,
    batch_size: int = None,
    data_format: str = None,
    input_size: tuple = None,
    channel_size: int = None,
    input_sequence_length: Optional[int] = None,
) -> None:
    """Print formatted benchmark results."""
    print("====================================================================")
    print(f"| {model_title} Benchmark Results:".ljust(67) + "|")
    print("--------------------------------------------------------------------")
    print(f"| Model: {full_model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")

    if cpu_samples_per_sec is not None:
        print(f"| CPU samples per second: {cpu_samples_per_sec}")

    if evaluation_score is not None:
        print(f"| Evaluation score: {evaluation_score}")

    if batch_size is not None:
        print(f"| Batch size: {batch_size}")

    if data_format is not None:
        print(f"| Data format: {data_format}")

    if input_size is not None:
        print(f"| Input size: {input_size}")

    if channel_size is not None:
        print(f"| Channel size: {channel_size}")

    if input_sequence_length is not None:
        print(f"| Input sequence length: {input_sequence_length}")

    print("====================================================================")


def create_measurement(
    measurement_name: str,
    value: Any,
    step_name: str,
    iteration: int = 1,
    step_warm_up_num_iterations: int = 0,
    target: float = -1,
    device_power: float = -1.0,
    device_temperature: float = -1.0,
) -> Dict[str, Any]:
    """Create a single measurement dictionary."""
    return {
        "iteration": iteration,
        "step_name": step_name,
        "step_warm_up_num_iterations": step_warm_up_num_iterations,
        "measurement_name": measurement_name,
        "value": value,
        "target": target,
        "device_power": device_power,
        "device_temperature": device_temperature,
    }


def create_benchmark_result(
    full_model_name: str,
    model_type: str,
    dataset_name: str,
    num_layers: int,
    batch_size: int,
    input_size: tuple,
    loop_count: int,
    data_format: str,
    training: bool,
    total_time: float,
    total_samples: int,
    evaluation_score: Optional[float] = None,
    custom_measurements: Optional[List[Dict[str, Any]]] = None,
    optimizer_enabled: bool = False,
    program_cache_enabled: bool = False,
    memory_layout_analysis_enabled: bool = False,
    trace_enabled: bool = False,
    model_info: str = "",
    torch_xla_enabled: bool = True,
    backend: str = "tt",
    channel_size: int = 3,
    device_name: str = "",
    galaxy: bool = False,
    arch: str = "",
    chips: int = 1,
    input_is_image: bool = True,
    input_sequence_length: Optional[int] = -1,
) -> Dict[str, Any]:
    """Create a standardized benchmark result dictionary.

    Args:
        custom_measurements: List of additional measurement dictionaries to include.
                           Each measurement should have keys: measurement_name, value, and optionally
                           iteration, step_name, step_warm_up_num_iterations, target, device_power, device_temperature
    """
    # Create standard measurements
    measurements = [
        create_measurement("total_samples", total_samples, full_model_name),
        create_measurement("total_time", total_time, full_model_name),
    ]

    # Add evaluation score if provided
    if evaluation_score is not None:
        measurements.append(create_measurement("evaluation_score", evaluation_score, full_model_name))

    # Add custom measurements if provided
    if custom_measurements:
        for custom_measurement in custom_measurements:
            # Ensure required fields are present
            if "measurement_name" not in custom_measurement or "value" not in custom_measurement:
                raise ValueError("Custom measurements must include 'measurement_name' and 'value' fields")

            # Fill in default values for missing fields
            measurement = {
                "iteration": custom_measurement.get("iteration", 1),
                "step_name": custom_measurement.get("step_name", full_model_name),
                "step_warm_up_num_iterations": custom_measurement.get("step_warm_up_num_iterations", 0),
                "measurement_name": custom_measurement["measurement_name"],
                "value": custom_measurement["value"],
                "target": custom_measurement.get("target", -1),
                "device_power": custom_measurement.get("device_power", -1.0),
                "device_temperature": custom_measurement.get("device_temperature", -1.0),
            }
            measurements.append(measurement)

    config = {
        "model_size": "small",
        "optimizer_enabled": optimizer_enabled,
        "program_cache_enabled": program_cache_enabled,
        "memory_layout_analysis_enabled": memory_layout_analysis_enabled,
        "trace_enabled": trace_enabled,
        "model_info": model_info,
    }

    if torch_xla_enabled:
        config.update(
            {
                "torch_xla_enabled": torch_xla_enabled,
                "backend": backend,
            }
        )

    image_dimension = ""
    if input_is_image:
        image_dimension = f"{channel_size}x{input_size[0]}x{input_size[1]}"

    return {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(full_model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}",
        "config": config,
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": data_format,
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": input_sequence_length,
        "output_sequence_length": -1,
        "image_dimension": image_dimension,
        "perf_analysis": False,
        "training": training,
        "measurements": measurements,
        "device_info": {
            "device_name": device_name,
            "galaxy": galaxy,
            "arch": arch,
            "chips": chips,
        },
    }


def torch_xla_warmup_model(model, inputs, device, loop_count):
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


def torch_xla_measure_fps(model, inputs, device, loop_count):
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
