# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time
import socket
from datetime import datetime
from typing import Dict, Any, Optional, List
import torch


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
        "input_sequence_length": -1,
        "output_sequence_length": -1,
        "image_dimension": f"{channel_size}x{input_size[0]}x{input_size[1]}",
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
    torch_xla_measure_fps(model, inputs, device, loop_count, sequential=False, print_iterations=False)
    print("Warming up completed.")


def torch_xla_measure_fps(model, inputs, device, loop_count, sequential=False, print_iterations=True):
    """
    Measure the fps of the model for a given number of loop_count

    Parameters:
    ----------
    model: torch.nn.Module
        The model to benchmark.
    inputs: list of torch.Tensor
        The input data for benchmarking.
    device: torch.device
        The device to run the benchmark on.
    loop_count: int
        Number of batches to process.

    Returns:
    -------
    predictions: list of torch.Tensor
        The predictions made by the model.
    total_time: float
        The total time taken to process the inputs in seconds.
    """
    if len(inputs) != loop_count:
        raise ValueError("Number of inputs must be equal to loop count.")

    if print_iterations:
        print("Starting benchmark loop...")

    predictions = []
    itteration_times = []
    with torch.no_grad():
        outputs = []
        for i in range(loop_count):
            start_time = time.perf_counter_ns()

            device_input = inputs[i].to(device)
            output = model(device_input)
            if hasattr(output, "logits"):
                output = output.logits

            if sequential:
                predictions.append(output.to("cpu"))
            else:
                outputs.append(output)

            end_time = time.perf_counter_ns()
            itteration_times.append(end_time - start_time)

            if print_iterations:
                print(f"Iteration\t{i+1}/{loop_count}\ttook {itteration_times[-1] / 1e6:.04} ms")

        # print all outputs and measure time
        output_start = time.perf_counter_ns()
        if not sequential:
            for output in outputs:
                cpu_output = output.to("cpu")
                predictions.append(cpu_output)
        output_end = time.perf_counter_ns()

        output_time = output_end - output_start
        if print_iterations:
            print(f"Moving all outputs to CPU took {output_time / 1e6:.04} ms")

    total_time_itterations = sum(itteration_times)
    total_time = total_time_itterations + output_time
    if print_iterations:
        print(f"Total time over itterations: {total_time_itterations / 1e9:.04} s")
        print(f"Total output wait time: {output_time / 1e6:.04} ms")
        print(f"Total time: {total_time / 1e9:.04} s")

    # Convert to seconds
    total_time /= 1e9
    return predictions, total_time
