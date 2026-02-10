# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import socket
import time

import jax
import numpy as np
import torch
from transformers import FlaxResNetForImageClassification

from benchmark.utils import aggregate_ttnn_perf_metrics, get_jax_device_arch, sanitize_filename
from tt_jax import serialize_compiled_artifacts_to_disk
from utils import compute_pcc, create_benchmark_result, get_benchmark_metadata, print_benchmark_results

MODULE_EXPORT_PATH = "modules"
WARMUP_STEPS = 8


def execute_and_measure_fps(compiled_model, inputs, params, loop_count):
    """
    Benchmark the model for a given number of loop_count.

    Parameters:
    ----------
    compiled_model: Callable
        The compiled JAX model to benchmark.
    inputs: list of jax.Array
        The inputs to the model.
    params: dict
        Model parameters.
    loop_count: int
        Number of batches to process.

    Returns:
    -------
    predictions: list of jax.Array
        The predictions made by the model (on CPU).
    total_time: float
        The total time taken to process the inputs in seconds.
    """
    predictions = []
    start_time = time.perf_counter_ns()

    outputs = []
    for i in range(loop_count):
        start_iteration_time = time.perf_counter_ns()

        # Model forward pass
        output = compiled_model(inputs[i], train=False, params=params)
        outputs.append(output.logits)

        end_iteration_time = time.perf_counter_ns()
        print(f"Iteration {i} took {(end_iteration_time - start_iteration_time) / 1e6:.04} ms")

    start_to_cpu_time = time.perf_counter_ns()
    # Move all outputs to CPU
    predictions = [jax.device_put(out, jax.devices("cpu")[0]) for out in outputs]
    end_to_cpu_time = time.perf_counter_ns()
    print(f"Moving all outputs to CPU took {(end_to_cpu_time - start_to_cpu_time) / 1e6:.04} ms")

    end_time = time.perf_counter_ns()
    total_time = end_time - start_time
    print(f"Total time: {total_time / 1e9:.04}s for {loop_count} iterations")
    # Convert to seconds
    total_time /= 1e9
    return predictions, total_time


def benchmark_resnet_jax(
    variant,
    input_size,
    batch_size,
    loop_count,
    data_format,
    model_name,
    optimization_level=1,
    program_cache_enabled=False,
    trace_enabled=False,
    enable_weight_bfp8_conversion=False,
    required_pcc=0.97,
):
    """
    Benchmark ResNet model using JAX and tt-jax.

    This function compiles a ResNet model with JAX for the Tenstorrent backend,
    and measures its inference performance.

    Args:
        variant: HuggingFace model variant (e.g., "microsoft/resnet-50")
        input_size: Tuple of (channels, height, width) for model inputs
        batch_size: Batch size for inference
        loop_count: Number of inference iterations to benchmark
        data_format: Data format (e.g., "float32")
        model_name: Model name for identification and reporting
        optimization_level: tt-mlir optimization level for compilation
        program_cache_enabled: Whether to enable program cache
        trace_enabled: Whether to enable tracing
        enable_weight_bfp8_conversion: Whether to enable weight BFP8 conversion
        required_pcc: Minimum PCC threshold for output validation

    Returns:
        Benchmark result containing performance metrics and model information
    """

    tt_device = jax.devices("tt")[0]
    cpu_device = jax.devices("cpu")[0]

    with jax.default_device(cpu_device):
        # Instantiate the model on CPU
        framework_model = FlaxResNetForImageClassification.from_pretrained(
            variant,
            from_pt=True,
        )
        model_info = variant

    # Generate inputs (on CPU - RNG requires an unsupported SHLO op)
    inputs = []
    for i in range(loop_count):
        with jax.default_device(cpu_device):
            input_sample = jax.random.normal(
                jax.random.PRNGKey(i), (batch_size, input_size[0], input_size[1], input_size[2])
            )
        inputs.append(input_sample)

    # Generate golden output for PCC calculation (run on CPU)
    with jax.default_device(cpu_device):
        golden_input = inputs[0]
        golden_output = framework_model(golden_input, train=False, params=framework_model.params)
        golden_output = golden_output.logits

    # Move inputs to TT device
    inputs = [jax.device_put(inp, tt_device) for inp in inputs]

    # Move model parameters to TT device
    framework_model.params = jax.tree_util.tree_map(lambda x: jax.device_put(x, tt_device), framework_model.params)

    # Serialize compiled artifacts
    serialize_compiled_artifacts_to_disk(
        framework_model, inputs[0], output_prefix=f"{MODULE_EXPORT_PATH}/{model_name}", params=framework_model.params
    )

    # Compile the forward pass
    compiled_fwd = jax.jit(framework_model.__call__, static_argnames=["train"])

    # Warmup
    print("Starting warmup...")
    warmup_loop_count = min(WARMUP_STEPS, loop_count)
    warmup_inputs = inputs[:warmup_loop_count]
    execute_and_measure_fps(
        compiled_model=compiled_fwd,
        inputs=warmup_inputs,
        params=framework_model.params,
        loop_count=warmup_loop_count,
    )
    print("Warmup completed.")

    # Benchmark
    print("Starting benchmark...")
    predictions, total_time = execute_and_measure_fps(
        compiled_model=compiled_fwd,
        inputs=inputs,
        params=framework_model.params,
        loop_count=loop_count,
    )
    print("Benchmark completed.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = "Resnet 50 HF"
    model_type = "Classification, Random Input Data"
    dataset_name = full_model_name + ", Random Data"
    num_layers = 50

    print_benchmark_results(
        model_title="Resnet",
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        batch_size=batch_size,
        data_format=data_format,
        input_size=input_size,
    )

    # Evaluate PCC
    # Convert JAX arrays to PyTorch tensors for PCC comparison
    golden_tensor = torch.from_numpy(np.asarray(golden_output))
    prediction_tensor = torch.from_numpy(np.asarray(predictions[0]))
    pcc_value = compute_pcc(golden_tensor, prediction_tensor, required_pcc=required_pcc)
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
        optimization_level=optimization_level,
        program_cache_enabled=program_cache_enabled,
        trace_enabled=trace_enabled,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        model_info=model_info,
        torch_xla_enabled=False,
        device_name=socket.gethostname(),
        arch=get_jax_device_arch(),
        device_count=len(jax.devices("tt")),
    )

    return result


def test_resnet_jax(output_file):
    # Configuration
    variant = "microsoft/resnet-50"
    batch_size = 8
    loop_count = 32
    input_size = (3, 224, 224)
    data_format = "float32"
    model_name = "resnet_jax"

    # Sanitize model name for safe filesystem usage
    sanitized_model_name = sanitize_filename(model_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{sanitized_model_name}_perf_metrics"

    print(f"Running JAX benchmark for model: {model_name}")
    print(
        f"""Configuration:
    variant={variant}
    batch_size={batch_size}
    loop_count={loop_count}
    input_size={input_size}
    data_format={data_format}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_resnet_jax(
        variant=variant,
        input_size=input_size,
        batch_size=batch_size,
        loop_count=loop_count,
        data_format=data_format,
        model_name=model_name,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
