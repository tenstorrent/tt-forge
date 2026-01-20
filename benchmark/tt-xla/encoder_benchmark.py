# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import socket
import time
from typing import List

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from benchmark.utils import get_xla_device_arch
from utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
    move_to_cpu,
    get_export_options,
    get_mode_tag,
    find_generated_ttir_files,
    print_ttir_export_result,
    export_source_model,
    MODULE_EXPORT_PATH,
)

xr.set_device_type("TT")

WARMUP_STEPS = 3  # Number of warmup iterations before benchmarking


def run_encoder_model(
    model,
    raw_inputs: List[str],
    preprocess_fn,
    device,
    output_processor_fn,
) -> torch.Tensor:
    """Preprocess (tokenize) and encode sentences using the provided output processor.

    Args:
        model: Encoder model instance
        raw_inputs: Raw input sentences to encode
        preprocess_fn: Function to preprocess inputs (tokenization + moving to device).
            Signature: fn(sentences, device) -> dict with model input kwargs
        device: Device to run inference on
        output_processor_fn: Function to process model outputs into embeddings.
            Signature: fn(outputs, model_inputs) -> embeddings

    Returns:
        torch.Tensor: Sentence embeddings with shape [batch_size, hidden_size]
    """
    model_inputs = preprocess_fn(raw_inputs, device)
    outputs = model(**model_inputs)

    # Move all outputs to CPU before running processor_fn to avoid
    # creating extra XLA graphs for post-processing operations
    outputs = move_to_cpu(outputs)
    model_inputs = move_to_cpu(model_inputs)

    return output_processor_fn(outputs, model_inputs)


def warmup_encoder_model(model, raw_inputs, preprocess_fn, device, output_processor_fn, loop_count):
    """
    Warmup the encoder model for a given number of iterations.

    Parameters:
    ----------
    model: Callable
        The model to warmup.
    raw_inputs: List[str]
        Raw input sentences for the model.
    preprocess_fn: Callable
        Function to preprocess inputs (tokenization + device placement).
    device: torch.device
        The device to run the warmup on.
    output_processor_fn: Callable
        Function to process model outputs into embeddings.
    loop_count: int
        The number of iterations to warmup the model.
    """
    print("Warming up the device...")

    with torch.no_grad():
        for i in range(loop_count):
            output = run_encoder_model(model, raw_inputs, preprocess_fn, device, output_processor_fn)

    print("Warming up completed.")


def measure_fps_encoder_model(model, raw_inputs, preprocess_fn, device, output_processor_fn, loop_count):
    """
    Benchmark the encoder model for a given number of iterations.

    Parameters:
    ----------
    model: Callable
        The model to benchmark.
    raw_inputs: List[str]
        Raw input sentences for the model.
    preprocess_fn: Callable
        Function to preprocess inputs (tokenization + device placement).
    device: torch.device
        The device to run the benchmark on.
    output_processor_fn: Callable
        Function to process model outputs into embeddings.
    loop_count: int
        Number of iterations to process.

    Returns:
    -------
    predictions: list of torch.Tensor
        The predictions made by the model.
    total_time: float
        The total time taken to process the inputs in seconds.
    """
    print("Starting benchmark loop...")

    predictions = []
    iteration_times = []

    with torch.no_grad():
        for i in range(loop_count):
            start_time = time.perf_counter_ns()
            output = run_encoder_model(model, raw_inputs, preprocess_fn, device, output_processor_fn)
            predictions.append(output)
            end_time = time.perf_counter_ns()

            iteration_times.append(end_time - start_time)
            print(f"Iteration\t{i+1}/{loop_count}\ttook {iteration_times[-1] / 1e6:.04} ms")

    total_time = sum(iteration_times)
    # Convert to seconds
    total_time /= 1e9

    return predictions, total_time


def benchmark_encoder_torch_xla(
    model,
    model_info_name,
    optimization_level,
    trace_enabled,
    training,
    batch_size,
    input_sequence_length,
    loop_count,
    data_format,
    measure_cpu,
    experimental_compile,
    ttnn_perf_metrics_output_file,
    load_inputs_fn,
    preprocess_fn,
    output_processor_fn,
    required_pcc=0.97,
    enable_weight_bfp8_conversion=False,
    experimental_enable_permute_matmul_fusion=False,
    single_block=False,
    single_layer=False,
    model_nickname=None,
    dump_source=False,
):
    """
    Benchmark an encoder model using PyTorch and torch-xla.

    This function compiles an encoder model with torch-xla for the Tenstorrent backend,
    and measures its end-to-end inference performance. It performs warmup runs, collects metrics,
    and validates output correctness via PCC (Pearson Correlation Coefficient).

    Preprocessing (tokenization) happens per iteration inside encode_sentences(), so its time
    is included in benchmark measurements.

    Args:
        model: Loaded encoder model instance in eval mode
        model_info_name: Model name for identification and reporting
        optimization_level: tt-mlir optimization level for compilation
        trace_enabled: Whether to enable tracing
        training: Whether to run in training mode (not supported)
        batch_size: Batch size for inference
        input_sequence_length: Maximum sequence length for tokenization
        loop_count: Number of inference iterations to benchmark
        data_format: Data precision format
        measure_cpu: Whether to measure CPU baseline performance
        experimental_compile: Whether to use experimental compilation features
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        load_inputs_fn: Function to load raw inputs for the model.
            Signature: fn(batch_size) -> List[str]
        preprocess_fn: Function to preprocess inputs (tokenization + device placement).
            Signature: fn(sentences, device) -> dict with model input kwargs
        output_processor_fn: Function to process model outputs into embeddings.
            Signature: fn(outputs, model_inputs) -> embeddings.
            This function should extract hidden states and apply the appropriate pooling.
        required_pcc: Minimum PCC threshold for output validation
        enable_weight_bfp8_conversion: Whether to enable bfp8 weight conversion
        experimental_enable_permute_matmul_fusion: Whether to enable permute matmul fusion optimization
        single_block: If True, compile and export encoder block only (no embeddings)
        single_layer: If True, compile and export single layer model (full model with 1 layer)
        model_nickname: Optional nickname for the model (used in export filenames)
        dump_source: If True, dump model source code to Python file

    Returns:
        Benchmark result containing performance metrics and model information
    """
    if training:
        raise ValueError("Training is not supported for encoder benchmarks")

    # Check for conflicting flags
    if single_block and single_layer:
        raise ValueError(
            "Cannot use --generate-block-test and --generate-layer-test together. " "Run them as separate commands."
        )

    # Determine mode tag for file naming
    if model_nickname is None:
        # Sanitize model name for safe filesystem usage
        model_nickname = model_info_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    mode_tag = get_mode_tag(single_block, single_layer)

    # Get export options using shared utility
    options = get_export_options(
        model_name=model_nickname,
        mode=mode_tag,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
    )
    export_model_name = options["export_model_name"]
    print(f"Exporting to: {MODULE_EXPORT_PATH} (e.g., ttir_{export_model_name}_g0_timestamp.mlir)")
    torch_xla.set_custom_compile_options(options)

    # =========================================================================
    # SINGLE BLOCK/LAYER MODE: Compile and export only (no benchmarking)
    # =========================================================================
    if single_block or single_layer:
        mode = "single_block" if single_block else "single_layer"

        # Get hidden_size from model (handles both wrapped blocks and full models)
        hidden_size = getattr(model, "hidden_size", None)
        if hidden_size is None and hasattr(model, "config"):
            hidden_size = getattr(model.config, "hidden_size", 768)
        if hidden_size is None:
            hidden_size = 768  # Default fallback

        # Dump model to Python code if requested (before compilation)
        if dump_source:
            if single_block:
                example_input = torch.randn(batch_size, input_sequence_length, hidden_size, dtype=torch.bfloat16)
                export_source_model(export_model_name, model, example_input)
            else:
                # single_layer: create example inputs using preprocess_fn if available
                if preprocess_fn is not None and load_inputs_fn is not None:
                    raw_inputs = load_inputs_fn(batch_size)
                    example_inputs = preprocess_fn(raw_inputs, "cpu")
                    # Export with dict inputs (not directly supported, dump structure only)
                    export_source_model(export_model_name, model)
                else:
                    export_source_model(export_model_name, model)

        # Connect the device
        device = torch_xla.device()

        # Compile model
        model_on_device = model.to(device, dtype=torch.bfloat16)
        compiled_model = torch.compile(
            model_on_device, backend="tt", options={"tt_experimental_compile": experimental_compile}
        )

        with torch.no_grad():
            if single_block:
                # For single block: use hidden_states input
                hidden_states = torch.randn(batch_size, input_sequence_length, hidden_size, dtype=torch.bfloat16).to(
                    device
                )
                compiled_model(hidden_states)
                xm.mark_step()
            else:
                # For single layer: use full model input format
                raw_inputs = load_inputs_fn(batch_size)
                model_inputs = preprocess_fn(raw_inputs, device)
                compiled_model(**model_inputs)
                xm.mark_step()

        # Find and print generated TTIR files
        generated_files = find_generated_ttir_files(export_model_name)
        print_ttir_export_result(generated_files, mode, model_info_name, export_model_name)
        return {"status": "success", "mode": mode, "model": model_info_name}

    # =========================================================================
    # FULL MODEL MODE: Standard encoder benchmark
    # =========================================================================
    framework_model = model

    # Load raw inputs for all iterations
    raw_inputs = load_inputs_fn(batch_size)

    # Measure CPU performance
    if measure_cpu:
        print("Measuring CPU performance...")

        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = run_encoder_model(framework_model, raw_inputs, preprocess_fn, "cpu", output_processor_fn)
        elapsed = time.time() - start_time
        cpu_fps = (num_runs * batch_size) / elapsed
        print(f"CPU samples per second: {cpu_fps:.2f}")
    else:
        cpu_fps = -1.0

    # Generate golden output for PCC calculation
    print("Generating golden output on CPU...")
    with torch.no_grad():
        golden_output = run_encoder_model(framework_model, raw_inputs, preprocess_fn, "cpu", output_processor_fn)

    # Compile model (XLA options already set at the top)
    framework_model.compile(backend="tt", options={"tt_experimental_compile": experimental_compile})

    device = torch_xla.device()

    framework_model = framework_model.to(device)

    # Warmup
    warmup_count = min(WARMUP_STEPS, loop_count)
    warmup_encoder_model(
        model=framework_model,
        raw_inputs=raw_inputs,
        preprocess_fn=preprocess_fn,
        device=device,
        output_processor_fn=output_processor_fn,
        loop_count=warmup_count,
    )

    # Benchmark
    predictions, total_time = measure_fps_encoder_model(
        model=framework_model,
        raw_inputs=raw_inputs,
        preprocess_fn=preprocess_fn,
        device=device,
        output_processor_fn=output_processor_fn,
        loop_count=loop_count,
    )

    # Evaluate PCC
    pcc_value = compute_pcc(predictions[0], golden_output, required_pcc=required_pcc)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")
    evaluation_score = pcc_value

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_info_name
    model_type = "Encoder, Text Embedding"
    dataset_name = "Benchmark Sentences"
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
        input_sequence_length=input_sequence_length,
    )

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
        total_samples=total_samples,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        model_info=full_model_name,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        input_is_image=False,
        input_sequence_length=input_sequence_length,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
    )

    return result
