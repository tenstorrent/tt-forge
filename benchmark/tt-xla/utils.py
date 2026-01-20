# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import secrets
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections.abc import Sequence
import torch


# Default export path for generated MLIR files
MODULE_EXPORT_PATH = "modules"


def get_export_options(
    model_name: str,
    mode: str = "full",
    batch_size: int = None,
    input_sequence_length: int = None,
    export_path: str = MODULE_EXPORT_PATH,
    optimization_level: int = 0,
    trace_enabled: bool = False,
    ttnn_perf_metrics_output_file: str = "",
    enable_weight_bfp8_conversion: bool = False,
    **extra_options,
) -> Dict[str, Any]:
    """
    Generate standardized export options for any benchmark.

    Args:
        model_name: Name of the model (e.g., "phi1_5", "resnet50")
        mode: Export mode tag - "blk" (block), "lyr" (layer), or "full" (default)
        batch_size: Batch size (optional, included in name if provided)
        input_sequence_length: Input sequence length (optional, included in name if provided)
        export_path: Directory to export MLIR files to
        optimization_level: XLA optimization level
        trace_enabled: Whether tracing is enabled
        ttnn_perf_metrics_output_file: Path for performance metrics output
        enable_weight_bfp8_conversion: Whether to enable BFP8 weight conversion
        **extra_options: Additional options to include

    Returns:
        Dict with export options including a unique export_model_name
    """
    run_id = secrets.token_hex(2)  # 4 hex chars, e.g., "a7f3"

    # Build export model name with optional components
    name_parts = [mode, model_name]
    if batch_size is not None:
        name_parts.append(f"bs{batch_size}")
    if input_sequence_length is not None:
        name_parts.append(f"isl{input_sequence_length}")
    name_parts.append(run_id)
    export_model_name = "_".join(name_parts)

    options = {
        "optimization_level": optimization_level,
        "export_path": export_path,
        "export_model_name": export_model_name,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "enable_trace": trace_enabled,
        "experimental_enable_weight_bfp8_conversion": enable_weight_bfp8_conversion,
    }

    # Add any extra options
    options.update(extra_options)

    return options


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

    Supports single tensors or collections of tensors (e.g., multi-scale outputs).
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
    ttft_ms: Optional[float] = None,
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

    if ttft_ms is not None:
        print(f"| TTFT (ms): {ttft_ms}")

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
    optimization_level: int = 0,
    program_cache_enabled: bool = False,
    trace_enabled: bool = False,
    enable_weight_bfp8_conversion: bool = False,
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
        "optimization_level": optimization_level,
        "program_cache_enabled": program_cache_enabled,
        "trace_enabled": trace_enabled,
        "enable_weight_bfp8_conversion": enable_weight_bfp8_conversion,
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


# ============================================================================
# Pooling functions for encoder models
# ============================================================================


def apply_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply mean pooling over hidden states.

    Args:
        hidden_states: Token embeddings with shape [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask with shape [batch_size, seq_len]

    Returns:
        Sentence embeddings with shape [batch_size, hidden_size]
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sentence_embeddings = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embeddings


def apply_last_token_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply last token pooling over hidden states.

    Args:
        hidden_states: Token embeddings with shape [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask with shape [batch_size, seq_len]

    Returns:
        Sentence embeddings with shape [batch_size, hidden_size]
    """
    # Check if left padding was used (all sequences end with non-padding tokens)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]


def move_to_cpu(data):
    """Recursively move all tensors in a data structure to CPU.

    Handles dicts, lists, tuples, and HuggingFace ModelOutput objects.
    Preserves the original data structure types.
    """
    if isinstance(data, torch.Tensor):
        return data.cpu()
    # Check for HuggingFace ModelOutput BEFORE dict (ModelOutput inherits from OrderedDict)
    # ModelOutput has to_tuple() method which plain dicts don't have
    elif hasattr(data, "to_tuple") and hasattr(data, "keys"):
        # HuggingFace ModelOutput - modify in-place to preserve the object type
        for key in list(data.keys()):
            value = data[key]
            if isinstance(value, torch.Tensor):
                data[key] = value.cpu()
            elif value is not None:
                data[key] = move_to_cpu(value)
        return data
    elif isinstance(data, dict):
        # Plain dicts - recursively move values
        return {k: move_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        moved = [move_to_cpu(item) for item in data]
        return type(data)(moved)
    return data


# ============================================================================
# Model loader utilities
# ============================================================================

import inspect


def supports_num_layers(loader_class) -> bool:
    """Check if a model loader class supports the num_layers parameter.

    Args:
        loader_class: Model loader class to check

    Returns:
        True if the loader's __init__ accepts num_layers parameter
    """
    try:
        sig = inspect.signature(loader_class.__init__)
        return "num_layers" in sig.parameters
    except (ValueError, TypeError):
        return False


# ============================================================================
# Model wrappers (re-exported from model_wrappers module)
# ============================================================================

from model_wrappers import (
    # Generic
    extract_single_block,
    # Decoder (LLaMA, Qwen, Falcon, etc.)
    DecoderBlockWrapper,
    extract_decoder_block,
    make_decoder_single_layer,
    # Encoder
    EncoderBlockWrapper,
    extract_encoder_block,
    make_encoder_single_layer,
    # Vision (ViT, Swin, SegFormer) - unified wrappers
    VisionBlockWrapper,
    extract_vision_block,
    extract_vision_single_layer_model,
)


# ============================================================================
# TTIR export utilities (shared between LLM and encoder benchmarks)
# ============================================================================

import glob
import os


def get_mode_tag(single_block: bool = False, single_layer: bool = False) -> str:
    """Get export mode tag for file naming.

    Args:
        single_block: Whether running single block mode
        single_layer: Whether running single layer mode

    Returns:
        Mode tag: "blk", "lyr", or "full"
    """
    if single_block:
        return "blk"
    elif single_layer:
        return "lyr"
    return "full"


def find_generated_ttir_files(export_model_name: str, export_path: str = None) -> List[str]:
    """Find generated TTIR files for a given export model name.

    Args:
        export_model_name: The export model name used during compilation
        export_path: Base export path (default: MODULE_EXPORT_PATH)

    Returns:
        Sorted list of matching TTIR file paths
    """
    if export_path is None:
        export_path = MODULE_EXPORT_PATH
    pattern = f"{export_path}/irs/ttir_{export_model_name}_g*.mlir"
    return sorted(glob.glob(pattern))


def print_ttir_export_result(
    generated_files: List[str],
    mode: str,
    model_name: str,
    export_model_name: str,
    export_path: str = None,
) -> None:
    """Print TTIR export results.

    Args:
        generated_files: List of generated TTIR file paths
        mode: Mode description (e.g., "single_block", "single_layer")
        model_name: Human-readable model name
        export_model_name: Export model name used in filenames
        export_path: Base export path (default: MODULE_EXPORT_PATH)
    """
    if export_path is None:
        export_path = MODULE_EXPORT_PATH

    mode_labels = {
        "single_block": "single block test",
        "single_layer": "single layer test",
    }
    mode_label = mode_labels.get(mode, mode)

    if generated_files:
        print(f"Generated {mode_label}:")
        for f in generated_files:
            print(f"  {f}")
    else:
        print(f"Generated {mode_label}: {export_model_name} (files not found in {export_path}/irs/)")


# ============================================================================
# Python code dumping utilities
# ============================================================================


def dump_model_to_python(
    model_name: str,
    model: torch.nn.Module,
    example_input: torch.Tensor,
    output_dir: str = "dumped_models",
) -> str:
    """Dump a PyTorch model to Python code using torch.export.

    Uses torch.export which handles dynamic shapes and control flow better.
    Always includes model repr first for reference.

    Args:
        model_name: Name for the exported model (used for filename and code)
        model: The PyTorch model to dump
        example_input: Example input tensor for tracing
        output_dir: Directory to save the Python file

    Returns:
        Path to the generated Python file
    """
    print(f"Exporting model {model_name} with torch.export...")

    output_path = f"{output_dir}/{model_name}.py"
    os.makedirs(output_dir, exist_ok=True)
    example_inputs = (example_input,)

    model_repr = repr(model)

    try:
        exported = torch.export.export(model, example_inputs)
        export_code = str(exported)

        # Clean up export output:
        # - Move source file annotations (# File: ...) to the end
        # - Remove all blank lines for compact output
        clean_lines = []
        file_annotations = []
        for line in export_code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('# File:'):
                file_annotations.append(stripped)
                continue
            # Skip all blank lines
            if stripped == '':
                continue
            clean_lines.append(line)
        export_code = '\n'.join(clean_lines)

        with open(output_path, "w") as f:
            f.write(f"# Exported model: {model_name}\n")
            f.write(f"# Generated using torch.export\n\n")
            f.write("# " + "=" * 70 + "\n")
            f.write("# Model structure:\n")
            f.write("# " + "=" * 70 + "\n")
            f.write(f"'''\n{model_repr}\n'''\n\n")
            f.write("# " + "=" * 70 + "\n")
            f.write("# Exported graph:\n")
            f.write("# " + "=" * 70 + "\n\n")
            f.write(export_code)
            # Add source annotations at the end
            if file_annotations:
                f.write("\n\n# " + "=" * 70 + "\n")
                f.write("# Source file annotations:\n")
                f.write("# " + "=" * 70 + "\n")
                for annotation in file_annotations:
                    f.write(f"{annotation}\n")

        print(f"Dumped model to: {output_path}")

    except Exception as e:
        print(f"Warning: torch.export failed ({e}), dumping model structure only...")
        with open(output_path, "w") as f:
            f.write(f"# Model structure: {model_name}\n")
            f.write(f"# torch.export failed: {e}\n\n")
            f.write(f"'''\n{model_repr}\n'''\n")
        print(f"Dumped model structure to: {output_path}")

    return output_path
