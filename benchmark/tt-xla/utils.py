# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import socket
import secrets
import re
import inspect
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from collections.abc import Sequence
import torch


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


def sanitize_model_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_").lower()
    return text or "na"


def create_model_loader(ModelLoader, num_layers: Optional[int] = None, *args, **kwargs):
    """Create a model loader with optional num_layers override.

    Returns None if num_layers is requested but the loader does not support it.
    """
    if num_layers is None:
        return ModelLoader(*args, **kwargs)
    params = inspect.signature(ModelLoader.__init__).parameters
    supports_num_layers = "num_layers" in params or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if not supports_num_layers:
        return None
    return ModelLoader(*args, num_layers=num_layers, **kwargs)


def build_xla_export_name(
    model_name: str,
    num_layers: Optional[Union[int, str]],
    batch_size: int,
    input_sequence_length: Optional[int],
) -> str:
    """Build a standardized export name for XLA benchmark runs."""
    run_id = secrets.token_hex(2)

    if num_layers is None or (isinstance(num_layers, int) and num_layers <= 0):
        layers_part = None
    else:
        layers_part = f"lyr{num_layers}"

    if not isinstance(model_name, str) and hasattr(model_name, "name"):
        model_name = model_name.name
    parts = [sanitize_model_name(model_name)]
    if layers_part:
        parts.append(layers_part)
    parts.append(f"bs{batch_size}")
    if input_sequence_length is not None and input_sequence_length > 0:
        parts.append(f"isl{input_sequence_length}")
    parts.append(f"run{run_id}")
    return "_".join(parts)


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
        # input_size is (channels, height, width)
        image_dimension = f"{input_size[0]}x{input_size[1]}x{input_size[2]}"

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
