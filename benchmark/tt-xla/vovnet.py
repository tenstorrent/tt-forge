# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import pytest
import socket


OPTIMIZER_ENABLED = False
PROGRAM_CACHE_ENABLED = True
MEMORY_LAYOUT_ANALYSIS_ENABLED = False
TRACE_ENABLED = False

if PROGRAM_CACHE_ENABLED:
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")
cache_dir = f"{os.getcwd()}/cachedir"
xr.initialize_cache(cache_dir)

from benchmark.utils import load_benchmark_dataset, evaluate_classification, measure_cpu_fps, get_xla_device_arch
from third_party.tt_forge_models.vovnet.pytorch.loader import ModelLoader as VovNetLoader, ModelVariant as VovNetVariant
from .utils import (
    get_benchmark_metadata,
    determine_model_type_and_dataset,
    print_benchmark_results,
    create_benchmark_result,
    torch_xla_measure_fps,
    torch_xla_warmup_model,
    compute_pcc,
    serialize_modules,
)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants

# Machine learning task
TASK = [
    "classification",
]

# Batch size configurations
BATCH_SIZE = [
    1,
]

# Data format configurations
DATA_FORMAT = [
    "bfloat16",
]

# Input size configurations
INPUT_SIZE = [
    (224, 224),
]

# Channel size configurations
CHANNEL_SIZE = [
    3,
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]

# Variants for image classification
VARIANTS = [
    "ese_vovnet19b_dw.ra_in1k",
]


@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
def test_vovnet_torch_xla(
    training,
    batch_size,
    input_size,
    channel_size,
    loop_count,
    variant,
    task,
    data_format,
    model_name,
    measure_cpu,
):
    """
    This function creates a VoVNet model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    if task == "classification":
        inputs, labels = load_benchmark_dataset(
            task=task,
            model_version="microsoft/resnet-50",
            dataset_name="imagenet-1k",
            split="validation",
            batch_size=batch_size,
            loop_count=loop_count,
        )
    elif task == "na":
        torch.manual_seed(1)
        inputs = []
        for i in range(loop_count):
            inputs.append(torch.randn(batch_size, channel_size, input_size[0], input_size[1]))
    else:
        raise ValueError(f"Unsupported task: {task}")

    warmup_inputs = [torch.randn(batch_size, channel_size, input_size[0], input_size[1])] * loop_count

    if data_format == "bfloat16":
        inputs = [input.to(torch.bfloat16) for input in inputs]
        warmup_inputs = [item.to(torch.bfloat16) for item in warmup_inputs]

    # Load model using tt_forge_models
    model_variant = VovNetVariant.TIMM_VOVNET19B_DW_RAIN1K
    vovnet_loader = VovNetLoader(model_variant)
    model_info = vovnet_loader.get_model_info(model_variant).name
    framework_model: nn.Module = vovnet_loader.load_model()

    if data_format == "bfloat16":
        framework_model = framework_model.to(torch.bfloat16)
    framework_model.eval()

    if measure_cpu:
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    if task == "na":
        golden_input = inputs[0]
        if data_format == "bfloat16":
            golden_input = golden_input.to(torch.bfloat16)
        with torch.no_grad():
            golden_output = framework_model(golden_input)

    options = {
        "enable_optimizer": OPTIMIZER_ENABLED,
        "enable_memory_layout_analysis": MEMORY_LAYOUT_ANALYSIS_ENABLED,
        "enable_l1_interleaved": False,
        "enable_fusing_conv2d_with_multiply_pattern": True,
    }

    torch_xla.set_custom_compile_options(options)
    framework_model.compile(backend="tt")

    device = xm.xla_device()

    if data_format == "bfloat16":
        framework_model = framework_model.to(device, dtype=torch.bfloat16)
    else:
        framework_model = framework_model.to(device)

    # Move first input to device for verification
    torch_xla_warmup_model(model=framework_model, inputs=warmup_inputs, device=device, loop_count=loop_count)

    # Benchmark
    predictions, total_time = torch_xla_measure_fps(
        model=framework_model, inputs=inputs, device=device, loop_count=loop_count
    )

    serialize_modules(f"modules/{model_name}", cache_dir)

    if task == "classification":
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        evaluation_score = evaluate_classification(predictions, labels)
    elif task == "na":
        pcc_value = compute_pcc(predictions[0], golden_output, required_pcc=0.97)
        print(f"PCC verification passed with PCC={pcc_value:.6f}")
        evaluation_score = 0.0
    else:
        raise ValueError(f"Unsupported task: {task}")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = "VoVNet Torch-XLA"
    model_type, dataset_name = determine_model_type_and_dataset(task, full_model_name)
    num_layers = -1

    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title="VoVNet Torch-XLA",
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
        optimizer_enabled=OPTIMIZER_ENABLED,
        program_cache_enabled=PROGRAM_CACHE_ENABLED,
        memory_layout_analysis_enabled=MEMORY_LAYOUT_ANALYSIS_ENABLED,
        trace_enabled=TRACE_ENABLED,
        model_info=model_info,
        torch_xla_enabled=True,
        backend="tt",
        channel_size=channel_size,
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
    )

    return result


def benchmark(config: dict):
    """
    Run the vovnet torch-xla benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]
    variant = VARIANTS[0]
    task = config["task"]
    data_format = config["data_format"]
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]

    return test_vovnet_torch_xla(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        variant=variant,
        task=task,
        data_format=data_format,
        model_name=model_name,
        measure_cpu=measure_cpu,
    )
