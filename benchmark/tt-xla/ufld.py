# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import pytest
import socket


OPTIMIZATION_LEVEL = 2
PROGRAM_CACHE_ENABLED = True
TRACE_ENABLED = False

if PROGRAM_CACHE_ENABLED:
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

xr.set_device_type("TT")
cache_dir = f"{os.getcwd()}/cachedir"
xr.initialize_cache(cache_dir)

from benchmark.utils import measure_cpu_fps, get_xla_device_arch
from third_party.tt_forge_models.ultra_fast_lane_detection.pytorch.loader import (
    ModelLoader as UFLDLoader,
    ModelVariant as UFLDVariant,
)
from .utils import (
    get_benchmark_metadata,
    determine_model_type_and_dataset,
    print_benchmark_results,
    create_benchmark_result,
    torch_xla_measure_fps,
    torch_xla_warmup_model,
    compute_pcc,
)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants

# Machine learning task
TASK = [
    "na",  # Lane detection doesn't fit standard classification/segmentation categories
]

# Batch size configurations
BATCH_SIZE = [
    1,
]

# Data format configurations
DATA_FORMAT = [
    "bfloat16",
]

# Input size configurations (UFLD uses 288x800 for ResNet-18, 320x800 for ResNet-34)
INPUT_SIZE = [
    (288, 800),
    (320, 800),
]

# Model variant configurations
MODEL_VARIANTS = [
    UFLDVariant.TUSIMPLE_RESNET18,
    UFLDVariant.TUSIMPLE_RESNET34,
]

# Channel size configurations
CHANNEL_SIZE = [
    3,
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("model_variant", MODEL_VARIANTS, ids=[f"variant={item}" for item in MODEL_VARIANTS])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
def test_ufld_torch_xla(
    training, batch_size, model_variant, channel_size, loop_count, task, data_format, model_name, measure_cpu
):
    """
    This function creates a Ultra Fast Lane Detection model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    # Load model using tt_forge_models to get the correct input size
    model_variant = UFLDVariant.TUSIMPLE_RESNET34
    ufld_loader = UFLDLoader(model_variant)
    model_info = ufld_loader.get_model_info(model_variant).name
    input_size = ufld_loader.config.input_size

    if task == "na":
        torch.manual_seed(1)
        # Random data
        inputs = []
        for _ in range(loop_count):
            inputs.append(torch.randn(batch_size, channel_size, *input_size))
    else:
        raise ValueError(f"Unsupported task: {task}.")

    warmup_inputs = [torch.randn(batch_size, channel_size, *input_size)] * loop_count

    if data_format == "bfloat16":
        inputs = [item.to(torch.bfloat16) for item in inputs]
        warmup_inputs = [item.to(torch.bfloat16) for item in warmup_inputs]

    framework_model: nn.Module = ufld_loader.load_model()

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
            if hasattr(golden_output, "logits"):
                golden_output = golden_output.logits

    options = {
        "optimization_level": OPTIMIZATION_LEVEL,
        "export_path": "modules",
    }

    torch_xla.set_custom_compile_options(options)
    framework_model.compile(backend="tt", options={"tt_experimental_compile": True})

    device = torch_xla.device()

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

    if task == "na":
        pcc_value = compute_pcc(predictions[0], golden_output, required_pcc=0.97)
        print(f"PCC verification passed with PCC={pcc_value:.6f}")
        evaluation_score = 0.0
    else:
        raise ValueError(f"Unsupported task: {task}.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    # Determine model name and layers based on variant
    backbone_num = ufld_loader.config.backbone
    full_model_name = f"Ultra Fast Lane Detection Torch-XLA ResNet{backbone_num}"
    model_type, dataset_name = determine_model_type_and_dataset(task, full_model_name)
    num_layers = int(backbone_num)

    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title="Ultra Fast Lane Detection Torch-XLA",
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
        optimization_level=OPTIMIZATION_LEVEL,
        program_cache_enabled=PROGRAM_CACHE_ENABLED,
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
    Run the ufld torch-xla benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    task = config["task"]
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]

    # Determine model variant from config, default to ResNet18
    model_variant = config.get("model_variant", UFLDVariant.TUSIMPLE_RESNET18)
    if isinstance(model_variant, str):
        model_variant = UFLDVariant(model_variant)

    return test_ufld_torch_xla(
        training=training,
        batch_size=batch_size,
        model_variant=model_variant,
        channel_size=channel_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        model_name=model_name,
        measure_cpu=measure_cpu,
    )
