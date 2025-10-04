# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import pytest


OPTIMIZER_ENABLED = True
PROGRAM_CACHE_ENABLED = True
MEMORY_LAYOUT_ANALYSIS_ENABLED = True
TRACE_ENABLED = False

if PROGRAM_CACHE_ENABLED:
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import tt_torch
from tqdm import tqdm

from benchmark.utils import YoloWrapper, measure_cpu_fps
from .utils import (
    get_benchmark_metadata,
    determine_model_type_and_dataset,
    print_benchmark_results,
    create_benchmark_result,
    torch_xla_measure_fps,
    torch_xla_warmup_model,
)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants

# Machine learning task
TASK = [
    "na",
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
    (640, 640),
]

# Channel size configurations
CHANNEL_SIZE = [
    3,
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
def test_yolov10_torch_xla(
    training, batch_size, input_size, channel_size, loop_count, task, data_format, model_name, measure_cpu
):
    """
    This function creates a YOLOv10 model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    # Create random inputs
    if task == "na":
        torch.manual_seed(1)
        inputs = []
        for i in range(loop_count):
            inputs.append(torch.randn(batch_size, channel_size, input_size[0], input_size[1]))
    else:
        raise ValueError(f"Unsupported task: {task}.")

    warmup_inputs = [torch.randn(batch_size, channel_size, input_size[0], input_size[1])] * loop_count

    if data_format == "bfloat16":
        # Convert input to bfloat16
        inputs = [item.to(torch.bfloat16) for item in inputs]
        warmup_inputs = [item.to(torch.bfloat16) for item in warmup_inputs]

    # Load YOLO model weights, initialize and load model
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt"
    framework_model = YoloWrapper(url)

    if data_format == "bfloat16":
        # Convert model to bfloat16
        framework_model = framework_model.to(torch.bfloat16)
    framework_model.eval()

    if measure_cpu:
        # Use batch size 1
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    options = {
        "enable_optimizer": OPTIMIZER_ENABLED,
        "enable_memory_layout_analysis": MEMORY_LAYOUT_ANALYSIS_ENABLED,
        "enable_l1_interleaved": False,
        "enable_fusing_conv2d_with_multiply_pattern": True,
    }

    torch_xla.set_custom_compile_options(options)

    # torch_xla compilation
    framework_model.compile(backend="tt")

    # Connect the device
    device = xm.xla_device()

    # Move inputs and model to device
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
        # PCC
        evaluation_score = 0.0
    else:
        raise ValueError(f"Unsupported task: {task}.")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = "YOLOv10 Torch-XLA"
    model_type, dataset_name = determine_model_type_and_dataset(task, full_model_name)
    num_layers = -1

    # Create custom measurements for CPU FPS
    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title="YOLOv10 Torch-XLA",
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
        model_info="YOLOv10",
        torch_xla_enabled=True,
        backend="tt",
        channel_size=channel_size,
    )

    return result


def benchmark(config: dict):
    """
    Run the yolov10 torch-xla benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    task = config.get("task", TASK[0])
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]

    return test_yolov10_torch_xla(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        model_name=model_name,
        measure_cpu=measure_cpu,
    )
