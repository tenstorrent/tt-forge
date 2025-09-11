# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import pytest

# Third-party modules
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from tqdm import tqdm

from benchmark.utils import measure_cpu_fps
from third_party.tt_forge_models.yolov9.pytorch.loader import ModelLoader as YOLOv9Loader
from .utils import (
    get_benchmark_metadata,
    determine_model_type_and_dataset,
    print_benchmark_results,
    create_benchmark_result,
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
def test_yolov9_torch_xla(
    training, batch_size, input_size, channel_size, loop_count, task, data_format, model_name, measure_cpu
):
    """
    This function creates a YOLOv9 model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    OPTIMIZER_ENABLED = False
    PROGRAM_CACHE_ENABLED = False
    MEMORY_LAYOUT_ANALYSIS_ENABLED = False
    TRACE_ENABLED = False
    BACKEND = "tt"

    # Create random inputs
    if task == "na":
        torch.manual_seed(1)
        inputs = [torch.randn(batch_size, channel_size, input_size[0], input_size[1])]
    else:
        raise ValueError(f"Unsupported task: {task}.")

    if data_format == "bfloat16":
        # Convert input to bfloat16
        inputs = [item.to(torch.bfloat16) for item in inputs]

    # Load model using tt_forge_models
    yolov9_loader = YOLOv9Loader()
    model_info = yolov9_loader.get_model_info().name
    if data_format == "bfloat16":
        framework_model: nn.Module = yolov9_loader.load_model(dtype_override=torch.bfloat16)
    else:
        framework_model: nn.Module = yolov9_loader.load_model()
    framework_model.eval()

    if measure_cpu:
        # Use batch size 1
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    # torch_xla compilation
    framework_model.compile(backend=BACKEND)

    # Connect the device
    device = xm.xla_device()

    # Move inputs and model to device
    framework_model = framework_model.to(device)

    # Move first input to device for verification
    device_input = inputs[0].to(device)

    with torch.no_grad():
        fw_out = framework_model(device_input)

    fw_out_cpu = [output.to("cpu") for output in fw_out]
    print(f"Model verification - Output shapes: {[out.shape for out in fw_out_cpu]}")
    print(f"Model verification - Output (first 10 values): {fw_out_cpu[0].flatten()[:10]}")

    if task == "na":
        start = time.time()
        for i in tqdm(range(loop_count)):
            with torch.no_grad():
                output = framework_model(device_input)
        end = time.time()
        evaluation_score = 0.0
    else:
        raise ValueError(f"Unsupported task: {task}.")

    total_time = end - start
    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = "YOLOv9 Torch-XLA"
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
        model_title="YOLOv9 Torch-XLA",
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
        openxla_backend=True,
        channel_size=channel_size,
    )

    return result


def benchmark(config: dict):
    """
    Run the yolov9 torch-xla benchmark.
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

    return test_yolov9_torch_xla(
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
