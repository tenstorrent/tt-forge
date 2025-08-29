# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import socket
import pytest
from datetime import datetime

# Third-party modules
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from tqdm import tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model

from benchmark.utils import download_model, measure_cpu_fps

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants

BATCH_SIZE = [
    1,
]

DATA_FORMAT = ["bfloat16", "float32"]

INPUT_SIZE = [
    (224, 224),
]

CHANNEL_SIZE = [
    3,
]

LOOP_COUNT = [1, 2, 4, 8, 16, 32]

VARIANTS = [
    "unet_cityscapes",
]


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANTS)
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
def test_unet_torch_xla(
    training,
    batch_size,
    input_size,
    channel_size,
    loop_count,
    data_format,
    variant,
    model_name,
    measure_cpu,
):
    """
    This function creates a UNet model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    OPTIMIZER_ENABLED = False
    PROGRAM_CACHE_ENABLED = False
    MEMORY_LAYOUT_ANALYSIS_ENABLED = False
    TRACE_ENABLED = False

    module_name = "UNetTorchXLA"

    # Create random inputs
    input_sample = torch.randn(batch_size, channel_size, input_size[0], input_size[1])

    if variant == "unet_cityscapes":
        framework_model = download_model(ptcv_get_model, variant, pretrained=False)
    else:
        raise ValueError(f"Unsupported UNet variant: {variant}")

    if data_format == "bfloat16":
        input_sample = input_sample.to(torch.bfloat16)
        framework_model = framework_model.to(torch.bfloat16)
    elif data_format == "float32":
        input_sample = input_sample.to(torch.float32)
        framework_model = framework_model.to(torch.float32)

    framework_model.eval()

    if measure_cpu:
        # Use batch size 1
        cpu_input = input_sample[0].reshape(1, *input_sample[0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    # torch_xla compilation
    framework_model.compile(backend="openxla")

    # Connect the device
    device = xm.xla_device()

    # Move inputs and model to device
    if data_format == "bfloat16":
        framework_model = framework_model.to(device, dtype=torch.bfloat16)
    else:
        framework_model = framework_model.to(device)

    input_sample = input_sample.to(device)

    # Run framework model for verification
    with torch.no_grad():
        fw_out = framework_model(input_sample)

    fw_out_cpu = fw_out.to("cpu")
    print(f"Model verification - Output shape: {fw_out_cpu.shape}")
    print(f"Model verification - Output (first 10 values): {fw_out_cpu.flatten()[:10]}")

    # Benchmark run
    start = time.time()
    for _ in tqdm(range(loop_count)):
        with torch.no_grad():
            compiled_out = framework_model(input_sample)
    end = time.time()

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    full_model_name = "UNet Torch-XLA"
    model_type = "Segmentation, Random Input Data"
    dataset_name = "UNet, Random Data"
    num_layers = -1  # Number of layers varies by variant

    print("====================================================================")
    print("| UNet Torch-XLA Benchmark Results:                               |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {full_model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| CPU samples per second: {cpu_fps}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {data_format}")
    print(f"| Input size: {input_size}")
    print(f"| Channel size: {channel_size}")
    print("====================================================================")

    result = {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(full_model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{loop_count}",
        "config": {
            "model_size": "small",
            "torch_xla_enabled": True,
            "openxla_backend": True,
            "optimizer_enabled": OPTIMIZER_ENABLED,
            "program_cache_enabled": PROGRAM_CACHE_ENABLED,
            "memory_layout_analysis_enabled": MEMORY_LAYOUT_ANALYSIS_ENABLED,
            "trace_enabled": TRACE_ENABLED,
        },
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": data_format,
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "output_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "image_dimension": f"{channel_size}x{input_size[0]}x{input_size[1]}",
        "perf_analysis": False,
        "training": training,
        "measurements": [
            {
                "iteration": 1,
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_samples",
                "value": total_samples,
                "target": -1,
                "device_power": -1.0,
                "device_temperature": -1.0,
            },
            {
                "iteration": 1,
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,
                "device_power": -1.0,
                "device_temperature": -1.0,
            },
            {
                "iteration": 1,
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "cpu_fps",
                "value": cpu_fps,
                "target": -1,
                "device_power": -1.0,
                "device_temperature": -1.0,
            },
        ],
        "device_info": {
            "device_name": "TT",
            "galaxy": False,
            "arch": "torch-xla",
            "chips": 1,
        },
        "device_ip": None,
    }

    return result


def benchmark(config: dict):
    """
    Run the unet torch-xla benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    variant = VARIANTS[0]
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]

    return test_unet_torch_xla(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        data_format=data_format,
        variant=variant,
        model_name=model_name,
        measure_cpu=measure_cpu,
    )
