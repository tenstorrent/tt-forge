# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import json
import socket
import pytest
from datetime import datetime

# Third-party modules
import timm
import torch

# Forge modules
import forge
from forge.verify.verify import verify
from forge._C.runtime.experimental import configure_devices, DeviceSettings

from utils import download_model

# Common constants

# Batch size configurations
BATCH_SIZE = [
    1,
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


@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
def test_efficientnet_timm(training, batch_size, input_size, channel_size, loop_count):
    """
    Test the efficientnet_timm benchmark function.
    This function is a placeholder for the actual test implementation.
    """

    if training:
        pytest.skip("Training is not supported")

    module_name = "EfficientNetTimmB0"

    # Load model
    framework_model = download_model(timm.create_model, "efficientnet_b0", pretrained=True)
    framework_model.eval()

    torch.manual_seed(1)
    input_sample = [torch.randn(batch_size, channel_size, input_size[0], input_size[1])]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=input_sample, module_name=module_name)

    # Model Verification
    verify(input_sample, framework_model, compiled_model)

    # Enable program cache on all devices
    # This feature introduces a bug in tt-metal, we enable it when we solve the issue
    # settings = DeviceSettings()
    # settings.enable_program_cache = True
    # configure_devices(device_settings=settings)

    # Run for the first time to warm up the model.
    # This is required to get accurate performance numbers.
    compiled_model(*input_sample)
    start = time.time()
    for _ in range(loop_count):
        compiled_model(*input_sample)
    end = time.time()

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    model_name = "EfificientNet Timm B0"
    model_type = "Classification, Random Input Data"
    dataset_name = "EfificientNet Timm B0, Random Data"
    num_layers = 82  # Number of layers in the model, in this case number of convolutional layers

    print("====================================================================")
    print("| Efficient Net Benchmark Results:                                        |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: : {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| Batch size: {batch_size}")
    print(f"| Input size: {input_size}")
    print("====================================================================")

    result = {
        "model": model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}",
        "config": {"model_size": "small"},
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": "f32",  # This is we call dataformat, it should be generic, too, but for this test we don't experiment with it
        # "math_fidelity": math_fidelity, @TODO - For now, we are skipping these parameters, because we are not supporting them
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "output_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "image_dimension": f"{input_size[0]}x{input_size[1]}",
        "perf_analysis": False,
        "training": training,
        "measurements": [
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_samples",
                "value": total_samples,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
        ],
        "device_info": {
            "device_name": "",
            "galaxy": False,
            "arch": "",
            "chips": 1,
        },
        "device_ip": None,
    }

    return result


def benchmark(config: dict):
    """
    Run the efficientnet_timm benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]

    return test_efficientnet_timm(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
    )
