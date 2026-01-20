# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import pytest
import time
import socket
import subprocess
import json
import os
from datetime import datetime

# Third-party modules
import torch
from torch import nn

# Forge modules
import forge
from forge.verify.config import VerifyConfig
from forge.verify.value_checkers import AutomaticValueChecker
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge.verify.verify import verify
from forge.config import CompilerConfig, MLIRConfig

from benchmark.utils import measure_cpu_fps, get_ffe_device_arch


# Common constants

# Batch size configurations
MNIST_BATCH_SIZE_EXP_RANGE = 7

# Input size configurations
MNIIST_INPUT_SIZE_EXP_RANGE = [5, 7]
MNIIST_INPUT_SIZE_FACTORS = [1, 3, 5, 7]

# Hidden layer size configurations
MNIST_HIDDEN_SIZE_EXP_RANGE = [5, 7]
MNIIST_HIDDEN_SIZE_FACTORS = [1, 3]

MNIST_INPUT_FEATURE_SIZE = 784  # 784 = 28 * 28, default size of MNIST image
MNIST_OUTPUT_FEATURE_SIZE = 10  # 10 classes in MNIST, default output size
MNIIST_HIDDEN_SIZE = 256  # Hidden layer size, default size

BATCH_SIZE = [
    2**i for i in range(MNIST_BATCH_SIZE_EXP_RANGE)
]  # Batch size, sizes will be 1, 2, 4, 8, 16, 32, 64, etc.
INPUT_SIZE = [  # Input size, sizes will be 1 * 2^5 = 32, 3 * 2^5 = 96, 5 * 2^5 = 160, 7 * 2^5 = 224, etc.
    factor * hidden
    for factor in MNIIST_INPUT_SIZE_FACTORS
    for hidden in [2**i for i in range(MNIIST_INPUT_SIZE_EXP_RANGE[0], MNIIST_INPUT_SIZE_EXP_RANGE[1])]
]
HIDDEN_SIZE = [  # Hidden layer size, sizes will be 1 * 2^5 = 32, 3 * 2^5 = 96, 1 * 2^6 = 64, 3 * 2^6 = 192, etc.
    factor * hidden
    for factor in MNIIST_HIDDEN_SIZE_FACTORS
    for hidden in [2**i for i in range(MNIST_HIDDEN_SIZE_EXP_RANGE[0], MNIST_HIDDEN_SIZE_EXP_RANGE[1])]
]
ARCH = []
DATAFORMAT = []
MATH_FIDELITY = []
LOOP_COUNT = [1, 2, 4, 8, 16, 32]

# Fix seed for reproducibility
torch.manual_seed(42)


# Model definition
class MNISTLinear(nn.Module):
    def __init__(
        self, input_size=MNIST_INPUT_FEATURE_SIZE, output_size=MNIST_OUTPUT_FEATURE_SIZE, hidden_size=MNIIST_HIDDEN_SIZE
    ):

        super(MNISTLinear, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return nn.functional.softmax(x)


# @TODO - For now, we are skipping these parameters, because we are not supporting them
# @pytest.mark.parametrize("math_fidelity", MATH_FIDELITY, ids=[f"math_fidelity={item}" for item in MATH_FIDELITY])
# @pytest.mark.parametrize("dataformat", DATAFORMAT, ids=[f"dataformat={item}" for item in DATAFORMAT])
# @pytest.mark.parametrize("arch", ARCH, ids=[f"arch={item}" for item in ARCH])
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE, ids=[f"hidden_size={item}" for item in HIDDEN_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
def test_mnist_linear(
    training,
    batch_size,
    input_size,
    hidden_size,
    loop_count,
    model_name,
    measure_cpu,
    # arch,
    # dataformat,
    # math_fidelity,
):

    if training:
        pytest.skip("Training is not supported")

    inputs = [torch.rand(batch_size, input_size)]

    framework_model = MNISTLinear(input_size=input_size, hidden_size=hidden_size)
    fw_out = framework_model(*inputs)

    if measure_cpu:
        # Use batch size 1
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    OPTIMIZER_ENABLED = True
    MEMORY_LAYOUT_ANALYSIS_ENABLED = True
    TRACE_ENABLED = False
    compiler_cfg = CompilerConfig()
    compiler_cfg.mlir_config = MLIRConfig().set_enable_optimizer(OPTIMIZER_ENABLED)
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=compiler_cfg)
    compiled_model.save(f"{model_name}.ttnn")

    # Enable program cache on all devices
    # TODO: enable the program cache - when the optimizer is enabled, running with program cache is not working.
    PROGRAM_CACHE_ENABLED = False
    # settings = DeviceSettings()
    # settings.enable_program_cache = PROGRAM_CACHE_ENABLED
    # configure_devices(device_settings=settings)

    # Run for the first time to warm up the model, it will be done by verify function.
    # This is required to get accurate performance numbers.
    verify(inputs, framework_model, compiled_model)
    start = time.time()
    for _ in range(loop_count):
        co_out = compiled_model(*inputs)
    end = time.time()

    co_out = [co.to("cpu") for co in co_out]
    AutomaticValueChecker().check(fw_out=fw_out, co_out=co_out[0])

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    device_arch = get_ffe_device_arch()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    full_model_name = "MNIST Linear"
    model_type = "Classification, Random Input Data"
    num_layers = 2  # Number of layers in the model, in this case 2 Linear hidden layers
    dataset_name = "MNIST, Random Data"

    print("====================================================================")
    print("| MNIST Benchmark Results:                                         |")
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
    print(f"| Input size: {input_size}")
    print(f"| Hidden size: {hidden_size}")
    print(f"| Number of layers: {num_layers}")
    print("====================================================================")

    result = {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": f"{full_model_name}_{batch_size}_{input_size}_{hidden_size}",
        "config": {
            "model_size": "small",
            "optimizer_enabled": OPTIMIZER_ENABLED,
            "program_cache_enabled": PROGRAM_CACHE_ENABLED,
            "memory_layout_analysis_enabled": MEMORY_LAYOUT_ANALYSIS_ENABLED,
            "trace_enabled": TRACE_ENABLED,
        },
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": "f32",  # This is we call dataformat, it should be generic, too, but for this test we don't experiment with it
        # "math_fidelity": math_fidelity, @TODO - For now, we are skipping these parameters, because we are not supporting them
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "output_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "image_dimension": f"{MNIST_INPUT_FEATURE_SIZE}",
        "perf_analysis": False,
        "training": training,
        "measurements": [
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_samples",
                "value": total_samples,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "cpu_fps",
                "value": cpu_fps,
                "target": -1,  # This is the target evaluation score.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
        ],
        "device_info": {
            "device_name": machine_name,
            "galaxy": False,
            "arch": device_arch,
            "chips": 1,
        },
        "device_ip": None,
    }

    return result


def benchmark(config: dict):

    training = config["training"]
    batch_size = config["batch_size"]
    loop_count = config["loop_count"]

    input_size = MNIST_INPUT_FEATURE_SIZE if config["input_size"] is None else config["input_size"]
    hidden_size = MNIIST_HIDDEN_SIZE if config["hidden_size"] is None else config["hidden_size"]
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]

    return test_mnist_linear(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        hidden_size=hidden_size,
        loop_count=loop_count,
        model_name=model_name,
        measure_cpu=measure_cpu,
    )
