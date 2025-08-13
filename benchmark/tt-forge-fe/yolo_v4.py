# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import pytest
import time
import socket
from datetime import datetime

# Third-party modules
import torch

# Forge modules
import forge
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge.config import CompilerConfig, MLIRConfig
from forge._C import DataFormat

from yolov4_utils import Yolov4Wrapper, ModelLoader


# Common constants
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
    (480, 640),
]

# Channel size configurations
CHANNEL_SIZE = [
    3,
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
def test_yolo_v4(
    training,
    batch_size,
    input_size,
    channel_size,
    loop_count,
    data_format,
    model_name,
):
    """
    This function creates a basic Yolo8 model for image classification task using PyTorch.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    module_name = "YOLOv4"

    # Create random inputs
    input_sample = [
        torch.randn(
            batch_size,
            channel_size,
            input_size[0],
            input_size[1],
        )
    ]

    if data_format == "bfloat16":
        # Convert input to bfloat16
        input_sample = [input.to(torch.bfloat16) for input in input_sample]

    # Load YOLO model
    framework_model = ModelLoader.load_model()
    framework_model = Yolov4Wrapper(framework_model)

    if data_format == "bfloat16":
        # Convert model to bfloat16
        framework_model = framework_model.to(torch.bfloat16)

    # Compiler configuration
    OPTIMIZER_ENABLED = True
    MEMORY_LAYOUT_ANALYSIS_ENABLED = False
    TRACE_ENABLED = False
    compiler_config = CompilerConfig()
    # Turn on MLIR optimizations.
    compiler_config.mlir_config = (
        MLIRConfig()
        .set_enable_optimizer(OPTIMIZER_ENABLED)
        .set_enable_memory_layout_analysis(MEMORY_LAYOUT_ANALYSIS_ENABLED)
        .set_enable_fusing(True)
    )
    if data_format == "bfloat16":
        # Convert model to bfloat16
        compiler_config.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=input_sample, module_name=module_name, compiler_cfg=compiler_config
    )
    compiled_model.save(f"{model_name}.ttnn")

    # Enable program cache on all devices
    PROGRAM_CACHE_ENABLED = True
    settings = DeviceSettings()
    settings.enable_program_cache = PROGRAM_CACHE_ENABLED
    configure_devices(device_settings=settings)

    # Run for the first time to warm up the model, it will be done by verify function.
    # This is required to get accurate performance numbers.
    verify(input_sample, framework_model, compiled_model)
    start = time.time()
    for _ in range(loop_count):
        co_out = compiled_model(*input_sample)
    end = time.time()

    fw_out = framework_model(*input_sample)
    co_out = [co.to("cpu") for co in co_out]
    AutomaticValueChecker().check(fw_out=fw_out[0], co_out=co_out[0])

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    full_model_name = "YOLOv4"
    model_type = "Detection, Random Input Data"
    dataset_name = "YOLOv4, Random Data"
    num_layers = -1  # When this value is negative, it means it is not applicable

    print("====================================================================")
    print("| YOLOv4 Benchmark Results:                                        |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {full_model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {data_format}")
    print(f"| Input size: {input_size}")
    print(f"| Channel size: {channel_size}")
    print("====================================================================")

    result = {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(full_model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}",
        "config": {
            "model_size": "small",
            "optimizer_enabled": OPTIMIZER_ENABLED,
            "program_cache_enabled": PROGRAM_CACHE_ENABLED,
            "memory_layout_analysis_enabled": MEMORY_LAYOUT_ANALYSIS_ENABLED,
            "trace_enabled": TRACE_ENABLED,
        },
        "num_layers": num_layers,
        "batch_size": batch_size,
        "precision": data_format,
        # "math_fidelity": math_fidelity, @TODO - For now, we are skipping these parameters, because we are not supporting them
        "dataset_name": dataset_name,
        "profile_name": "",
        "input_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "output_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "image_dimension": f"{channel_size}x{input_size[0]}x{input_size[1]}",
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
    Run the yolo4 benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    model_name = config["model"]

    return test_yolo_v4(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        data_format=data_format,
        model_name=model_name,
    )
