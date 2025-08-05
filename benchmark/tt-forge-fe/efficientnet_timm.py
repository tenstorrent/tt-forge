# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import time
import socket
import pytest
from datetime import datetime

# Third-party modules
import timm
import torch
from tqdm import tqdm

# Forge modules
import forge
from forge.verify.value_checkers import AutomaticValueChecker
from forge.verify.verify import verify
from forge.verify.config import VerifyConfig
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge.config import CompilerConfig, MLIRConfig
from forge._C import DataFormat

from benchmark.utils import download_model, load_benchmark_dataset, evaluate_classification


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


@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
def test_efficientnet_timm(training, batch_size, input_size, channel_size, loop_count, task, data_format, model_name):
    """
    Test the efficientnet_timm benchmark function.
    This function is a placeholder for the actual test implementation.
    """

    if training:
        pytest.skip("Training is not supported")

    module_name = "EfficientNetTimmB0"

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
        inputs = [torch.randn(batch_size, channel_size, input_size[0], input_size[1])]
    else:
        raise ValueError(f"Unsupported task: {task}")

    if data_format == "bfloat16":
        # Convert input to bfloat16
        inputs = [item.to(torch.bfloat16) for item in inputs]

    # Load model
    framework_model = download_model(timm.create_model, "efficientnet_b0", pretrained=True)
    if data_format == "bfloat16":
        # Convert model to bfloat16
        framework_model = framework_model.to(torch.bfloat16)
    framework_model.eval()

    # Compiler configuration
    OPTIMIZER_ENABLED = True
    compiler_config = CompilerConfig()
    # Turn on MLIR optimizations.
    compiler_config.mlir_config = (
        MLIRConfig()
        .set_enable_optimizer(OPTIMIZER_ENABLED)
        .set_enable_memory_layout_analysis(False)
        .set_enable_fusing(True)
    )
    if data_format == "bfloat16":
        # Convert model to bfloat16
        compiler_config.default_df_override = DataFormat.Float16_b

    # Forge compile framework model
    compiled_model = forge.compile(
        framework_model, sample_inputs=inputs[0], module_name=module_name, compiler_cfg=compiler_config
    )
    compiled_model.save(f"{model_name}.ttnn")

    # Enable program cache on all devices
    PROGRAM_CACHE_ENABLED = True
    settings = DeviceSettings()
    settings.enable_program_cache = PROGRAM_CACHE_ENABLED
    configure_devices(device_settings=settings)

    if task == "classification":

        compiled_model(inputs[0])  # Warm up the model
        predictions = []
        start = time.time()
        for i in tqdm(range(loop_count)):
            co_out = compiled_model(inputs[i])[0]
            predictions.append(co_out)
        end = time.time()
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        evaluation_score = evaluate_classification(predictions, labels)

    elif task == "na":

        # Run for the first time to warm up the model, it will be done by verify function.
        # This is required to get accurate performance numbers.
        verify_cfg = VerifyConfig()
        verify_cfg.value_checker = AutomaticValueChecker()
        verify(
            [
                inputs[0],
            ],
            framework_model,
            compiled_model,
            verify_cfg=verify_cfg,
        )
        start = time.time()
        for i in tqdm(range(loop_count)):
            co_out = compiled_model(inputs[0])[0]
        end = time.time()

        fw_out = framework_model(inputs[-1])[0]
        co_out = co_out.to("cpu")[0]
        AutomaticValueChecker().check(fw_out=fw_out, co_out=co_out)

        evaluation_score = 0.0

    else:
        raise ValueError(f"Unsupported task: {task}.")

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    full_model_name = "EfficientNet Timm B0"
    model_type = "Classification"
    if task == "classification":
        model_type += ", ImageNet-1K"
        dataset_name = "ImageNet-1K"
    elif task == "na":
        model_type += ", Random Input Data"
        dataset_name = full_model_name + ", Random Data"
    else:
        raise ValueError(f"Unsupported task: {task}.")
    num_layers = 82  # Number of layers in the model, in this case number of convolutional layers

    print("====================================================================")
    print("| Efficient Net Benchmark Results:                                 |")
    print("--------------------------------------------------------------------")
    print(f"| Model: {full_model_name}")
    print(f"| Model type: {model_type}")
    print(f"| Dataset name: {dataset_name}")
    print(f"| Date: {date}")
    print(f"| Machine name: {machine_name}")
    print(f"| Total execution time: {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| Evaluation score: {evaluation_score}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {data_format}")
    print(f"| Input size: {input_size}")
    print(f"| Channel size: {channel_size}")
    print("====================================================================")

    result = {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(full_model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}",
        "config": {"model_size": "small"},
        "num_layers": num_layers,
        "batch_size": batch_size,
        "data_format": data_format,
        "optimizer_enabled": OPTIMIZER_ENABLED,
        "program_cache_enabled": PROGRAM_CACHE_ENABLED,
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
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "evaluation_score",
                "value": evaluation_score,
                "target": -1,
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
    task = config["task"]
    data_format = config["data_format"]
    model_name = config["model"]

    return test_efficientnet_timm(
        training=training,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        model_name=model_name,
    )
