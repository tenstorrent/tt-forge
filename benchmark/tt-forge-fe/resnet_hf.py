# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import pytest
import time
import socket
from datetime import datetime
from tqdm import tqdm

# Third-party modules
import torch
from transformers import ResNetForImageClassification

# Forge modules
import forge
from forge.verify.value_checkers import AutomaticValueChecker
from forge._C.runtime.experimental import configure_devices, DeviceSettings
from forge._C import DataFormat
from forge.config import CompilerConfig, MLIRConfig

from benchmark.utils import download_model, load_benchmark_dataset, evaluate_classification

TASK = [
    "classification",
]

BATCH_SIZE = [
    1,
]

DATA_FORMAT = [
    "bfloat16",
]

INPUT_SIZE = [
    (224, 224),
]

CHANNEL_SIZE = [
    3,
]

LOOP_COUNT = [1, 2, 4, 8, 16, 32]

VARIANTS = [
    "microsoft/resnet-50",
]

MLIR_CONFIG_OVERRIDES = [
    None,
    "override-conv2d-config=conv2d_81.dc.conv2d.2=shard_layout#height_sharded,conv2d_97.dc.conv2d.2=shard_layout#height_sharded",
]


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANTS)
@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
@pytest.mark.parametrize("mlir_config_overrides", MLIR_CONFIG_OVERRIDES, ids=MLIR_CONFIG_OVERRIDES)
def test_resnet_hf(
    variant,
    channel_size,
    input_size,
    batch_size,
    loop_count,
    task,
    data_format,
    mlir_config_overrides,
    training,
):

    if training:
        pytest.skip("Training is not supported")

    module_name = "ResNetForImageClassificationConfig"

    if task == "classification":
        inputs, labels = load_benchmark_dataset(
            task=task,
            model_version=variant,
            dataset_name="imagenet-1k",
            split="validation",
            batch_size=batch_size,
            loop_count=loop_count,
        )
    elif task == "na":
        torch.manual_seed(1)
        inputs = [torch.rand(batch_size, channel_size, *input_size)]
    else:
        raise ValueError(f"Unsupported task: {task}.")

    if data_format == "bfloat16":
        inputs = [item.to(torch.bfloat16) for item in inputs]

    if data_format == "bfloat16":
        framework_model = download_model(
            ResNetForImageClassification.from_pretrained, variant, return_dict=False, torch_dtype=torch.bfloat16
        )
        framework_model = framework_model.to(dtype=torch.bfloat16)
    else:
        framework_model = download_model(ResNetForImageClassification.from_pretrained, variant, return_dict=False)

    compiler_cfg = CompilerConfig()
    if data_format == "bfloat16":
        compiler_cfg.default_df_override = DataFormat.Float16_b

    # Turn on MLIR optimizations.
    mlir_config = (
        MLIRConfig().set_enable_optimizer(True).set_enable_fusing(True).set_enable_memory_layout_analysis(True)
    )
    if mlir_config_overrides:
        mlir_config.set_custom_config(mlir_config_overrides)

    # Enable Forge FE optimizations
    compiler_cfg.enable_optimization_passes = True
    compiler_cfg.mlir_config = mlir_config
    compiled_model = forge.compile(framework_model, inputs[0], compiler_cfg=compiler_cfg)

    # Enable program cache on all devices
    settings = DeviceSettings()
    settings.enable_program_cache = True
    configure_devices(device_settings=settings)

    # Run for the first time to warm up the model. This is required to get accurate performance numbers.
    compiled_model(inputs[0])

    if task == "classification":
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
        start = time.time()
        for i in tqdm(range(loop_count)):
            co_out = compiled_model(inputs[0])[0]
        end = time.time()
        evaluation_score = 0.0
    else:
        raise ValueError(f"Unsupported task: {task}.")

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    model_name = "Resnet 50 HF"
    model_type = "Classification"
    if task == "classification":
        model_type += ", ImageNet-1K"
        dataset_name = "ImageNet-1K"
    elif task == "na":
        model_type += ", Random Input Data"
        dataset_name = model_name + ", Random Data"
    else:
        raise ValueError(f"Unsupported task: {task}.")
    num_layers = 50  # Number of layers in the model, in this case 50 layers

    print("====================================================================")
    print("| Resnet Benchmark Results:                                        |")
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
    variant = VARIANTS[0]
    channel_size = CHANNEL_SIZE[0]
    input_size = INPUT_SIZE[0]
    batch_size = config["batch_size"]
    loop_count = config["loop_count"]
    task = config.get("task", "na")
    data_format = config.get("data_format", DATA_FORMAT[0])
    mlir_config_overrides = config.get("mlir_config_overrides", MLIR_CONFIG_OVERRIDES[0])
    training = config.get("training", False)

    return test_resnet_hf(
        variant=variant,
        channel_size=channel_size,
        input_size=input_size,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        mlir_config_overrides=mlir_config_overrides,
        training=training,
    )
