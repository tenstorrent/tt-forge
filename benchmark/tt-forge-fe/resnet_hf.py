# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
print(f'import os TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
import pytest
print(f'import pytest TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
import time
print(f'import time TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')

import socket
print(f'import socket TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from datetime import datetime
print(f'from datetime import datetime TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from tqdm import tqdm
print(f'from tqdm import tqdm TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')

# Third-party modules
import torch
print(f'import torch TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from transformers import ResNetForImageClassification
print(f'from transformers import ResNetForImageClassification TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from tqdm import tqdm
print(f'from tqdm import tqdm TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')

# Forge modules
import forge
print(f'import forge TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from forge.verify.config import VerifyConfig
print(f'from forge.verify.config import VerifyConfig TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from forge.verify.verify import verify
print(f'from forge.verify.verify import verify TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from forge.verify.value_checkers import AutomaticValueChecker
print(f'from forge.verify.value_checkers import AutomaticValueChecker TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from forge._C.runtime.experimental import configure_devices, DeviceSettings
print(f'from forge._C.runtime.experimental import configure_devices, DeviceSettings TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from forge._C import DataFormat
print(f'from forge._C import DataFormat TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')
from forge.config import CompilerConfig, MLIRConfig
print(f'from forge.config import CompilerConfig, MLIRConfig TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')

from benchmark.utils import (
    download_model,
    load_benchmark_dataset,
    evaluate_classification,
    measure_cpu_fps,
    get_ffe_device_arch,
)
print(f'from benchmark.utils import download_model, load_benchmark_dataset, evaluate_classification, measure_cpu_fps, get_ffe_device_arch TT_METAL_HOME: {os.environ["TT_METAL_HOME"]}')

# Common constants

# Machine learning task
TASK = [
    "classification",
]

# Target evaluation score for classification tasks, given as a percentage (e.g., 75.0 for 75%)
EVALUATION_SCORE_TARGET = 75.0

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

VARIANTS = [
    "microsoft/resnet-50",
]


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANTS)
@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
def test_resnet_hf(
    variant,
    channel_size,
    input_size,
    batch_size,
    loop_count,
    task,
    data_format,
    training,
    model_name,
    measure_cpu,
):

    if training:
        pytest.skip("Training is not supported")

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
        # Random data
        inputs = [torch.rand(batch_size, channel_size, *input_size)]
    else:
        raise ValueError(f"Unsupported task: {task}.")

    if data_format == "bfloat16":
        inputs = [item.to(torch.bfloat16) for item in inputs]

    # Load framework model
    if data_format == "bfloat16":
        framework_model = download_model(
            ResNetForImageClassification.from_pretrained, variant, return_dict=False, torch_dtype=torch.bfloat16
        )
        framework_model = framework_model.to(dtype=torch.bfloat16)
    else:
        framework_model = download_model(ResNetForImageClassification.from_pretrained, variant, return_dict=False)

    if measure_cpu:
        # Use batch size 1
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    # Compile model
    compiler_cfg = CompilerConfig()
    if data_format == "bfloat16":
        compiler_cfg.default_df_override = DataFormat.Float16_b

    # Turn on MLIR optimizations.
    OPTIMIZER_ENABLED = True
    MEMORY_LAYOUT_ANALYSIS_ENABLED = True
    TRACE_ENABLED = False
    compiler_cfg.mlir_config = (
        MLIRConfig()
        .set_enable_optimizer(OPTIMIZER_ENABLED)
        .set_enable_fusing(True)
        .set_enable_fusing_conv2d_with_multiply_pattern(True)
        .set_enable_memory_layout_analysis(MEMORY_LAYOUT_ANALYSIS_ENABLED)
    )

    # TODO: Remove this line when the issue with reinitialization is resolved.
    os.environ["TT_METAL_FORCE_REINIT"] = "1"

    # Enable Forge FE optimizations
    compiler_cfg.enable_optimization_passes = True
    compiled_model = forge.compile(framework_model, inputs[0], compiler_cfg=compiler_cfg)
    compiled_model.save(f"{model_name}.ttnn")

    # Enable program cache on all devices
    PROGRAM_CACHE_ENABLED = True
    settings = DeviceSettings()
    settings.enable_program_cache = PROGRAM_CACHE_ENABLED
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
    device_arch = get_ffe_device_arch()
    total_time = end - start
    total_samples = batch_size * loop_count

    samples_per_sec = total_samples / total_time
    full_model_name = "Resnet 50 HF"
    model_type = "Classification"
    if task == "classification":
        model_type += ", ImageNet-1K"
        dataset_name = "ImageNet-1K"
    elif task == "na":
        model_type += ", Random Input Data"
        dataset_name = full_model_name + ", Random Data"
    else:
        raise ValueError(f"Unsupported task: {task}.")
    num_layers = 50  # Number of layers in the model, in this case 50 layers

    print("====================================================================")
    print("| Resnet Benchmark Results:                                        |")
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
    print(f"| Evaluation score: {evaluation_score}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {data_format}")
    print(f"| Input size: {input_size}")
    print(f"| Channel size: {channel_size}")
    print("====================================================================")

    if task == "classification":
        if evaluation_score <= EVALUATION_SCORE_TARGET:
            raise ValueError(f"Evaluation score {evaluation_score} is less than the target {EVALUATION_SCORE_TARGET}.")
    elif task == "na":
        fw_out = framework_model(inputs[-1])[0]
        co_out = co_out.to("cpu")
        AutomaticValueChecker(pcc=0.95).check(fw_out=fw_out, co_out=co_out)
    else:
        raise ValueError(f"Unsupported task: {task}.")

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
                "target": -1,
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "evaluation_score",
                "value": evaluation_score,
                "target": EVALUATION_SCORE_TARGET,  # This is the target evaluation score.
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
    """
    Run the resnet benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """
    variant = VARIANTS[0]
    channel_size = CHANNEL_SIZE[0]
    input_size = INPUT_SIZE[0]
    batch_size = config["batch_size"]
    loop_count = config["loop_count"]
    task = config.get("task", "na")
    data_format = config.get("data_format", DATA_FORMAT[0])
    training = config.get("training", False)
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]

    return test_resnet_hf(
        variant=variant,
        channel_size=channel_size,
        input_size=input_size,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        training=training,
        model_name=model_name,
        measure_cpu=measure_cpu,
    )
