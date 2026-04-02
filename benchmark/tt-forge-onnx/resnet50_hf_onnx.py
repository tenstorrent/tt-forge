# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import socket
import time
from datetime import datetime
from pathlib import Path

# Third-party modules
import onnx
import pytest
import torch
from loguru import logger
from tqdm import tqdm
from transformers import ResNetForImageClassification

# Forge modules
import forge
from forge._C import MLIRConfig
from forge._C.runtime.experimental import DeviceSettings, configure_devices
from forge.config import CompilerConfig
from forge.verify.value_checkers import AutomaticValueChecker

from benchmark.utils import (
    download_model,
    evaluate_classification,
    get_ffe_device_arch,
    load_benchmark_dataset,
    measure_cpu_fps,
)

# Common constants

# Machine learning task
TASK = [
    "classification",
    "na",
]

# Target evaluation score for classification tasks, given as a percentage (e.g., 75.0 for 75%)
EVALUATION_SCORE_TARGET = 75.0

# Batch size configurations
BATCH_SIZE = [
    8,
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
LOOP_COUNT = [128]

# Warmup count configurations — number of untimed iterations before the benchmark run
WARMUP_COUNT = [32]

# Model variant configurations
VARIANTS = [
    "microsoft/resnet-50",
]

# Path where the exported ONNX file is saved
_HERE = Path(__file__).resolve().parent
ONNX_PATH = _HERE / "models" / "resnet50_hf_imagenet1k.onnx"


def get_compiler_cfg(data_format: str = DATA_FORMAT[0]) -> CompilerConfig:
    """
    Build a CompilerConfig with the optimised MLIR settings for ResNet-50 HF ONNX.
    Mirrors the config used in test_resnet_benchmark.py / test_resnet_vision_benchmark.py.
    """
    # Turn on MLIR optimizations.
    OPTIMIZER_ENABLED = True
    MEMORY_LAYOUT_ANALYSIS_ENABLED = True
    TRACE_ENABLED = True
    compiler_cfg = CompilerConfig(
        mlir_config=(
            MLIRConfig()
            .set_enable_consteval(OPTIMIZER_ENABLED)
            .set_optimization_level(2)
            .set_enable_trace(TRACE_ENABLED)
            .set_enable_l1_interleaved_fallback_analysis(MEMORY_LAYOUT_ANALYSIS_ENABLED)
            .set_compute_cfg_math_fidelity(forge._C.MathFidelity.HiFi2)
            .set_enable_remove_dead_values(True)
        )
    )
    if data_format == "bfloat16":
        compiler_cfg.default_df_override = forge._C.DataFormat.Float16_b
    compiler_cfg.enable_optimization_passes = True
    return compiler_cfg


def configure_device_settings() -> None:
    """Enable program cache on all TT devices."""
    PROGRAM_CACHE_ENABLED = True
    settings = DeviceSettings()
    settings.enable_program_cache = PROGRAM_CACHE_ENABLED
    configure_devices(device_settings=settings)


class HFResNetOnnxWrapper(torch.nn.Module):
    """Wrap a HuggingFace ResNet model to return a plain logits tensor for ONNX export.

    HuggingFace models default to ``return_dict=True``, which produces a
    ``ModelOutput`` object that ``torch.onnx.export`` cannot handle.  This
    wrapper forces ``return_dict=False`` and returns only the logits tensor.
    """

    def __init__(self, hf_model: torch.nn.Module) -> None:
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values, return_dict=False)[0]


def export_hf_resnet_to_onnx(
    pytorch_model: torch.nn.Module,
    batch_size: int = BATCH_SIZE[0],
    channel_size: int = CHANNEL_SIZE[0],
    input_size: tuple = INPUT_SIZE[0],
) -> None:
    """Export the HuggingFace ResNet-50 to ONNX opset 17 (float32 graph).

    The ONNX graph is exported in float32; bfloat16 precision is applied on
    the device side via ``default_df_override=DataFormat.Float16_b`` in the
    compiler config.
    """
    wrapper = HFResNetOnnxWrapper(pytorch_model)
    wrapper.eval()

    input_shape = (batch_size, channel_size, *input_size)
    dummy = torch.zeros(*input_shape, dtype=torch.float32)
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(ONNX_PATH),
            opset_version=17,
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes=None,
        )

    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)
    logger.info(f"ONNX model saved and verified: {ONNX_PATH}")


def compile_onnx_model(
    onnx_model: onnx.ModelProto,
    sample_input: torch.Tensor,
    data_format: str = DATA_FORMAT[0],
):
    """Compile the ONNX model with Forge and enable the program cache."""
    os.environ["TT_METAL_FORCE_REINIT"] = "1"
    compiled = forge.compile(
        onnx_model,
        sample_inputs=[sample_input],
        compiler_cfg=get_compiler_cfg(data_format),
    )
    configure_device_settings()
    return compiled


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANTS)
@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("warmup_count", WARMUP_COUNT, ids=[f"warmup_count={item}" for item in WARMUP_COUNT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
def test_resnet50_hf_onnx(
    variant,
    channel_size,
    input_size,
    batch_size,
    data_format,
    loop_count,
    warmup_count,
    task,
    training,
    model_name,
    measure_cpu,
):
    """
    Benchmark ResNet-50 HuggingFace (microsoft/resnet-50) compiled via the ONNX path in bfloat16.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    full_model_name = "ResNet-50 HF"

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
        # Random data — inputs stay float32; bfloat16 is applied on-device via default_df_override.
        inputs = [torch.rand(batch_size, channel_size, *input_size)]
    else:
        raise ValueError(f"Unsupported task: {task}.")

    # Load framework model
    framework_model = download_model(
        ResNetForImageClassification.from_pretrained,
        variant,
        return_dict=False,
    )
    framework_model.eval()

    # Export to ONNX and wrap in forge.OnnxModule.
    # CPU FPS is measured after ONNX export so it reflects inference through
    # the same ONNX graph that Forge compiles for the device.
    logger.info(f"Exporting {variant} to ONNX (batch_size={batch_size})...")
    export_hf_resnet_to_onnx(framework_model, batch_size=batch_size, channel_size=channel_size, input_size=input_size)
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    onnx_module = forge.OnnxModule("ResNet50HF", onnx_model)

    if measure_cpu:
        # Use batch size 1
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(onnx_module, cpu_input)
    else:
        cpu_fps = -1.0

    # Compile ONNX model with Forge
    compiled_model = compile_onnx_model(onnx_model, inputs[0], data_format=data_format)
    compiled_model.save(f"{model_name}.ttnn")

    # Run warmup iterations to prime the device before the timed benchmark
    for i in range(warmup_count):
        compiled_model(inputs[i % len(inputs)])

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
        for _ in tqdm(range(loop_count)):
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
    model_type = "Classification"
    if task == "classification":
        model_type += ", ImageNet-1K"
        dataset_name = "ImageNet-1K"
    elif task == "na":
        model_type += ", Random Input Data"
        dataset_name = full_model_name + ", Random Data"
    else:
        raise ValueError(f"Unsupported task: {task}.")

    print("====================================================================")
    print("| ResNet-50 HF ONNX Benchmark Results:                            |")
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
    print(f"| Warmup iterations: {warmup_count}")
    print("====================================================================")

    if task == "classification":
        if evaluation_score <= EVALUATION_SCORE_TARGET:
            raise ValueError(f"Evaluation score {evaluation_score} is less than the target {EVALUATION_SCORE_TARGET}.")
    elif task == "na":
        fw_out = onnx_module(inputs[-1])[0]
        co_out = co_out.to("cpu")
        AutomaticValueChecker(pcc=0.95).check(fw_out=fw_out, co_out=co_out)
    else:
        raise ValueError(f"Unsupported task: {task}.")

    result = {
        "model": full_model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(full_model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{loop_count}",
        "config": {
            "model_size": "medium",
            "program_cache_enabled": True,
            "trace_enabled": True,
            "consteval_enabled": True,
            "optimization_level": 2,
            "math_fidelity": "HiFi2",
            "l1_interleaved_fallback_analysis": True,
            "remove_dead_values": True,
            "enable_optimization_passes": True,
        },
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
                "step_warm_up_num_iterations": warmup_count,
                "measurement_name": "total_samples",
                "value": total_samples,
                "target": -1,
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": warmup_count,
                "measurement_name": "total_time",
                "value": total_time,
                "target": -1,
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": warmup_count,
                "measurement_name": "evaluation_score",
                "value": evaluation_score,
                "target": EVALUATION_SCORE_TARGET,  # This is the target evaluation score.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": full_model_name,
                "step_warm_up_num_iterations": warmup_count,
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
    Run the ResNet-50 HF ONNX benchmark.
    This function is a placeholder for the actual benchmark implementation.
    """
    return test_resnet50_hf_onnx(
        variant=VARIANTS[0],
        channel_size=CHANNEL_SIZE[0],
        input_size=INPUT_SIZE[0],
        batch_size=config.get("batch_size", BATCH_SIZE[0]),
        data_format=config.get("data_format", DATA_FORMAT[0]),
        loop_count=config.get("loop_count", LOOP_COUNT[0]),
        warmup_count=config.get("warmup_count", WARMUP_COUNT[0]),
        task=config.get("task", "na"),
        training=config.get("training", False),
        model_name=config["model"],
        measure_cpu=config["measure_cpu"],
    )
