# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import pytest
import socket


OPTIMIZATION_LEVEL = 2
PROGRAM_CACHE_ENABLED = True
TRACE_ENABLED = False
ENABLE_WEIGHT_BFP8_CONVERSION = True

if PROGRAM_CACHE_ENABLED:
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")
cache_dir = f"{os.getcwd()}/cachedir"
xr.initialize_cache(cache_dir)

from benchmark.utils import measure_cpu_fps, get_xla_device_arch
from third_party.tt_forge_models.vgg19_unet.pytorch.loader import ModelLoader as UNetLoader
from .utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    torch_xla_measure_fps,
    torch_xla_warmup_model,
    compute_pcc,
)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants

BATCH_SIZE = [
    1,
]

DATA_FORMAT = ["bfloat16", "float32"]

INPUT_SIZE = [
    (256, 256),
]

CHANNEL_SIZE = [
    3,
]

LOOP_COUNT = [1, 2, 4, 8, 16, 32]

VARIANTS = [
    "unet_cityscapes",
]

MODULE_EXPORT_PATH = "modules"


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
    ttnn_perf_metrics_output_file,
):
    """
    This function creates a UNet model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if training:
        pytest.skip("Training is not supported")

    # Create random inputs
    torch.manual_seed(1)
    inputs = []
    for i in range(loop_count):
        inputs.append(torch.randn(batch_size, channel_size, input_size[0], input_size[1]))

    warmup_inputs = [torch.randn(batch_size, channel_size, input_size[0], input_size[1])] * loop_count

    unet_loader = UNetLoader()
    model_info = unet_loader.get_model_info().name
    framework_model: nn.Module = unet_loader.load_model()

    if data_format == "bfloat16":
        inputs = [item.to(torch.bfloat16) for item in inputs]
        warmup_inputs = [item.to(torch.bfloat16) for item in warmup_inputs]
        framework_model = framework_model.to(torch.bfloat16)
    elif data_format == "float32":
        inputs = [item.to(torch.float32) for item in inputs]
        warmup_inputs = [item.to(torch.float32) for item in warmup_inputs]
        framework_model = framework_model.to(torch.float32)

    framework_model.eval()

    if measure_cpu:
        # Use batch size 1 for CPU measurement
        cpu_input = inputs[0][0].reshape(1, *inputs[0][0].shape[0:])
        cpu_fps = measure_cpu_fps(framework_model, cpu_input)
    else:
        cpu_fps = -1.0

    # Generate golden output for PCC calculation
    with torch.no_grad():
        golden_output = framework_model(inputs[0])

    options = {
        "optimization_level": OPTIMIZATION_LEVEL,
        "export_path": MODULE_EXPORT_PATH,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "experimental_enable_weight_bfp8_conversion": ENABLE_WEIGHT_BFP8_CONVERSION,
    }

    torch_xla.set_custom_compile_options(options)

    framework_model.compile(backend="tt", options={"tt_experimental_compile": True})

    device = xm.xla_device()

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

    pcc_value = compute_pcc(predictions[0], golden_output, required_pcc=0.97)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = "UNet Torch-XLA"
    model_type = "Segmentation, Random Input Data"
    dataset_name = "UNet, Random Data"
    num_layers = -1

    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title="UNet Torch-XLA",
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        cpu_samples_per_sec=cpu_fps,
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
        custom_measurements=custom_measurements,
        optimization_level=OPTIMIZATION_LEVEL,
        program_cache_enabled=PROGRAM_CACHE_ENABLED,
        trace_enabled=TRACE_ENABLED,
        enable_weight_bfp8_conversion=ENABLE_WEIGHT_BFP8_CONVERSION,
        model_info=model_info,
        torch_xla_enabled=True,
        backend="tt",
        channel_size=channel_size,
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
    )

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
    ttnn_perf_metrics_output_file = config.get("ttnn_perf_metrics_output_file", "")

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
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
    )
