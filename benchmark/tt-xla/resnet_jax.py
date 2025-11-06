# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time
from datetime import datetime
import socket
from tqdm import tqdm
from .utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
)

from tt_jax import serialize_compiled_artifacts_to_disk

import jax.numpy as jnp
import jax

from transformers import FlaxResNetForImageClassification
from jax import device_put

from benchmark.utils import get_jax_device_arch


BATCH_SIZE = [
    1,
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

DATA_FORMAT = ["float32"]

MODULE_EXPORT_PATH = "modules"


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANTS)
@pytest.mark.parametrize("channel_size", CHANNEL_SIZE, ids=[f"channel_size={item}" for item in CHANNEL_SIZE])
@pytest.mark.parametrize("input_size", INPUT_SIZE, ids=[f"input_size={item}" for item in INPUT_SIZE])
@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
@pytest.mark.parametrize("training", [False], ids=["training=False"])
def test_resnet(
    variant,
    channel_size,
    input_size,
    batch_size,
    loop_count,
    data_format,
    training,
    model_name,
):

    if training:
        pytest.skip("Training is not supported")

    OPTIMIZER_ENABLED = True
    PROGRAM_CACHE_ENABLED = False
    MEMORY_LAYOUT_ANALYSIS_ENABLED = False
    TRACE_ENABLED = False
    tt_device = jax.devices("tt")[0]
    with jax.default_device(jax.devices("cpu")[0]):
        # Instantiating the model seems to also run it in op by op mode once for whatver reason, also do that on the CPU
        framework_model = FlaxResNetForImageClassification.from_pretrained(
            variant,
            from_pt=True,
        )
        model_info = variant
        # Make sure to generate on the CPU, RNG requires an unsupported SHLO op
        input_sample = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, channel_size, input_size[0], input_size[1])
        )

    framework_model.params = jax.tree_util.tree_map(lambda x: device_put(x, tt_device), framework_model.params)
    input_sample = device_put(input_sample, tt_device)

    serialize_compiled_artifacts_to_disk(
        framework_model, input_sample, output_prefix=f"{MODULE_EXPORT_PATH}/{model_name}", params=framework_model.params
    )

    compiled_fwd = jax.jit(framework_model.__call__, static_argnames=["train"])

    # Warm up the model
    compiled_fwd(input_sample, train=False, params=framework_model.params)
    # Run the model
    start = time.time()
    for _ in tqdm(range(loop_count)):
        compiled_fwd(input_sample, train=False, params=framework_model.params)
    end = time.time()

    total_time = end - start
    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    task = "na"
    full_model_name = "Resnet 50 HF"
    model_type = "Classification, Random Input Data"
    dataset_name = full_model_name + ", Random Data"
    num_layers = 50

    print_benchmark_results(
        model_title="Resnet",
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
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
        optimizer_enabled=OPTIMIZER_ENABLED,
        program_cache_enabled=PROGRAM_CACHE_ENABLED,
        memory_layout_analysis_enabled=MEMORY_LAYOUT_ANALYSIS_ENABLED,
        trace_enabled=TRACE_ENABLED,
        model_info=model_info,
        torch_xla_enabled=False,
        channel_size=channel_size,
        device_name=socket.gethostname(),
        arch=get_jax_device_arch(),
    )

    return result


def benchmark(config: dict):

    training = config["training"]
    batch_size = config["batch_size"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    variant = VARIANTS[0]
    model_name = config["model"]

    return test_resnet(
        variant=variant,
        channel_size=channel_size,
        input_size=input_size,
        batch_size=batch_size,
        loop_count=loop_count,
        data_format=data_format,
        training=training,
        model_name=model_name,
    )
