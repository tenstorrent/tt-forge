# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import time
from datetime import datetime
import socket
from tqdm import tqdm

import jax.numpy as jnp
import jax

from transformers import FlaxResNetForImageClassification
from jax import device_put
from ttxla_tools import serialize_function_to_mlir


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

TTIR_FILE_PATH = "./model_dir/tt-xla/resnet/ttir.mlir"


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
):

    if training:
        pytest.skip("Training is not supported")

    tt_device = jax.devices("tt")[0]
    with jax.default_device(jax.devices("cpu")[0]):
        # Instantiating the model seems to also run it in op by op mode once for whatver reason, also do that on the CPU
        framework_model = FlaxResNetForImageClassification.from_pretrained(
            variant,
            from_pt=True,
        )
        # Make sure to generate on the CPU, RNG requires an unsupported SHLO op
        input_sample = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, channel_size, input_size[0], input_size[1])
        )

    framework_model.params = jax.tree_util.tree_map(lambda x: device_put(x, tt_device), framework_model.params)
    input_sample = device_put(input_sample, tt_device)

    # Preserve the TTIR file
    serialize_function_to_mlir(framework_model.__call__, TTIR_FILE_PATH, input_sample)

    compiled_fwd = jax.jit(framework_model.__call__, static_argnames=["train"])

    # Warm up the model
    res = compiled_fwd(input_sample, train=False, params=framework_model.params)
    # Run the model
    start = time.time()
    for _ in tqdm(range(loop_count)):
        compiled_fwd(input_sample, train=False, params=framework_model.params)
    end = time.time()

    date = datetime.now().strftime("%d-%m-%Y")
    machine_name = socket.gethostname()
    total_time = end - start
    total_samples = batch_size * loop_count

    task = "na"
    samples_per_sec = total_samples / total_time
    model_name = "Resnet 50 HF"
    model_type = "Classification"
    if task == "na":
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
    print(f"| Total execution time: {total_time}")
    print(f"| Total samples: {total_samples}")
    print(f"| Sample per second: {samples_per_sec}")
    print(f"| Batch size: {batch_size}")
    print(f"| Data format: {data_format}")
    print(f"| Input size: {input_size}")
    print(f"| Channel size: {channel_size}")
    print("====================================================================")

    result = {
        "model": model_name,
        "model_type": model_type,
        "run_type": f"{'_'.join(model_name.split())}_{batch_size}_{'_'.join([str(dim) for dim in input_size])}_{num_layers}_{loop_count}",
        "config": {"model_size": "small"},
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
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_samples",
                "value": total_samples,
                "target": -1,
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            },
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": model_name,
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_time",
                "value": total_time,
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

    training = config["training"]
    batch_size = config["batch_size"]
    input_size = INPUT_SIZE[0]
    channel_size = CHANNEL_SIZE[0]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    variant = VARIANTS[0]

    return test_resnet(
        variant=variant,
        channel_size=channel_size,
        input_size=input_size,
        batch_size=batch_size,
        loop_count=loop_count,
        data_format=data_format,
        training=training,
    )
