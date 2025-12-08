# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Optional
from loguru import logger
import random
import requests
import time
import os
import shutil
import urllib
from filelock import FileLock

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoImageProcessor
from datasets import load_dataset

try:
    import paddle
except ImportError:
    paddle = None
    logger.warning("Paddle is not installed. Skipping paddle-related functionality.")
try:
    import tensorflow as tf
except ImportError:
    tf = None
    logger.warning("TensorFlow is not installed. Skipping TensorFlow-related functionality.")
try:
    from forge.module import FrameworkModule
except ImportError:
    FrameworkModule = None
    logger.warning("Forge module is not installed. Skipping Forge-related functionality.")


def align_arch(arch: str):
    for item in ["wormhole", "blackhole"]:
        if item in arch:
            return item
    return ""


def get_ffe_device_arch():

    import forge._C

    # Access TTSystem through the runtime.experimental module
    TTSystem = forge._C.runtime.experimental.TTSystem

    # Get the singleton TTSystem instance
    system = TTSystem.get_system()

    # Work with devices
    for device in system.devices:
        arch_name = str(device.arch).lower()
        return align_arch(arch_name)

    return ""


def get_jax_device_arch():

    import jax

    devices = jax.devices("tt")
    for device in devices:
        arch_name = str(device.device_kind).lower()
        return align_arch(arch_name)

    return ""


def get_xla_device_arch():

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    device = xm.xla_device_kind(device)
    arch_name = str(device).lower()
    return align_arch(arch_name)


def download_model(download_func, *args, num_retries=3, timeout=180, **kwargs):
    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)
        except (
            requests.exceptions.HTTPError,
            urllib.error.HTTPError,
            requests.exceptions.ReadTimeout,
            urllib.error.URLError,
        ):
            logger.trace("HTTP error occurred. Retrying...")
            shutil.rmtree(os.path.expanduser("~") + "/.cache", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torch/models", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torchxrayvision/models_data", ignore_errors=True)
            os.mkdir(os.path.expanduser("~") + "/.cache")
        time.sleep(timeout)

    logger.error("Failed to download the model after multiple retries.")
    assert False, "Failed to download the model after multiple retries."


def create_batch_classification(dataset, image_processor, batch_size):
    """
    Create a batch of data for benchmarking.

    Parameters:
    ----------
    dataset: iterable
        The dataset to create the batch from.
    image_processor: AutoImageProcessor, ...
        The image processor to use for processing the images.
    batch_size: int
        The batch size for the dataset.

    Returns:
    -------
    X: torch.Tensor
        The input data for the batch.
    y: torch.Tensor
        The labels for the input data.
    """
    X, y = [], []
    # For each batch, we will get batch size number of samples
    for _ in tqdm(range(batch_size), desc="Creating the batch as number of samples"):
        # Get the next sample from the dataset, the next image and its label
        item = next(dataset)
        # Fetch the image and the label, decode them and add into the batch
        image = item["image"]
        label = item["label"]
        if image.mode == "L":
            image = image.convert(mode="RGB")
        temp = image_processor(image, return_tensors="pt")["pixel_values"]
        X.append(temp)
        y.append(label)
    X = torch.cat(X)
    y = torch.tensor(y)

    return X, y


def create_input_classification(dataset, image_processor, batch_size, loop_count):
    """
    Create input data for benchmarking. Input is made of the batches.

    Parameters:
    ----------
    dataset: iterable
        The dataset to create the batch from.
    image_processor: AutoImageProcessor, ...
        The image processor to use for processing the images.
    batch_size: int
        The batch size for the dataset.
    loop_count: int
        The number of times to loop through the dataset. Number of batches to process.

    Returns:
    -------
    inputs: list
        The input data for benchmarking.
    labels: list
        The labels for the input data.
    """

    inputs, labels = [], []
    # Number of batches we want to process is loop count
    for _ in tqdm(range(loop_count)):
        X, y = create_batch_classification(dataset, image_processor, batch_size)
        inputs.append(X)
        labels.append(y)

    return inputs, labels


def load_dataset_classification(model_version, dataset_name, split, batch_size, loop_count):
    """
    Load the classification dataset for benchmarking.

    Parameters:
    ----------
    model_version: str
        The version of the model to use.
    dataset_name: str
        The name of the dataset to load.
    split: str
        The split of the dataset to load (e.g., "train", "test").
    batch_size: int
        The batch size for the dataset.
    loop_count: int
        The number of times to loop through the dataset. Number of batches to process.

    Returns:
    -------
    inputs: list
        The input data for benchmarking.
    labels: list
        The labels for the input data.
    """

    image_processor = AutoImageProcessor.from_pretrained(model_version)
    # Load the dataset as a generator
    dataset = iter(load_dataset(dataset_name, split=split, use_auth_token=True, streaming=True))
    inputs, labels = create_input_classification(dataset, image_processor, batch_size, loop_count)

    return inputs, labels


def load_benchmark_dataset(task, model_version, dataset_name, split, batch_size, loop_count):
    """
    Load the dataset for benchmarking.

    Parameters:
    ----------
    task: str
        The task to benchmark (e.g., "classification").
    model_version: str
        The version of the model to use.
    dataset_name: str
        The name of the dataset to load.
    split: str
        The split of the dataset to load (e.g., "train", "test").
    batch_size: int
        The batch size for the dataset.
    loop_count: int
        The number of times to loop through the dataset. Number of batches to process.

    Returns:
    -------
    inputs: list
        The input data for benchmarking.
    labels: list
        The labels for the input data.
    """

    if task == "classification":
        return load_dataset_classification(
            model_version=model_version,
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            loop_count=loop_count,
        )
    else:
        raise ValueError(f"Unsupported task: {task}. Supported tasks are: classification.")


def evaluate_classification(predictions, labels):
    """
    Evaluate the classification model.

    Parameters:
    ----------
    predictions: torch.Tensor
        The predictions made by the model.
    labels: torch.Tensor
        The true labels for the input data.

    Returns:
    -------
    target: float
        The accuracy of the model.
    """

    predictions = predictions.softmax(-1).argmax(-1)
    correct = (predictions == labels).sum()
    accuracy = 100.0 * correct / len(labels)
    accuracy = accuracy.item()

    return accuracy


def reset_seeds():
    random.seed(0)
    if paddle is not None:
        paddle.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if tf is not None:
        tf.random.set_seed(0)


def measure_cpu_fps(model, input, iterations=512):
    """
    Measure the fps of the model for 512 iterations and take the max value for stability.

    Parameters:
    ----------
    model: Framework model (not compiled).
    input: Framework input tensor. Should be batch size 1.

    Returns:
    -------
    fps: float
        The fps of the model.
    """
    # Measure fps
    best_time = float("inf")
    for i in range(iterations):
        start = time.perf_counter()
        model(input)
        end = time.perf_counter()
        best_time = min(best_time, end - start)
    return 1 / best_time


def aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results):
    """
    Aggregate TTNN performance metrics from multiple graph files and update results.

    Parameters:
    ----------
    ttnn_perf_metrics_output_file: str
        Base name for the perf metrics files to aggregate.
    results: dict
        Results dictionary to update with aggregated metrics. Modified in place.
    """
    import json

    # If the perf_metrics report files exist, load and aggregate results from all graphs
    base_name = os.path.basename(ttnn_perf_metrics_output_file)
    perf_files = [f for f in os.listdir(".") if f.startswith(base_name) and f.endswith(".json")]

    if perf_files:
        # Initialize aggregated metrics
        total_ops = 0
        total_shardable_ops = 0
        effectively_sharded_ops = 0
        system_memory_ops = 0
        num_graphs_with_metrics = 0

        for perf_file in sorted(perf_files):
            with open(perf_file, "r") as f:
                perf_metrics_data = json.load(f)

            if "summary" in perf_metrics_data and isinstance(perf_metrics_data["summary"], dict):
                summary = perf_metrics_data["summary"]
                total_ops += summary.get("total_ops", 0)
                total_shardable_ops += summary.get("total_shardable_ops", 0)
                effectively_sharded_ops += summary.get("effectively_sharded_ops", 0)
                system_memory_ops += summary.get("system_memory_ops", 0)
                num_graphs_with_metrics += 1

        if num_graphs_with_metrics > 0:
            results["config"]["ttnn_total_ops"] = total_ops
            results["config"]["ttnn_total_shardable_ops"] = total_shardable_ops
            results["config"]["ttnn_effectively_sharded_ops"] = effectively_sharded_ops
            results["config"]["ttnn_system_memory_ops"] = system_memory_ops

            # Calculate aggregated percentage
            if total_shardable_ops > 0:
                results["config"]["ttnn_effectively_sharded_percentage"] = (
                    effectively_sharded_ops / total_shardable_ops
                ) * 100
            else:
                results["config"]["ttnn_effectively_sharded_percentage"] = 0.0

            results["config"]["ttnn_num_graphs"] = num_graphs_with_metrics
