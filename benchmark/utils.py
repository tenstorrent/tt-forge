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
from ultralytics.nn.tasks import DetectionModel

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


# def get_cache_dir() -> str:
#     """Get models cache directory from env var or use local default."""
#     cache_dir = os.environ.get("FORGE_MODELS_CACHE")
#     if not cache_dir:
#         cache_dir = os.path.join(os.getcwd(), ".forge_models_cache")
#         os.makedirs(cache_dir, exist_ok=True)
#     return cache_dir


# def default_loader(path: str):
#     """Load model with PyTorch."""
#     try:
#         return torch.load(path, map_location="cpu")
#     except Exception as e:
#         print(f"PyTorch loading error: {e}")
#         return None


# def yolov5_loader(path: str, variant: str = "ultralytics/yolov5"):
#     try:
#         model = torch.hub.load(variant, "custom", path=path)
#         return model
#     except Exception as e:
#         print(f"YOLOv5 loading error: {e}")
#         return None


# def fetch_model(
#     model_name: str,
#     url: str,
#     loader: Optional[Callable] = default_loader,
#     max_retries: int = 3,
#     timeout: int = 30,
#     **kwargs: Any,
# ) -> FrameworkModule:
#     """Fetch model from URL, cache it, and load it."""

#     model_file = model_name + ".pt"

#     model_path = os.path.join(get_cache_dir(), model_file)

#     # Download if needed
#     if not os.path.exists(model_path):
#         os.makedirs(os.path.dirname(model_path), exist_ok=True)
#         lock_path = model_path + ".lock"
#         lock = FileLock(lock_path)

#         with lock:
#             # Check again after acquiring the lock to handle concurrent processes
#             if not os.path.exists(model_path):
#                 for attempt in range(1, max_retries + 1):
#                     try:
#                         print(f"Downloading {model_name}, attempt {attempt}/{max_retries}...")
#                         response = requests.get(url, timeout=timeout)
#                         response.raise_for_status()

#                         # Write to temporary file first
#                         temp_path = model_path + ".tmp"
#                         with open(temp_path, "wb") as f:
#                             f.write(response.content)

#                         # Atomic rename after successful download
#                         try:
#                             os.rename(temp_path, model_path)
#                         except OSError as e:
#                             print(f"Error during rename: {e}")
#                             # Clean up the temp file
#                             if os.path.exists(temp_path):
#                                 os.remove(temp_path)
#                             raise  # Let this be caught by the outer exception handler

#                         break  # Successfully downloaded and renamed

#                     except (requests.exceptions.RequestException, OSError) as e:
#                         print(f"Attempt {attempt} failed: {e}")
#                         # Clean up temp file if it exists
#                         if os.path.exists(temp_path):
#                             os.remove(temp_path)

#                         if attempt < max_retries:
#                             time.sleep(2**attempt)  # Exponential backoff
#                         else:
#                             raise RuntimeError(f"Failed to download {model_name} after {max_retries} attempts.")

#     # Load model
#     model = loader(model_path, **kwargs) if loader else None
#     return model


class YoloWrapper(torch.nn.Module):
    def __init__(self, url):
        super().__init__()
        self.model = self.load_model(url)
        self.model.model[-1].end2end = False  # Disable internal post processing steps

    def forward(self, image: torch.Tensor):
        y, x = self.model(image)
        # Post processing inside model casts output to float32, even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return (y.to(image.dtype), *x)

    def load_model(self, url):
        # Load YOLO model weights
        weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")

        # Initialize and load model
        model = DetectionModel(cfg=weights["model"].yaml)
        model.load_state_dict(weights["model"].float().state_dict())
        model.eval()

        return model


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
