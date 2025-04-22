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

import numpy as np
import torch
import paddle
import tensorflow as tf
from forge.module import FrameworkModule


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


def get_cache_dir() -> str:
    """Get models cache directory from env var or use local default."""
    cache_dir = os.environ.get("FORGE_MODELS_CACHE")
    if not cache_dir:
        cache_dir = os.path.join(os.getcwd(), ".forge_models_cache")
        os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def default_loader(path: str):
    """Load model with PyTorch."""
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"PyTorch loading error: {e}")
        return None


def yolov5_loader(path: str, variant: str = "ultralytics/yolov5"):
    try:
        model = torch.hub.load(variant, "custom", path=path)
        return model
    except Exception as e:
        print(f"YOLOv5 loading error: {e}")
        return None


def fetch_model(
    model_name: str,
    url: str,
    loader: Optional[Callable] = default_loader,
    max_retries: int = 3,
    timeout: int = 30,
    **kwargs: Any,
) -> FrameworkModule:
    """Fetch model from URL, cache it, and load it."""

    model_file = model_name + ".pt"

    model_path = os.path.join(get_cache_dir(), model_file)

    # Download if needed
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        lock_path = model_path + ".lock"
        lock = FileLock(lock_path)

        with lock:
            # Check again after acquiring the lock to handle concurrent processes
            if not os.path.exists(model_path):
                for attempt in range(1, max_retries + 1):
                    try:
                        print(f"Downloading {model_name}, attempt {attempt}/{max_retries}...")
                        response = requests.get(url, timeout=timeout)
                        response.raise_for_status()

                        # Write to temporary file first
                        temp_path = model_path + ".tmp"
                        with open(temp_path, "wb") as f:
                            f.write(response.content)

                        # Atomic rename after successful download
                        try:
                            os.rename(temp_path, model_path)
                        except OSError as e:
                            print(f"Error during rename: {e}")
                            # Clean up the temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            raise  # Let this be caught by the outer exception handler

                        break  # Successfully downloaded and renamed

                    except (requests.exceptions.RequestException, OSError) as e:
                        print(f"Attempt {attempt} failed: {e}")
                        # Clean up temp file if it exists
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                        if attempt < max_retries:
                            time.sleep(2**attempt)  # Exponential backoff
                        else:
                            raise RuntimeError(f"Failed to download {model_name} after {max_retries} attempts.")

    # Load model
    model = loader(model_path, **kwargs) if loader else None
    return model


def reset_seeds():
    random.seed(0)
    paddle.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tf.random.set_seed(0)
