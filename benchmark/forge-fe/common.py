# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Common functions for the forge-fe benchmarks.
Provides pre_test and post_test functions that can modify configurations and results.
"""

from typing import Dict, Any

import random
import numpy as np
import torch
import paddle
import tensorflow as tf


def pre_test(config: Dict[str, Any], test_name: str) -> Dict[str, Any]:
    """
    Function run before any benchmark test in the forge-fe project.

    Args:
        config: The original configuration dictionary
        test_name: The name of the test being run

    Returns:
        Modified configuration dictionary
    """
    print(f"Pre-test processing for {test_name} - initialize random seeds")

    random.seed(0)
    paddle.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tf.random.set_seed(0)

    return config


def post_test(results: Dict[str, Any], config: Dict[str, Any], test_name: str) -> Dict[str, Any]:
    """
    Function run after any benchmark test in the forge-fe project.

    Args:
        results: The original results from the benchmark
        config: The configuration used for the benchmark
        test_name: The name of the test that was run

    Returns:
        Modified results dictionary
    """
    print(f"Post-test processing for {test_name}")

    return results
