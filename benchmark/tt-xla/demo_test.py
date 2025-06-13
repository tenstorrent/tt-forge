# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import jax


def benchmark(config: dict):
    jax.devices("tt")

    return {
        "model": "JaxDeviceInit",
        "model_type": "Demo",
        "run_type": "Demo_0_0_0_0",
        "config": {"model_size": "small"},
        "num_layers": 0,
        "batch_size": 0,
        "precision": "f32",
        "dataset_name": "",
        "profile_name": "",
        "input_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "output_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "image_dimension": "0x0",
        "perf_analysis": False,
        "training": False,
        "measurements": [
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": "None",
                "step_warm_up_num_iterations": 0,
                "measurement_name": "total_samples",
                "value": 1,
                "target": -1,  # This value is negative, because we don't have a target value.
                "device_power": -1.0,  # This value is negative, because we don't have a device power value.
                "device_temperature": -1.0,  # This value is negative, because we don't have a device temperature value.
            }
        ],
        "device_info": {
            "device_name": "",
            "galaxy": False,
            "arch": "",
            "chips": 1,
        },
        "device_ip": None,
    }
