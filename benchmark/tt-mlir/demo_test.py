# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import ttcore, ttir


def benchmark(config: dict):

    with Context() as ctx:

        module = Module.parse(
            """
        %0 = ttir.empty() : tensor<64x128xf32>
        %1 = ttir.empty() : tensor<64x128xf32>
        %2 = ttir.empty() : tensor<64x128xf32>
        %3 = "ttir.multiply"(%0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
        """
        )
        print(str(module))

    return {
        "model": "MlirSmokeTest",
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
