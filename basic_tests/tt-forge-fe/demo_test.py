# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import forge
from forge.verify.verify import verify


def test_eltwise_add(shape, dtype):
    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    input1 = torch.randn(shape, dtype=dtype)
    input2 = torch.randn(shape, dtype=dtype)
    inputs = [input1, input2]

    model = AddModel()
    model.eval()

    compiled_model = forge.compile(model, sample_inputs=inputs)
    verify(inputs, model, compiled_model)


def benchmark(config: dict):
    test_eltwise_add((4, 4), torch.float32)
    test_eltwise_add((6, 7), torch.float32)
    return {
        "model": "EltwiseAddDemoModel",
        "model_type": "Demo",
        "run_type": "Demo_2_0_0_1",
        "config": {"model_size": "small"},
        "num_layers": 0,
        "batch_size": 2,
        "precision": "f32",
        "dataset_name": "",
        "profile_name": "",
        "input_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "output_sequence_length": -1,  # When this value is negative, it means it is not applicable
        "image_dimension": "1x1",
        "perf_analysis": False,
        "training": False,
        "measurements": [
            {
                "iteration": 1,  # This is the number of iterations, we are running only one iteration.
                "step_name": "Add",
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
