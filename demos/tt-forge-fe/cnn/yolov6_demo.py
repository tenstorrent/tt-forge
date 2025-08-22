# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# YOLOv6 Demo Script

import os
import forge
from third_party.tt_forge_models.yolov6.pytorch import ModelLoader, ModelVariant
import torch
from forge._C import DataFormat
from forge.config import CompilerConfig

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


class YoloV6Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        y, _ = self.model(x)
        # The model outputs float32, even if the input is bfloat16
        # Cast the output back to the input dtype
        return y.to(x.dtype)


def run_yolov6_demo_case(variant):

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    model = YoloV6Wrapper(model)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs], compiler_cfg=compiler_cfg)

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process and display results
    loader.post_process(output[0])

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.YOLOV6N,
        ModelVariant.YOLOV6S,
        ModelVariant.YOLOV6M,
        ModelVariant.YOLOV6L,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_yolov6_demo_case(variant)
