# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# YOLOv9 Demo Script

import sys
import forge
from third_party.tt_forge_models.yolov9.pytorch import ModelLoader
import torch
from forge._C import DataFormat
from forge.config import CompilerConfig


class YoloWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.model[-1].end2end = False  # Disable internal post processing steps

    def forward(self, image: torch.Tensor):
        y, x = self.model(image)
        # Post processing inside model casts output to float32, even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return (y.to(image.dtype), *x)


def run_yolov9_demo_case():

    # Load Model and inputs
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    framework_model = YoloWrapper(model)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=[inputs], compiler_cfg=compiler_cfg)

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process and display results
    loader.post_process(output[0])

    print("=" * 60, flush=True)


if __name__ == "__main__":

    # Run demo case
    run_yolov9_demo_case()
