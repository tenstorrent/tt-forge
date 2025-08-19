# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# YOLOv8 Demo Script

import sys
import forge
from third_party.tt_forge_models.yolov8.pytorch import ModelLoader, ModelVariant
import torch


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


def run_yolov8_demo_case(variant):

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()
    framework_model = YoloWrapper(model)

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=[inputs])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process and display results
    loader.post_process(output[0])

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.YOLOV8X,
        ModelVariant.YOLOV8N,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_yolov8_demo_case(variant)
