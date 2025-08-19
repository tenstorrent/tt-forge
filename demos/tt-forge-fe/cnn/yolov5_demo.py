# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# YOLOv5 Demo Script

import sys
import forge
from third_party.tt_forge_models.yolov5.pytorch import ModelLoader, ModelVariant


def run_yolov5_demo_case(variant):

    # Load Model and inputs
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    ims, n, files, shape0, shape1, inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs])

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process and display results
    loader.post_process(ims, inputs.shape, output, model, n, shape0, shape1, files)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.YOLOV5N,
        ModelVariant.YOLOV5S,
        ModelVariant.YOLOV5M,
        ModelVariant.YOLOV5L,
        ModelVariant.YOLOV5X,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_yolov5_demo_case(variant)
