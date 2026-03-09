# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# MobileNetV2 PaddlePaddle Demo Script

import forge
from third_party.tt_forge_models.mobilenetv2.image_classification.paddlepaddle import ModelLoader, ModelVariant


def run_mobilenetv2_demo_case(variant):
    """
    Run a MobileNetV2 PaddlePaddle model
    """

    # Load Model and inputs using the ModelLoader
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    compiled_model = forge.compile(model, inputs)

    # Run inference on Tenstorrent device
    output = compiled_model(*inputs)

    # Print the results
    loader.print_results(output)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.DEFAULT,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_mobilenetv2_demo_case(variant)
