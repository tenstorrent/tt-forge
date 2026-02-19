# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# AlexNet PaddlePaddle Demo Script

import forge
from third_party.tt_forge_models.alexnet.image_classification.paddlepaddle import ModelLoader, ModelVariant


def run_alexnet_demo_case(variant):
    """
    Run an AlexNet PaddlePaddle model
    """

    # Load Model and inputs using the ModelLoader
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model
    compiled_model = forge.compile(model, inputs)

    # Run inference
    output = compiled_model(*inputs)

    # Post-process
    loader.print_results(output)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.DEFAULT,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_alexnet_demo_case(variant)
