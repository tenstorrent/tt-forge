# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet PaddlePaddle Demo Script

import forge
from third_party.tt_forge_models.resnet.image_classification.paddlepaddle import ModelLoader, ModelVariant


def run_resnet_demo_case(variant):
    """
    Run a ResNet PaddlePaddle model
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
        ModelVariant.RESNET18,
        ModelVariant.RESNET34,
        ModelVariant.RESNET50,
        ModelVariant.RESNET101,
        ModelVariant.RESNET152,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_resnet_demo_case(variant)
