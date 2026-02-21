# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BLIP Demo Script

import forge
import paddle
from forge.tvm_calls.forge_utils import paddle_trace
from third_party.tt_forge_models.blip.vision_language.paddlepaddle import ModelLoader, ModelVariant


def run_blip_demo_case(variant):
    """
    Run a BLIP PaddlePaddle model
    """

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)

    if variant == ModelVariant.BLIP_IMAGE_CAPTIONING:
        # Test framework model
        outputs = model(*inputs)

        image_embed = outputs[1]
        text_embeds = outputs[0]

        image_embed = paddle.nn.functional.normalize(image_embed, axis=-1)
        text_embeds = paddle.nn.functional.normalize(text_embeds, axis=-1)

        similarities = paddle.matmul(text_embeds, image_embed.T)
        similarities = similarities.squeeze().numpy()

        for t, sim in zip(loader.text, similarities):
            print(f"{t}: similarity = {sim:.4f}")

        # Compile the model
        framework_model, _ = paddle_trace(model, inputs=inputs)
        compiled_model = forge.compile(framework_model, inputs)
        outputs = compiled_model(*inputs)
    else:
        # Run inference
        outputs = compiled_model(*inputs)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.BLIP_IMAGE_CAPTIONING,
        ModelVariant.BLIP_TEXT,
        ModelVariant.BLIP_VISION,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_blip_demo_case(variant)
