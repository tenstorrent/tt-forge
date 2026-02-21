# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
import forge
from third_party.tt_forge_models.resnet.image_classification.onnx import ModelLoader, ModelVariant


def run_resnet_onnx(variant):
    loader = ModelLoader(variant=variant)
    with tempfile.TemporaryDirectory() as tmpdir:

        # Load model and input
        onnx_model = loader.load_model(forge_tmp_path=tmpdir)
        inputs = loader.load_inputs()
        framework_model = forge.OnnxModule(variant.value, onnx_model)

        # Compile the model using Forge
        compiled_model = forge.compile(framework_model, [inputs])

        # Run inference on Tenstorrent device
        output = compiled_model(inputs)
        loader.print_cls_results(output)

        print("=" * 60, flush=True)


if __name__ == "__main__":
    demo_cases = [
        ModelVariant.RESNET_50_HF,
        ModelVariant.RESNET_50_HF_HIGH_RES,
        ModelVariant.RESNET_50_TIMM,
        ModelVariant.RESNET_50_TIMM_HIGH_RES,
        ModelVariant.RESNET_18,
        ModelVariant.RESNET_34,
        ModelVariant.RESNET_50,
        ModelVariant.RESNET_50_HIGH_RES,
        ModelVariant.RESNET_101,
        ModelVariant.RESNET_152,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_resnet_onnx(variant)
