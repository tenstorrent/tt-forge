# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
import forge
from third_party.tt_forge_models.alexnet.image_classification.onnx import ModelLoader, ModelVariant


def run_alexnet_demo_case(variant):
    """
    Run AlexNet ONNX model
    """

    loader = ModelLoader(variant=variant)
    with tempfile.TemporaryDirectory() as tmpdir:

        # Load Model and inputs using the ModelLoader
        onnx_model = loader.load_model(onnx_tmp_path=tmpdir)
        inputs = loader.load_inputs().contiguous()

        # Compile the model using Forge
        compiled_model = forge.compile(onnx_model, [inputs])

        # Run inference on Tenstorrent device
        output = compiled_model(inputs)

        # Print the results
        loader.print_cls_results(output)
        print("=" * 60, flush=True)


if __name__ == "__main__":
    demo_cases = [
        ModelVariant.ALEXNET,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_alexnet_demo_case(variant)
