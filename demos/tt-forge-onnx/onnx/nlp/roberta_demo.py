# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
import forge
from third_party.tt_forge_models.roberta.sequence_classification.onnx.loader import ModelLoader


def run_roberta_demo_case():
    loader = ModelLoader()
    with tempfile.TemporaryDirectory() as tmpdir:

        # Load model and input
        onnx_model = loader.load_model(forge_tmp_path=tmpdir)
        inputs = loader.load_inputs()
        framework_model = forge.OnnxModule("roberta_sequence_classification", onnx_model)

        # Compile the model using Forge
        compiled_model = forge.compile(framework_model, [inputs])

        # Run inference on Tenstorrent device
        output = compiled_model(inputs)

        # Decode the output
        predicted_value = loader.decode_output(output)
        print(f"Predicted Sentiment: {predicted_value}")


if __name__ == "__main__":
    run_roberta_demo_case()
