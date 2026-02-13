# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
import forge
from third_party.tt_forge_models.nbeats.time_series_forecasting.onnx import ModelLoader


def run_nbeats_demo_case():
    loader = ModelLoader()
    with tempfile.TemporaryDirectory() as tmpdir:

        # Load model and input
        onnx_model = loader.load_model(forge_tmp_path=tmpdir)
        inputs = loader.load_inputs()
        framework_model = forge.OnnxModule("nbeats", onnx_model)

        # Compile the model using Forge
        compiled_model = forge.compile(framework_model, inputs)

        # Run inference on Tenstorrent device
        compiled_model(*inputs)


if __name__ == "__main__":
    run_nbeats_demo_case()
