# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
import forge  # type: ignore[reportMissingImports]
from third_party.tt_forge_models.densenet.image_classification.onnx import ModelLoader


def run_densenet_demo_case():
    loader = ModelLoader()
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_model = loader.load_model(forge_tmp_path=tmpdir)
        inputs = loader.load_inputs()
        compiled_model = forge.compile(onnx_model, [inputs])
        output = compiled_model(inputs)
        loader.print_cls_results(output)


if __name__ == "__main__":
    run_densenet_demo_case()
