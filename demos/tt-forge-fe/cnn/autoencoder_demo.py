# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Autoencoder Demo Script

import forge
from third_party.tt_forge_models.autoencoder.pytorch import ModelLoader, ModelVariant
import shutil
from forge._C import DataFormat
from forge.config import CompilerConfig
import torch

SAVE_DIR = "demos/tt-forge-fe/cnn/autoencoder"


def run_autoencoder_demo_case(variant):

    # Load model and input
    loader = ModelLoader(variant=variant)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    data_format_override = DataFormat.Float16_b
    compiler_cfg = CompilerConfig(default_df_override=data_format_override)

    # Compile the model using Forge
    compiled_model = forge.compile(model, sample_inputs=[inputs], compiler_cfg=compiler_cfg)

    # Run inference on Tenstorrent device
    output = compiled_model(inputs)

    # Post-process the output
    loader.post_processing(output, save_path=SAVE_DIR)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.LINEAR,
    ]

    try:
        for variant in demo_cases:
            run_autoencoder_demo_case(variant)
    finally:
        shutil.rmtree(SAVE_DIR, ignore_errors=True)
