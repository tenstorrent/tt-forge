# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# DPR Demo Script

import sys
import forge
from third_party.tt_forge_models.dpr.context_encoder.pytorch import (
    ModelLoader as ContextEncoderLoader,
    ModelVariant as ContextEncoderVariant,
)
from third_party.tt_forge_models.dpr.question_encoder.pytorch import (
    ModelLoader as QuestionEncoderLoader,
    ModelVariant as QuestionEncoderVariant,
)


def run_dpr_demo_case(task_type, variant, loader_class):

    # Load Model and inputs
    loader = loader_class(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Run inference on Tenstorrent device
    output = compiled_model(*inputs)

    # print the output
    print("embeddings", output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Question Encoder
        ("question_encoder", QuestionEncoderVariant.DPR_SINGLE_NQ_BASE, QuestionEncoderLoader),
        ("question_encoder", QuestionEncoderVariant.DPR_MULTISET_BASE, QuestionEncoderLoader),
    ]

    # Run each demo case
    for task_type, variant, loader_class in demo_cases:
        run_dpr_demo_case(task_type, variant, loader_class)
