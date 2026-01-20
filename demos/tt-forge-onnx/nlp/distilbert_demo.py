# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# DistilBERT Demo Script

import sys
import forge
from third_party.tt_forge_models.distilbert.masked_lm.pytorch import (
    ModelLoader as MaskedLMLoader,
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.distilbert.question_answering.pytorch import (
    ModelLoader as QuestionAnsweringLoader,
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.distilbert.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
    ModelVariant as SequenceClassificationVariant,
)
from third_party.tt_forge_models.distilbert.token_classification.pytorch import (
    ModelLoader as TokenClassificationLoader,
    ModelVariant as TokenClassificationVariant,
)


def run_distilbert_demo_case(task_type, variant, loader_class):

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

    # Task-specific post-processing and display
    if task_type in ["masked_lm", "question_answering"]:
        loader.decode_output(output)
    else:
        loader.decode_output(output, framework_model)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Masked LM
        ("masked_lm", MaskedLMVariant.DISTILBERT_BASE_CASED, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.DISTILBERT_BASE_UNCASED, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.DISTILBERT_BASE_MULTILINGUAL_CASED, MaskedLMLoader),
        # Token Classification
        (
            "token_classification",
            TokenClassificationVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL,
            TokenClassificationLoader,
        ),
        # Question Answering
        ("question_answering", QuestionAnsweringVariant.DISTILBERT_BASE_CASED_DISTILLED_SQUAD, QuestionAnsweringLoader),
        # Sequence Classification
        (
            "sequence_classification",
            SequenceClassificationVariant.DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH,
            SequenceClassificationLoader,
        ),
    ]

    # Run each demo case
    for task_type, variant, loader_class in demo_cases:
        run_distilbert_demo_case(task_type, variant, loader_class)
