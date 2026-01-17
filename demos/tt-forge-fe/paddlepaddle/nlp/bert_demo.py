# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Albert Demo Script

import forge
from forge.tvm_calls.forge_utils import paddle_trace
from third_party.tt_forge_models.bert.masked_lm.paddlepaddle import (
    ModelLoader as MaskedLMLoader,
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.bert.question_answering.paddlepaddle import (
    ModelLoader as QuestionAnsweringLoader,
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.bert.sequence_classification.paddlepaddle import (
    ModelLoader as SequenceClassificationLoader,
    ModelVariant as SequenceClassificationVariant,
)


def run_bert_demo_case(task_type, variant, loader_class):

    # Load model and input
    loader = loader_class(variant=variant)
    model = loader.load_model()
    inputs = loader.load_inputs()

    # Compile the model using Forge
    framework_model, _ = paddle_trace(model, inputs=inputs)
    compiled_model = forge.compile(framework_model, inputs)

    # Run inference on Tenstorrent device
    output = compiled_model(*inputs)

    if task_type in ["masked_lm", "question_answering"]:
        answer = loader.decode_output(output)
        print(f"Answer: {answer} of task type {task_type} and variant {variant}", flush=True)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Masked LM
        ("masked_lm", MaskedLMVariant.BERT_BASE_UNCASED, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.BERT_BASE_JAPANESE, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.CHINESE_ROBERTA_BASE, MaskedLMLoader),
        # Question Answering
        ("question_answering", QuestionAnsweringVariant.BERT_BASE_UNCASED, QuestionAnsweringLoader),
        ("question_answering", QuestionAnsweringVariant.BERT_BASE_JAPANESE, QuestionAnsweringLoader),
        ("question_answering", QuestionAnsweringVariant.CHINESE_ROBERTA_BASE, QuestionAnsweringLoader),
        # Sequence Classification
        ("sequence_classification", SequenceClassificationVariant.BERT_BASE_UNCASED, SequenceClassificationLoader),
        ("sequence_classification", SequenceClassificationVariant.BERT_BASE_JAPANESE, SequenceClassificationLoader),
        ("sequence_classification", SequenceClassificationVariant.CHINESE_ROBERTA_BASE, SequenceClassificationLoader),
    ]

    # Run each demo case
    for task_type, variant, loader_class in demo_cases:
        run_bert_demo_case(task_type, variant, loader_class)
