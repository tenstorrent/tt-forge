# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BERT Demo Script

import sys
import forge
from third_party.tt_forge_models.bert.masked_lm.pytorch import (
    ModelLoader as MaskedLMLoader,
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.bert.question_answering.pytorch import (
    ModelLoader as QuestionAnsweringLoader,
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.bert.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
    ModelVariant as SequenceClassificationVariant,
)
from third_party.tt_forge_models.bert.token_classification.pytorch import (
    ModelLoader as TokenClassificationLoader,
    ModelVariant as TokenClassificationVariant,
)
from third_party.tt_forge_models.bert.sentence_embedding_generation.pytorch.loader import (
    ModelLoader as SentenceEmbeddingGenerationLoader,
)
from third_party.tt_forge_models.bert.sentence_embedding_generation.pytorch.loader import (
    ModelVariant as SentenceEmbeddingGenerationVariant,
)


def run_bert_demo_case(task_type, variant, loader_class):

    # Load Model and inputs
    loader = loader_class(variant=variant)
    framework_model = loader.load_model()
    input_dict = loader.load_inputs()

    if task_type in ["token_classification", "sequence_classification"]:
        inputs = [input_dict["input_ids"]]
    else:
        inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    mlir_config = forge.config.MLIRConfig()
    mlir_config.set_custom_config("enable-cpu-hoisted-const-eval=false")
    
    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs, compiler_cfg=forge.CompilerConfig(mlir_config=mlir_config))

    # Run inference on Tenstorrent device
    output = compiled_model(*inputs)

    # post-processing
    loader.decode_output(output)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Masked LM
        ("masked_lm", MaskedLMVariant.BERT_BASE_UNCASED, MaskedLMLoader),
        # Token Classification
        (
            "token_classification",
            TokenClassificationVariant.DBMDZ_BERT_LARGE_CASED_FINETUNED_CONLL03_ENGLISH,
            TokenClassificationLoader,
        ),
        # Question Answering
        ("question_answering", QuestionAnsweringVariant.PHIYODR_BERT_LARGE_FINETUNED_SQUAD2, QuestionAnsweringLoader),
        (
            "question_answering",
            QuestionAnsweringVariant.BERT_LARGE_CASED_WHOLE_WORD_MASKING_FINETUNED_SQUAD,
            QuestionAnsweringLoader,
        ),
        # Sequence Classification
        (
            "sequence_classification",
            SequenceClassificationVariant.TEXTATTACK_BERT_BASE_UNCASED_SST_2,
            SequenceClassificationLoader,
        ),
        # Sentence Embedding Generation
        (
            "sentence_embedding_generation",
            SentenceEmbeddingGenerationVariant.EMRECAN_BERT_BASE_TURKISH_CASED_MEAN_NLI_STSB_TR,
            SentenceEmbeddingGenerationLoader,
        ),
    ]

    # Run each demo case
    for task_type, variant, loader_class in demo_cases:
        run_bert_demo_case(task_type, variant, loader_class)
