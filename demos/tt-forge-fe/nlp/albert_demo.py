# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Albert Demo Script

import sys
import forge
# Add repository root to path to locate third_party modules
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.albert.masked_lm.pytorch import (
    ModelLoader as MaskedLMLoader,
    ModelVariant as MaskedLMVariant,
)
from third_party.tt_forge_models.albert.question_answering.pytorch import (
    ModelLoader as QuestionAnsweringLoader,
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.albert.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
    ModelVariant as SequenceClassificationVariant,
)
from third_party.tt_forge_models.albert.token_classification.pytorch import (
    ModelLoader as TokenClassificationLoader,
    ModelVariant as TokenClassificationVariant,
)


def run_albert_demo_case(task_type, variant, loader_class):

    # Load Model and inputs
    loader = loader_class(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    input_dict = loader.load_inputs()
    inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Run inference on Tenstorrent device
    output = compiled_model(inputs[0], inputs[1])

    # Task-specific post-processing and display
    if task_type == "masked_lm":
        predicted_tokens = loader.decode_output(output, input_dict)
        print(f"The predicted token for the [MASK] is: {predicted_tokens}", flush=True)

    elif task_type == "token_classification":
        predicted_tokens_classes = loader.decode_output(output, input_dict)
        print(f"Predicted token classes: {predicted_tokens_classes}", flush=True)

    elif task_type == "question_answering":
        predicted_answer = loader.decode_output(output, input_dict)
        print(f"Predicted answer: {predicted_answer}", flush=True)

    elif task_type == "sequence_classification":
        predicted_category = loader.decode_output(output)
        print(f"Predicted category: {predicted_category}", flush=True)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Masked LM
        ("masked_lm", MaskedLMVariant.BASE_V1, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.LARGE_V1, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.XLARGE_V1, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.XXLARGE_V1, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.BASE_V2, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.LARGE_V2, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.XLARGE_V2, MaskedLMLoader),
        ("masked_lm", MaskedLMVariant.XXLARGE_V2, MaskedLMLoader),
        # Token Classification
        ("token_classification", TokenClassificationVariant.BASE_V1, TokenClassificationLoader),
        ("token_classification", TokenClassificationVariant.LARGE_V1, TokenClassificationLoader),
        ("token_classification", TokenClassificationVariant.XLARGE_V1, TokenClassificationLoader),
        ("token_classification", TokenClassificationVariant.XXLARGE_V1, TokenClassificationLoader),
        ("token_classification", TokenClassificationVariant.BASE_V2, TokenClassificationLoader),
        ("token_classification", TokenClassificationVariant.LARGE_V2, TokenClassificationLoader),
        ("token_classification", TokenClassificationVariant.XLARGE_V2, TokenClassificationLoader),
        ("token_classification", TokenClassificationVariant.XXLARGE_V2, TokenClassificationLoader),
        # Question Answering
        ("question_answering", QuestionAnsweringVariant.SQUAD2, QuestionAnsweringLoader),
        # Sequence Classification
        ("sequence_classification", SequenceClassificationVariant.IMDB, SequenceClassificationLoader),
    ]

    # Run each demo case
    for task_type, variant, loader_class in demo_cases:
        run_albert_demo_case(task_type, variant, loader_class)
