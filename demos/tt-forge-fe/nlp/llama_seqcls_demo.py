# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import forge
from forge.verify.verify import verify
# Add repository root to path to locate third_party modules
import sys
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.llama.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
)
from third_party.tt_forge_models.llama.sequence_classification.pytorch import (
    ModelVariant as SequenceClassificationVariant,
)


class TextModelWrapper(torch.nn.Module):
    def __init__(self, model, text_embedding=None):
        super().__init__()
        self.model = model
        self.text_embedding = text_embedding

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None and self.text_embedding is not None:
            inputs_embeds = self.text_embedding(input_ids)
            past_key_values_length = 0
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
            )
            logits = self.model(attention_mask=causal_attention_mask, inputs_embeds=inputs_embeds).logits
        else:
            logits = self.model(input_ids=input_ids).logits
        return logits


def llama_seqcls_demo(variant):

    # Load model and tokenizer
    loader = SequenceClassificationLoader(variant=variant)
    model = loader.load_model()
    tokenizer = loader._load_tokenizer()
    framework_model = TextModelWrapper(model=model)

    # Prepare inputs
    input_dict = loader.load_inputs()

    inputs = [input_dict["input_ids"]]

    # Compile the model with Forge
    compiled_model = forge.compile(
        framework_model,
        sample_inputs=inputs,
    )

    # Verify correctness
    _, co_out = verify(inputs, framework_model, compiled_model)

    # post processing
    print(f"Prediction: {loader.decode_output(co_out)}")


if __name__ == "__main__":

    demo_cases = [
        SequenceClassificationVariant.LLAMA_3_2_1B,
        SequenceClassificationVariant.LLAMA_3_2_1B_INSTRUCT,
        SequenceClassificationVariant.LLAMA_3_8B,
        SequenceClassificationVariant.LLAMA_3_8B_INSTRUCT,
        SequenceClassificationVariant.LLAMA_3_1_8B,
        SequenceClassificationVariant.LLAMA_3_1_8B_INSTRUCT,
        SequenceClassificationVariant.LLAMA_3_2_3B,
        SequenceClassificationVariant.LLAMA_3_2_3B_INSTRUCT,
        SequenceClassificationVariant.HUGGYLLAMA_7B,
    ]

    # Run each demo case
    for variant in demo_cases:
        llama_seqcls_demo(variant)
