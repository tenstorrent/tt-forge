# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# GPT-Neo Demo Script

import sys
import forge
import torch
# Add repository root to path to locate third_party modules
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.gpt_neo.causal_lm.pytorch import (
    ModelLoader as CausalLMLoader,
    ModelVariant as CausalLMVariant,
)
from third_party.tt_forge_models.gpt_neo.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
    ModelVariant as SequenceClassificationVariant,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


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


def run_gptneo_demo_case(task_type, variant, loader_class):

    # Load Model and inputs
    loader = loader_class(variant=variant)
    model = loader.load_model()
    input_dict = loader.load_inputs()

    if task_type == "causal_lm":
        inputs = [input_dict["input_ids"], input_dict["attention_mask"]]
    else:
        inputs = [input_dict["input_ids"]]

    framework_model = TextModelWrapper(model=model, text_embedding=model.transformer.wte)
    framework_model.eval()

    # Compile the model using Forge
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Run inference on Tenstorrent device
    output = compiled_model(*inputs)

    if task_type == "sequence_classification":
        predicted_category = loader.decode_output(output)
        print(f"Predicted category: {predicted_category}", flush=True)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Causal Language Modeling
        ("causal_lm", CausalLMVariant.GPT_NEO_125M, CausalLMLoader),
        ("causal_lm", CausalLMVariant.GPT_NEO_1_3B, CausalLMLoader),
        # Sequence Classification
        ("sequence_classification", SequenceClassificationVariant.GPT_NEO_125M, SequenceClassificationLoader),
        ("sequence_classification", SequenceClassificationVariant.GPT_NEO_1_3B, SequenceClassificationLoader),
    ]

    # Run each demo case
    for task_type, variant, loader_class in demo_cases:
        run_gptneo_demo_case(task_type, variant, loader_class)
