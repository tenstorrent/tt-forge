# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
# CodeGen Demo - CasualLM

import pytest
# Add repository root to path to locate third_party modules
import sys
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.codegen.pytorch.loader import ModelLoader, ModelVariant
import torch
import forge
from forge.verify.verify import verify

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


def demo_codegen(variant):

    # Load model and inputs using model loader
    model_loader = ModelLoader(variant)
    model = model_loader.load_model()
    framework_model = TextModelWrapper(model=model, text_embedding=model.transformer.wte)
    framework_model.eval()
    inputs_dict = model_loader.load_inputs()
    input_ids = inputs_dict["input_ids"]
    attn_mask = inputs_dict["attention_mask"]
    inputs = [input_ids, attn_mask]

    # Forge compile framework model
    compiled_model = forge.compile(framework_model, sample_inputs=inputs)

    # Model Verification
    verify(inputs, framework_model, compiled_model)


if __name__ == "__main__":

    demo_cases = [
        ModelVariant.CODEGEN_350M_MONO,
        ModelVariant.CODEGEN_350M_MULTI,
        ModelVariant.CODEGEN_350M_NL,
    ]

    # Run each demo case
    for variant in demo_cases:
        demo_codegen(variant)
