# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
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


from third_party.tt_forge_models.deepseek.deepseek_coder.pytorch import (
    ModelLoader as CausalLMLoader,
)
import torch
from third_party.tt_forge_models.deepseek.deepseek_coder.pytorch import (
    ModelVariant as CausalLMVariant,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


DEEPSEEK_VARIANTS = [
    CausalLMVariant.DEEPSEEK_1_3B_INSTRUCT,
]


class DeepSeekWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_tensor, attention_mask=None, past_key_values=None):
        inputs_embeds = self.embed_tokens(input_tensor)
        past_key_values_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
        causal_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_tensor.shape, inputs_embeds, past_key_values_length
        )
        return self.model(input_ids=input_tensor, attention_mask=causal_attention_mask).logits


loader = CausalLMLoader()
model = loader.load_model()
framework_model = DeepSeekWrapper(model)
framework_model.eval()
tokenizer = loader._load_tokenizer()

padded_inputs = loader.load_inputs()

# Forge compile framework model
compiled_model = forge.compile(
    framework_model,
    sample_inputs=[padded_inputs],
)

# Model Verification
verify([padded_inputs], framework_model, compiled_model)

generated_text = loader.decode_output(
    max_new_tokens=512, model=compiled_model, inputs=padded_inputs, tokenizer=tokenizer
)
print(generated_text)
