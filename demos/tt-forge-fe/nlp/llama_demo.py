# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import forge
from forge.verify.verify import verify
from third_party.tt_forge_models.llama.causal_lm.pytorch import (
    ModelLoader as CausalLMLoader,
    ModelVariant as CausalLMVariant,
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


# Select the LLaMA 3 Causal LM variant
LLAMA_VARIANT = CausalLMVariant.LLAMA_3_2_1B

# Load model and tokenizer
loader = CausalLMLoader(variant=LLAMA_VARIANT)
model = loader.load_model()
tokenizer = loader._load_tokenizer()
framework_model = TextModelWrapper(model=model, text_embedding=model.model.embed_tokens)

# Prepare inputs
input_dict, seq_len = loader.load_inputs()
inputs = [input_dict["input_ids"], input_dict["attention_mask"]]

# Compile the model with Forge
compiled_model = forge.compile(
    framework_model,
    sample_inputs=inputs,
)

# Verify correctness
verify(inputs, framework_model, compiled_model)

# Generate output
generated_text = loader.decode_output(
    model=compiled_model,
    inputs=inputs,
    tokenizer=tokenizer,
    seq_len=seq_len,
    max_new_tokens=512,
)
print(generated_text)
