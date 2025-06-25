# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import numpy as np
from transformers import FlaxGPT2LMHeadModel, AutoTokenizer

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
with jax.default_device(jax.devices("cpu")[0]):
    model = FlaxGPT2LMHeadModel.from_pretrained(model_name)


prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="np")
input_ids = inputs["input_ids"]


@jax.jit
def generate_logits(input_ids, params):
    outputs = model(input_ids=input_ids, params=params)
    return outputs.logits


def softmax(x):
    c = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - c)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


outputs = generate_logits(input_ids, model.params)
probs = softmax(np.array(outputs))
last_token_probs = probs[0, -1, :]

next_token_id = int(np.argmax(last_token_probs, axis=-1))
next_token_prob = float(last_token_probs[next_token_id])
next_token = tokenizer.decode([next_token_id])

print(f"Prompt: {prompt}")
print(f"Next token: '{next_token}' (id: {next_token_id})")
print(f"Probability: {next_token_prob:.4f}")

top_k = 20
top_token_ids = np.argsort(-last_token_probs)[:top_k]
top_tokens = [tokenizer.decode([i]) for i in top_token_ids]
top_probs = last_token_probs[top_token_ids]

print(f"\n{'Rank':<5} {'Token ID':<10} {'Token':<15} {'Probability':<10}")
print("-" * 45)
for rank, (tid, tok, prob) in enumerate(zip(top_token_ids, top_tokens, top_probs), 1):
    print(f"{rank:<5} {tid:<10} {repr(tok):<15} {prob:<10.4f}")
