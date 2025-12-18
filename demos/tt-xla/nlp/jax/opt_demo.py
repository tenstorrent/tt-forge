# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# OPT Demo Script

import sys
import jax
import jax.numpy as jnp
import numpy as np
from third_party.tt_forge_models.opt.causal_lm.jax import (
    ModelLoader as OPTLoader,
    ModelVariant as OPTVariant,
)


def run_opt_demo_case(variant):

    # Load Model and inputs using thirdparty loader on CPU first
    with jax.default_device(jax.devices("cpu")[0]):
        loader = OPTLoader(variant=variant)
        model = loader.load_model()
        tokenizer = loader._load_tokenizer()
        input_dict = loader.load_inputs()

    # Get input_ids from the loaded inputs
    input_ids = input_dict["input_ids"]

    # Define the forward function for JAX compilation
    def generate_logits(input_ids, params):
        outputs = model(input_ids=input_ids, params=params)
        return outputs.logits

    # Compile the model using JAX JIT with TT backend
    compiled_generate_logits = jax.jit(
        generate_logits,
    )

    # Run inference on TT device
    outputs = compiled_generate_logits(input_ids, model.params)

    # Convert to numpy for post-processing
    logits = np.array(outputs)

    # Softmax function for probability calculation
    def softmax(x):
        c = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - c)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    # Calculate probabilities and get next token
    probs = softmax(logits)
    last_token_probs = probs[0, -1, :]

    next_token_id = int(np.argmax(last_token_probs, axis=-1))
    next_token_prob = float(last_token_probs[next_token_id])
    next_token = tokenizer.decode([next_token_id])

    # Get the original prompt text
    prompt = loader.sample_text

    print(f"Model Variant: {variant}")
    print(f"Prompt: {prompt}")
    print(f"Next token: '{next_token}' (id: {next_token_id})")
    print(f"Probability: {next_token_prob:.4f}")

    # Show top-k tokens
    top_k = 20
    top_token_ids = np.argsort(-last_token_probs)[:top_k]
    top_tokens = [tokenizer.decode([i]) for i in top_token_ids]
    top_probs = last_token_probs[top_token_ids]

    print(f"\n{'Rank':<5} {'Token ID':<10} {'Token':<15} {'Probability':<10}")
    print("-" * 45)
    for rank, (tid, tok, prob) in enumerate(zip(top_token_ids, top_tokens, top_probs), 1):
        print(f"{rank:<5} {tid:<10} {repr(tok):<15} {prob:<10.4f}")

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # OPT model variants
        OPTVariant._125M,
        OPTVariant._350M,
        OPTVariant._1_3B,
        OPTVariant._2_7B,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_opt_demo_case(variant)
