# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ALBERT Demo Script

import sys
import jax
import jax.numpy as jnp
import numpy as np
from third_party.tt_forge_models.albert.masked_lm.jax import (
    ModelLoader as AlbertLoader,
    ModelVariant as AlbertVariant,
)


def run_albert_demo_case(variant):

    # Load Model and inputs using thirdparty loader on CPU first
    with jax.default_device(jax.devices("cpu")[0]):
        loader = AlbertLoader(variant=variant)
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

    # Calculate probabilities and get predicted token for [MASK]
    probs = softmax(logits)

    # Find the [MASK] token position
    mask_token_id = tokenizer.mask_token_id
    mask_positions = np.where(np.array(input_ids)[0] == mask_token_id)[0]

    if len(mask_positions) > 0:
        mask_position = mask_positions[0]
        mask_probs = probs[0, mask_position, :]

        predicted_token_id = int(np.argmax(mask_probs, axis=-1))
        predicted_token_prob = float(mask_probs[predicted_token_id])
        predicted_token = tokenizer.decode([predicted_token_id])

        # Get the original prompt text
        prompt = loader.sample_text

        print(f"Model Variant: {variant}")
        print(f"Prompt: {prompt}")
        print(f"Predicted token for [MASK]: '{predicted_token}' (id: {predicted_token_id})")
        print(f"Probability: {predicted_token_prob:.4f}")

        # Show top-k tokens for the masked position
        top_k = 20
        top_token_ids = np.argsort(-mask_probs)[:top_k]
        top_tokens = [tokenizer.decode([i]) for i in top_token_ids]
        top_probs = mask_probs[top_token_ids]

        print(f"\n{'Rank':<5} {'Token ID':<10} {'Token':<15} {'Probability':<10}")
        print("-" * 45)
        for rank, (tid, tok, prob) in enumerate(zip(top_token_ids, top_tokens, top_probs), 1):
            print(f"{rank:<5} {tid:<10} {repr(tok):<15} {prob:<10.4f}")
    else:
        print(f"No [MASK] token found in input for variant {variant}")

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # ALBERT model variants
        AlbertVariant.BASE_V2,
        AlbertVariant.LARGE_V2,
        AlbertVariant.XLARGE_V2,
        AlbertVariant.XXLARGE_V2,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_albert_demo_case(variant)
