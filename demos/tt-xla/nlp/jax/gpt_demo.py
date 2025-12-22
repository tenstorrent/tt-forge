# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# GPT-2 Demo Script

import sys
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from flax import nnx
# Add repository root to path to locate third_party modules
from pathlib import Path
repo_root = Path(__file__).resolve()
while repo_root != repo_root.parent and not (repo_root / "third_party").exists():
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


from third_party.tt_forge_models.gpt2.causal_lm.jax import (
    ModelLoader as GPT2Loader,
    ModelVariant as GPT2Variant,
)


def run_gpt2_demo_case(variant):

    # Load model, tokenizer, and inputs
    loader = GPT2Loader(variant=variant)
    model = loader.load_model()
    tokenizer = loader._load_tokenizer()
    input_ids = loader.load_inputs()

    # Configure single device mesh - automatically uses first available device (CPU or TT)
    mesh = Mesh(jax.devices()[:1], axis_names=("x",))
    model.config.set_model_mesh(mesh)

    # Define the forward function for JAX compilation
    graphdef = nnx.split(model)[0]

    def generate_logits(input_ids, params):
        model_ = nnx.merge(graphdef, params)
        outputs = model_(input_ids)
        return outputs.logits

    # Compile the model using JAX JIT
    compiled_generate_logits = jax.jit(
        generate_logits,
    )

    # Run inference
    with model.mesh:
        outputs = compiled_generate_logits(input_ids, nnx.split(model)[1])

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
        # GPT-2 model variants
        GPT2Variant.BASE,
        GPT2Variant.MEDIUM,
        GPT2Variant.LARGE,
        GPT2Variant.XL,
    ]

    # Run each demo case
    for variant in demo_cases:
        run_gpt2_demo_case(variant)
