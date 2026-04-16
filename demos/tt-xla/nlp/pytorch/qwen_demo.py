# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Based on:
    - https://github.com/tenstorrent/tt-forge-models/blob/main/qwen_3/causal_lm/pytorch/loader.py
    - https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-xla/nlp/pytorch/llama_demo.py
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch import ModelLoader, ModelVariant
import sys


xr.set_device_type("TT")
device = torch_xla.device()

# Load model and inputs.
loader = ModelLoader(variant=ModelVariant.QWEN_3_0_6B)
tokenizer = loader.tokenizer
assert tokenizer is not None
prompt = loader.sample_text
framework_model = loader.load_model()
inputs = loader.load_inputs()

# Compile for Tenstorrent.
model = framework_model.to(device)
model.eval()
compiled_model = torch.compile(model, backend="tt")

# Run inference.
input_ids = inputs["input_ids"].to(device)
with torch.no_grad():
    outputs = compiled_model(input_ids)
    logits = outputs.logits[:, -1, :]

    # Get top 5 predictions.
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)

    # Print top prediction (original behavior).
    print(f"Prompt: `{prompt}`, Top prediction: `{tokenizer.decode(top5_indices[0][0])}`")

    # Print table.
    print(f"\n{'Rank':<6} {'Token':<15} {'Probability':<12}")
    print("-" * 35)
    for rank, (idx, prob) in enumerate(zip(top5_indices[0], top5_probs[0]), start=1):
        token = tokenizer.decode(idx)
        print(f"{rank:<6} {repr(token):<15} {prob.item():.4%}")
