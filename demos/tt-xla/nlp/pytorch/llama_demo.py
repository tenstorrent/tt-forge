# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer

xr.set_device_type("TT")
device = torch_xla.device()

# Load any HuggingFace model
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

model = model.to(device)
model.eval()

# Compile for Tenstorrent
compiled_model = torch.compile(model, backend="tt")

# Run inference
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    outputs = compiled_model(input_ids)
    logits = outputs.logits[:, -1, :]

    # Get top 5 predictions
    probs = torch.softmax(logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)

    # Print top prediction (original behavior)
    print(f"Prompt: `{prompt}`, Top prediction: `{tokenizer.decode(top5_indices[0][0])}`")

    # Print table
    print(f"\n{'Rank':<6} {'Token':<15} {'Probability':<12}")
    print("-" * 35)
    for rank, (idx, prob) in enumerate(zip(top5_indices[0], top5_probs[0]), start=1):
        token = tokenizer.decode(idx)
        print(f"{rank:<6} {repr(token):<15} {prob.item():.4%}")
