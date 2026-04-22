"""
Based on:
    - https://github.com/tenstorrent/tt-forge-models/blob/main/qwen_3/causal_lm/pytorch/loader.py
    - https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-xla/nlp/pytorch/llama_demo.py
    - https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer


xr.set_device_type("TT")
device = torch_xla.device()

# Load the HuggingFace model.
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model = model.to(device)
model.eval()
compiled_model = torch.compile(model, backend="tt")

# Prepare the model input.
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

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
