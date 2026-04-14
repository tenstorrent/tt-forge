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

model.eval()

# Compile for Tenstorrent
compiled_model = torch.compile(model, backend="tt")

# Run inference
inputs = tokenizer("The capital of France is", return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    outputs = compiled_model(input_ids)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    print(tokenizer.decode(next_token[0]))
