# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# ConvNext Demo - facebook/convnext-base-224 on Tenstorrent hardware
# https://huggingface.co/facebook/convnext-base-224

import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from transformers import AutoImageProcessor, ConvNextForImageClassification
from datasets import load_dataset


def run_convnext_demo():

    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load model and inputs
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    model_id = "facebook/convnext-base-224"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ConvNextForImageClassification.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    )
    model.eval()

    inputs = processor(image, return_tensors="pt")

    # Cast inputs to bfloat16
    inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}

    # Compile the model using XLA
    compiled_model = torch.compile(model, backend="tt")

    # Move model and inputs to the TT device
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)

    def attempt_to_device(x):
        if hasattr(x, "to"):
            return x.to(device)
        return x

    inputs = tree_map(attempt_to_device, inputs)

    # Run inference on Tenstorrent device
    with torch.no_grad():
        logits = compiled_model(**inputs).logits

    # Model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    print(f"Predicted label: {model.config.id2label[predicted_label]}")

    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_convnext_demo()
