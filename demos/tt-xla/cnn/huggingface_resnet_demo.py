# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ResNet Demo following Hugging Face tutorial
# https://huggingface.co/microsoft/resnet-50#how-to-use

import sys
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend
import torch.nn as nn

from transformers import AutoImageProcessor, ResNetForImageClassification
from datasets import load_dataset


def run_resnet_demo_case():

    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load model and inputs
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    inputs = processor(image, return_tensors="pt")

    # Compile the model using XLA
    compiled_model = torch.compile(model, backend=xla_backend)

    # Move model and inputs to the TT device
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)
    # Keep framework model on CPU for post-processing comparison
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
    run_resnet_demo_case()
