# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# OPT Demo Script

import sys
import torch
from torch.utils._pytree import tree_map
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend
from third_party.tt_forge_models.opt.causal_lm.pytorch import (
    ModelLoader as CausalLMLoader,
    ModelVariant as CausalLMVariant,
)
from third_party.tt_forge_models.opt.qa.pytorch import (
    ModelLoader as QuestionAnsweringLoader,
    ModelVariant as QuestionAnsweringVariant,
)
from third_party.tt_forge_models.opt.sequence_classification.pytorch import (
    ModelLoader as SequenceClassificationLoader,
    ModelVariant as SequenceClassificationVariant,
)


def run_opt_demo_case(task_type, variant, loader_class):

    # Set the XLA runtime device to TT
    xr.set_device_type("TT")

    # Load Model and inputs
    loader = loader_class(variant=variant)
    framework_model = loader.load_model()
    framework_model.config.return_dict = False
    inputs = loader.load_inputs()

    # Handle different input formats for different tasks
    if task_type == "causal_lm":
        # For causal LM, only use input_ids
        inputs = [inputs[0]]

    # Compile the model using XLA
    compiled_model = torch.compile(framework_model, backend=xla_backend)

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
        if task_type == "causal_lm":
            output = compiled_model(inputs[0])
        else:
            output = compiled_model(inputs[0], inputs[1])

    # Task-specific post-processing and display
    if task_type == "causal_lm":
        # For causal LM, extract logits from the output tuple
        if isinstance(output, tuple):
            logits = output[0]  # First element is usually logits
        else:
            logits = output
        print(f"Causal LM output shape: {logits.shape}", flush=True)
        print(f"Sample logits (first 5): {logits[0, -1, :5].tolist()}", flush=True)

    elif task_type == "question_answering":
        # For QA, extract logits from the output tuple
        if isinstance(output, tuple):
            logits = output[0]  # First element is usually logits
        else:
            logits = output
        print(f"QA output shape: {logits.shape}", flush=True)
        print(f"Sample logits (first 5): {logits[0, :5].tolist()}", flush=True)

    elif task_type == "sequence_classification":
        # Use the decode_output method if available
        if hasattr(loader, "decode_output"):
            loader.decode_output(output)
        else:
            # Extract logits from the output tuple
            if isinstance(output, tuple):
                logits = output[0]  # First element is usually logits
            else:
                logits = output
            print(f"Sequence classification output shape: {logits.shape}", flush=True)
            print(f"Predicted class: {logits.argmax(-1).item()}", flush=True)

    print("=" * 60, flush=True)


if __name__ == "__main__":

    demo_cases = [
        # Causal Language Modeling
        ("causal_lm", CausalLMVariant.OPT_125M, CausalLMLoader),
        ("causal_lm", CausalLMVariant.OPT_350M, CausalLMLoader),
        # Question Answering
        ("question_answering", QuestionAnsweringVariant.OPT_125M, QuestionAnsweringLoader),
        ("question_answering", QuestionAnsweringVariant.OPT_350M, QuestionAnsweringLoader),
        # Sequence Classification
        ("sequence_classification", SequenceClassificationVariant.OPT_125M, SequenceClassificationLoader),
        ("sequence_classification", SequenceClassificationVariant.OPT_350M, SequenceClassificationLoader),
        ("sequence_classification", SequenceClassificationVariant.OPT_1_3B, SequenceClassificationLoader),
    ]

    # Run each demo case
    for task_type, variant, loader_class in demo_cases:
        run_opt_demo_case(task_type, variant, loader_class)
