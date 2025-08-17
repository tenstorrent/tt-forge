# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This demo runs a single forward pass on the Llama 3.1 8B parameter model eagerly
# using Torch-XLA's SPMD mode.

# We must import tt_torch here as its import will register tenstorrents PJRT plugin with torch-xla.
import tt_torch

import os
import sys
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from transformers import AutoTokenizer, LlamaModel, LlamaForCausalLM

from tt_torch.tools.utils import (
    calculate_pcc,
)

PROMPT = "What is the name of the largest planet in our solar system?"


def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")

    # Basic XLA configuration
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"  # Enables the auto parallel pass in tt-mlir
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"  # Converts the StableHLO emitted by torch-xla to the Shardy dialect

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    os.environ["MESH_SHAPE"] = f"1,{num_devices}"  # Sets the mesh shape used by the auto parallel pass
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def apply_tensor_parallel_sharding_causal(causal_model: LlamaForCausalLM, mesh: Mesh) -> None:
    """
    Apply tensor parallel sharding to the causal Llama model.

    This function modifies the model in-place to add sharding annotations for tensor
    parallelism.
    """
    # Move model to XLA device first
    causal_model.to(torch_xla.device())

    # Shard the base model first
    apply_tensor_parallel_sharding_base(causal_model.model, mesh)

    # Now shard the LM head
    # lm_head: [vocab_size, hidden_size] -> shard dim 0
    xs.mark_sharding(causal_model.lm_head.weight, mesh, ("model", "batch"))


def apply_tensor_parallel_sharding_base(base_model: LlamaModel, mesh: Mesh) -> None:
    """
    Apply tensor parallel sharding to the base Llama model.
    """
    # Apply sharding to each transformer layer
    for layer in base_model.layers:
        # ========================================
        # MLP (Feed-Forward) Layer Sharding - shard the intermediate_size across devices
        # ========================================

        # up_proj: [intermediate_size, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))

        # gate_proj: [intermediate_size, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))

        # down_proj: [hidden_size, intermediate_size] -> shard dim 1
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        # ========================================
        # Self-Attention Layer Sharding - shard the heads across all devices
        # ========================================

        # q_proj: [num_heads * head_dim, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))

        # k_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))

        # v_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))

        # o_proj: [hidden_size, num_heads * head_dim] -> shard dim 1
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))

    print("Tensor parallel sharding applied successfully!")


def prepare_inputs(mesh: Mesh, input_ids: torch.Tensor) -> torch.Tensor:
    print(f"Preparing inputs for TP: batch_size={input_ids.shape[0]}, seq_length={input_ids.shape[1]}")

    # Move to XLA device
    input_ids = input_ids.to(torch_xla.device())

    # Replicate inputs to all devices
    xs.mark_sharding(input_ids, mesh, (None, None))

    return input_ids


def decode_output(logits: torch.Tensor, tokenizer: AutoTokenizer):
    next_token_logits = logits[:, -1, :]
    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
    decoded_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)

    return decoded_text, next_token_id


def run_inference_comparison():
    """
    Run a complete example comparing CPU vs tensor-parallel on-device inference.
    """
    model_name = "meta-llama/Meta-Llama-3.1-8B"
    print(f"Running inference comparison for {model_name}")

    # Setup environment
    setup_xla_environment()
    mesh = create_device_mesh()

    # Load model and configuration
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt", padding=True, truncation=True)

    # ========================================
    # CPU Reference Run
    # ========================================
    print("Running CPU Inference")

    # Run on CPU for reference
    with torch.no_grad():
        outputs_cpu = model(input_ids=input_ids, output_hidden_states=True)
        reference_hidden_states = outputs_cpu.hidden_states
        reference_logits = outputs_cpu.logits
    reference_text, reference_next_token = decode_output(reference_logits, tokenizer)

    # ========================================
    # Tensor Parallel Run
    # ========================================
    print("Running Tensor Parallel Inference")

    # Apply tensor parallelism
    apply_tensor_parallel_sharding_causal(model, mesh)

    # Prepare inputs for tensor parallel execution
    input_ids_tp = prepare_inputs(mesh, input_ids)

    # Run tensor parallel inference
    with torch.no_grad():
        outputs_tp = model(input_ids=input_ids_tp, output_hidden_states=True)
        torch_xla.sync()  # Ensure all computations are done
        tp_hidden_states = outputs_tp.hidden_states
        # Move the outputs to CPU
        tp_hidden_states_cpu = tuple(tensor.to("cpu") for tensor in tp_hidden_states)
        tp_logits = outputs_tp.logits.to("cpu")
    tp_text, tp_next_token = decode_output(tp_logits, tokenizer)

    # ========================================
    # Validation
    # ========================================
    print("\n=== Token Validation ===")
    print("Input Prompt: ", PROMPT)
    print(f"CPU Reference output text: {reference_text}")
    print(f"CPU Reference output token: {reference_next_token.item()}")
    print(f"Tensor parallel output text: {tp_text}")
    print(f"Tensor parallel output token: {tp_next_token.item()}")
    assert reference_next_token.item() == tp_next_token.item(), "ERROR: Output tokens differ"

    print("\n=== PCC Validation on hidden states ===")
    # Compare outputs
    layer_pccs = [
        calculate_pcc(ref_layer, tp_layer) for ref_layer, tp_layer in zip(reference_hidden_states, tp_hidden_states_cpu)
    ]
    for i, layer_pcc in enumerate(layer_pccs):
        print(f"Layer {i} {'(Embedding Layer)' if i == 0 else ''} PCC: {layer_pcc:.6f}")
    for i, layer_pcc in enumerate(layer_pccs):
        assert layer_pcc > 0.98, f"ERROR: Layer {i} PCC is below 0.98 threshold"


def main():
    print("Torch-XLA SPMD Tensor Parallelism for Llama 3.1 8B Model")
    print("=" * 50)

    try:
        run_inference_comparison()
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
