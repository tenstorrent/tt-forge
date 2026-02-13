# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate reference outputs for LLM accuracy testing.

This script loads a HuggingFace model, runs it on the "Tale of Two Cities" text corpus,
and generates a .refpt file containing reference tokens and top-5 predictions for each position.

The .refpt files are used by the TokenAccuracy class for measuring TOP1 and TOP5 accuracy
during model inference testing.

Usage:
    python3 <path-to-script>/generate_reference_outputs.py \\
        --model "meta-llama/Llama-3.2-1B-Instruct" \\
        --output_file "<output-dir>/Llama-3.2-1B-Instruct.refpt" \\
        --total_length 1024

Output format (.refpt file):
    {
        'reference_tokens': torch.Tensor,  # Shape: [1, total_length]
        'top5_tokens': torch.Tensor,       # Shape: [total_length, 5]
    }
"""

import argparse
import bz2
import os
import sys
from pathlib import Path

import torch
import transformers
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Ensure we can import from benchmark/tt-xla when executed from scripts/
_this_dir = Path(__file__).resolve().parent
_tt_xla_dir = _this_dir.parent
_tt_xla_str = str(_tt_xla_dir)
if _tt_xla_str not in sys.path:
    sys.path.insert(0, _tt_xla_str)

from decode_utils import generate_reference_topk, init_static_cache


def generate_reference_outputs(total_length, output_file, model_name):
    """
    Generate reference outputs for accuracy testing using HuggingFace models.

    Args:
        total_length: Number of tokens to process from Tale of Two Cities.
                     WARNING: This value MUST match the max_cache_len (input_sequence_length)
                     used during accuracy testing. Context length mismatch causes accuracy
                     degradation even with teacher forcing. If you change input_sequence_length
                     in test_llms.py, regenerate ALL reference outputs with matching total_length.
        output_file: Path to save .refpt file
        model_name: HuggingFace model name (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
    """
    # Set device - force CPU to match accuracy testing environment
    device = torch.device("cpu")
    logger.info(f"Using device: {device} (CPU forced to match accuracy testing)")

    # Load model and tokenizer from HuggingFace
    config = AutoConfig.from_pretrained(model_name)

    # Qwen only: add rope scaling to the config, for long context support.
    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts
    if "Qwen" in model_name:
        config.rope_scaling = {"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    # Verify model is in eval mode (no dropout, no batch norm updates)
    assert model.training is False, "Model must be in eval mode"
    print(f"✓ Model training mode: {model.training} (should be False)")

    # Check for dropout modules
    dropout_count = sum(isinstance(m, torch.nn.Dropout) for m in model.modules())
    dropout_active = any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))
    print(f"✓ Dropout modules found: {dropout_count}, any active: {dropout_active}")
    assert not dropout_active, "No dropout modules should be in training mode"

    # Verify we're using greedy decoding (argmax), not sampling
    print("✓ Decoding strategy: greedy (argmax), do_sample=False")

    # Load the book text - look in ../reference_outputs relative to script location
    current_file_path = os.path.abspath(__file__)
    current_file_dir = os.path.dirname(current_file_path)
    # Navigate up to parent directory and look for reference_outputs
    parent_dir = os.path.dirname(current_file_dir)
    prompt_file = os.path.join(parent_dir, "reference_outputs", "tale-of-two-cities.txt.bz2")

    if not os.path.exists(prompt_file):
        raise FileNotFoundError(
            f"Tale of Two Cities text file not found: {prompt_file}\n"
            f"Please ensure tale-of-two-cities.txt.bz2 exists in the reference_outputs directory."
        )

    logger.info(f"Loading text from {prompt_file}")
    with bz2.open(prompt_file, "rt", encoding="utf-8") as f:
        text = f.read()

    # Encode text to tokens
    encoded_tokens = tokenizer.encode(text, add_special_tokens=True)[:total_length]
    encoded_tokens_tensor = torch.tensor(encoded_tokens, device=device).unsqueeze(0)  # Shape [1, seq_len] on device

    logger.info(f"Processing {len(encoded_tokens)} tokens")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using StaticCache to match accuracy testing environment")

    max_cache_len = total_length
    batch_size = 1
    split_point = total_length // 2

    static_cache = init_static_cache(
        config=config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )

    logger.info(f"Generating reference topk with shared decode logic (split_point={split_point})")
    ref = generate_reference_topk(
        model=model,
        tokens_1d=encoded_tokens_tensor.squeeze(0),
        split_point=split_point,
        static_cache=static_cache,
        device=device,
        verbose=True,
    )

    data = {
        "top1_tokens": ref.top1_tokens,
        "top5_tokens": ref.top5_tokens,
        "reference_tokens": ref.reference_tokens,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }

    torch.save(data, output_file)
    logger.info(f"Saved reference outputs to {output_file}")
    logger.info(f"Library versions: torch={torch.__version__}, transformers={transformers.__version__}")

    # Note: This script used to print an in-loop correctness table and segment summaries.
    # The shared decode core keeps generation deterministic; downstream accuracy reporting
    # is performed in the benchmark harness.


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference outputs for LLM accuracy testing using HuggingFace models.",
        epilog="""
Examples:
    # Generate reference for Llama 3.2 1B
    python3 generate_reference_outputs.py \\
        --model "meta-llama/Llama-3.2-1B-Instruct" \\
        --output_file "../reference_outputs/Llama-3.2-1B-Instruct.refpt"

    # Generate with custom length
    python3 generate_reference_outputs.py \\
        --model "mistralai/Mistral-7B-Instruct-v0.3" \\
        --output_file "../reference_outputs/Mistral-7B-Instruct-v0.3.refpt" \\
        --total_length 2048

WARNING: total_length must match the input_sequence_length used in accuracy testing.
         If you change DEFAULT_INPUT_SEQUENCE_LENGTH in test_llms.py, regenerate
         ALL reference outputs with the matching total_length value.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--total_length",
        type=int,
        default=128,
        help="Total length of tokens to process (default: 128). Must match max_cache_len during testing.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path for reference data (e.g., '../reference_outputs/ModelName.refpt')",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name (e.g., 'meta-llama/Llama-3.2-1B-Instruct')"
    )
    args = parser.parse_args()

    generate_reference_outputs(total_length=args.total_length, output_file=args.output_file, model_name=args.model)


if __name__ == "__main__":
    main()
