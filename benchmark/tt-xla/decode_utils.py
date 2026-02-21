# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Decode utilities for LLM benchmarking and reference generation.

This module centralizes decode logic used by:
- scripts/generate_reference_outputs.py (reference .refpt generation)
- llm_benchmark.py (teacher-forced accuracy generation)

The goal is to avoid subtle drift between codepaths (tokenization, cache init,
cache_position semantics, and teacher-forced decode loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import time
from transformers.cache_utils import StaticCache


ReadLogitsFn = Callable[[object], torch.Tensor]


def default_read_logits_fn(output: object) -> torch.Tensor:
    """Extract logits from a HuggingFace model output."""
    # Common case: ModelingOutputs with .logits
    if hasattr(output, "logits"):
        return output.logits
    # Some model wrappers return tuple-like outputs (logits, past_key_values, ...)
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    raise TypeError(f"Unsupported model output type for logits extraction: {type(output)}")


def assert_eval_no_dropout(model: torch.nn.Module, *, verbose: bool = False) -> None:
    """Ensure determinism-relevant flags are set."""
    assert model.training is False, "Model must be in eval mode"
    dropout_active = any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))
    assert not dropout_active, "No dropout modules should be in training mode"
    if verbose:
        dropout_count = sum(isinstance(m, torch.nn.Dropout) for m in model.modules())
        print(f"Model training mode: {model.training} (should be False)")
        print(f"Dropout modules found: {dropout_count}, any active: {dropout_active}")
        print("Decoding strategy: greedy (argmax), do_sample=False")


def teacher_forced_generate(
    *,
    model: torch.nn.Module,
    input_args: dict,
    device: torch.device,
    max_tokens_to_generate: int,
    ground_truth_tokens: torch.Tensor,
    read_logits_fn: Optional[ReadLogitsFn] = None,
    tokenizer: Optional[object] = None,
    verbose: bool = True,
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    """Teacher-forced generation loop that matches llm_benchmark.py semantics.

    This intentionally takes already-constructed `input_args` (prompt input_ids,
    StaticCache, and cache_position) so it can be used for both CPU models and
    device-compiled models without reconstructing inputs.

    Returns:
        (output_logits, predicted_tokens, iteration_times_ns)
    """
    if read_logits_fn is None:
        read_logits_fn = default_read_logits_fn

    assert (
        ground_truth_tokens.ndim == 1
    ), f"ground_truth_tokens must be 1D token IDs, got shape {tuple(ground_truth_tokens.shape)}"

    assert_eval_no_dropout(model, verbose=verbose)

    output_tokens: list[list[str]] = []
    output_logits: list[torch.Tensor] = []
    predicted_tokens: list[int] = []
    iteration_times_ns: list[int] = []

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            start = time.perf_counter_ns()

            output = model(**input_args)
            logits = read_logits_fn(output).to("cpu")
            output_logits.append(logits)

            next_token_ids = logits[:, -1].argmax(dim=-1)
            predicted_tokens.append(next_token_ids[0].item())

            if tokenizer is not None:
                output_text = [tokenizer.decode(token_id) for token_id in next_token_ids]
                output_tokens.append(output_text)

            # Teacher forcing update: feed ground truth for next step.
            if step < ground_truth_tokens.shape[0]:
                batch_size = input_args["input_ids"].shape[0]
                gt_token = ground_truth_tokens[step].to(device)
                input_args["input_ids"] = gt_token.view(1, 1).expand(batch_size, 1).contiguous()
            else:
                # If caller asks for more steps than ground truth provides, keep feeding last GT token.
                batch_size = input_args["input_ids"].shape[0]
                gt_token = ground_truth_tokens[-1].to(device)
                input_args["input_ids"] = gt_token.view(1, 1).expand(batch_size, 1).contiguous()

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

            iteration_times_ns.append(time.perf_counter_ns() - start)
            if verbose:
                print(f"Iteration\t{step}/{max_tokens_to_generate}\ttook {iteration_times_ns[-1] / 1e6:.04} ms")

    if verbose and tokenizer is not None:
        print("Output tokens:", output_tokens)

    return output_logits, predicted_tokens, iteration_times_ns


def init_static_cache(
    *,
    config,
    batch_size: int,
    max_cache_len: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> StaticCache:
    """Initialize a transformers StaticCache consistently."""
    if hasattr(config, "head_dim") and getattr(config, "head_dim"):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    num_key_value_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)

    static_cache = StaticCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    return static_cache


def extract_topk(
    logits: torch.Tensor,
    *,
    k: int = 5,
    rank_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (top1, topk) token IDs from logits.

    Args:
        logits: Tensor shaped [..., vocab]
    Returns:
        top1_ids: Tensor shaped [...]
        topk_ids: Tensor shaped [..., k]
    """
    logits_rank = logits.to(dtype=rank_dtype)
    top1 = logits_rank.argmax(dim=-1)
    topk = torch.topk(logits_rank, k=k, dim=-1).indices

    # Ensure argmax token is in slot 0 of the topk list (swap if needed).
    # This prevents "topk" tie ordering from changing the meaning of "topk[:, 0]".
    top1_flat = top1.reshape(-1)
    topk_flat = topk.reshape(-1, k)
    for i in range(topk_flat.shape[0]):
        argmax_val = top1_flat[i].item()
        row = topk_flat[i]
        matches = (row == argmax_val).nonzero(as_tuple=False)
        if matches.numel() > 0:
            j = matches[0].item()
            if j != 0:
                row[0], row[j] = row[j].clone(), row[0].clone()

    return top1, topk


@dataclass(frozen=True)
class ReferenceDecodeResult:
    """Outputs suitable for saving into a .refpt file."""

    reference_tokens: torch.Tensor  # [1, total_length]
    top1_tokens: torch.Tensor  # [total_length-1]
    top5_tokens: torch.Tensor  # [total_length-1, 5]


def generate_reference_topk(
    *,
    model: torch.nn.Module,
    tokens_1d: torch.Tensor,
    split_point: int,
    static_cache: StaticCache,
    device: torch.device,
    read_logits_fn: Optional[ReadLogitsFn] = None,
    rank_dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> ReferenceDecodeResult:
    """Compute reference top1/top5 for every position in a fixed token stream.

    This matches the semantics expected by TokenAccuracy:
    - top* at position i predicts token at position i+1

    Args:
        tokens_1d: shape [total_length]
        split_point: prefill length; decode begins at this position

    Returns:
        ReferenceDecodeResult with top1/top5 for positions [0..total_length-2].
    """
    if read_logits_fn is None:
        read_logits_fn = default_read_logits_fn

    assert tokens_1d.ndim == 1, f"tokens_1d must be 1D, got shape {tuple(tokens_1d.shape)}"
    total_length = tokens_1d.shape[0]
    if split_point <= 0 or split_point >= total_length:
        raise ValueError(f"split_point must be in (0, total_length), got {split_point} / {total_length}")

    assert_eval_no_dropout(model, verbose=verbose)

    all_top1: list[torch.Tensor] = []
    all_top5: list[torch.Tensor] = []

    with torch.no_grad():
        # Prefill: process first split_point tokens in one pass.
        prefill_tokens = tokens_1d[:split_point].unsqueeze(0).to(device)
        cache_position = torch.arange(0, split_point, device=device)

        out = model(
            input_ids=prefill_tokens,
            past_key_values=static_cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = read_logits_fn(out)  # [1, split_point, vocab]

        top1, top5 = extract_topk(logits, k=5, rank_dtype=rank_dtype)
        all_top1.append(top1.squeeze(0).cpu())  # [split_point]
        all_top5.append(top5.squeeze(0).cpu())  # [split_point, 5]

        # Decode: teacher forcing over tokens[split_point:]
        decode_tokens = tokens_1d[split_point:]
        for step in range(decode_tokens.shape[0] - 1):
            gt_token = decode_tokens[step]
            input_ids = gt_token.view(1, 1).to(device)
            cache_position = torch.tensor([split_point + step], device=device)

            out = model(
                input_ids=input_ids,
                past_key_values=static_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            logits = read_logits_fn(out)  # [1, 1, vocab]

            top1, top5 = extract_topk(logits, k=5, rank_dtype=rank_dtype)
            all_top1.append(top1.squeeze(0).squeeze(0).view(1).cpu())  # [1]
            all_top5.append(top5.squeeze(0).squeeze(0).view(1, 5).cpu())  # [1, 5]

    top1_tokens = torch.cat(all_top1, dim=0)  # [total_length-1]
    top5_tokens = torch.cat(all_top5, dim=0)  # [total_length-1, 5]

    # Sanity: number of predictions is total_length-1
    if top1_tokens.shape[0] != total_length - 1:
        raise RuntimeError(f"Internal error: expected top1 length {total_length-1}, got {top1_tokens.shape[0]}")

    return ReferenceDecodeResult(
        reference_tokens=tokens_1d.view(1, -1).cpu(),
        top1_tokens=top1_tokens,
        top5_tokens=top5_tokens,
    )


def init_accuracy_testing(model_name_for_accuracy: str, max_cache_len: int, tokenizer):
    """Initialize token accuracy testing for LLM benchmarks.

    Args:
        model_name_for_accuracy: Model name for .refpt file lookup
        max_cache_len: Maximum cache length (determines prefill and decode splits).
                      WARNING: This value must match the total_length used when generating
                      reference outputs. Context length mismatch causes accuracy degradation.
                      If changed, regenerate ALL reference outputs with matching total_length.
        tokenizer: HuggingFace tokenizer instance

    Returns:
        Tuple of (token_accuracy, custom_input_prompt)
            - token_accuracy: TokenAccuracy instance
            - custom_input_prompt: Reference text string for benchmarking

    Raises:
        ValueError: If model_name_for_accuracy is None
    """
    from token_accuracy import TokenAccuracy

    if model_name_for_accuracy is None:
        raise ValueError("model_name_for_accuracy must be provided when accuracy_testing=True")

    # Use half the cache for prefill, half for decode
    # This ensures we fit within hardware constraints
    max_prefill = max_cache_len // 2
    max_decode = max_cache_len // 2

    token_accuracy = TokenAccuracy(
        model_name=model_name_for_accuracy,
        max_prefill_tokens=max_prefill,
        max_decode_tokens=max_decode,
    )

    # Get Tale of Two Cities text from reference data
    custom_input_prompt = token_accuracy.prepare_ref_tokens(tokenizer)
    print(
        f"Using reference text for accuracy testing:"
        f"\n  Max prefill: {max_prefill} tokens"
        f"\n  Max decode: {max_decode} tokens"
        f"\n  Text preview: {custom_input_prompt[:100]}..."
    )

    return token_accuracy, custom_input_prompt
