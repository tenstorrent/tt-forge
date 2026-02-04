# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
from loguru import logger
from pathlib import Path


class TokenAccuracy:
    """
    Manages token accuracy testing for LLMs using precomputed reference data.

    Loads reference tokens and top-5 predictions from .refpt files,
    implements teacher forcing for decode loop, and computes TOP1/TOP5 accuracy.
    """

    def __init__(
        self,
        model_name: str,
        reference_dir: str = None,
        max_prefill_tokens: int = None,
        max_decode_tokens: int = None,
    ):
        """
        Initialize TokenAccuracy with reference data for the specified model.

        Args:
            model_name: Name of the model (e.g., "Llama-3.2-1B-Instruct")
            reference_dir: Directory containing .refpt files (defaults to reference_outputs/ relative to this file)
            max_prefill_tokens: Maximum number of prefill tokens to use (takes last N tokens from prefill)
            max_decode_tokens: Maximum number of decode tokens to use (takes first N tokens from decode)
        """
        # Default reference_dir to reference_outputs/ relative to this file
        if reference_dir is None:
            reference_dir = str(Path(__file__).parent / "reference_outputs")

        self.gt_pos = -1  # Current ground truth position (-1 = not started)
        self.store_predicted_tokens = []  # Store predictions for accuracy calculation

        # Load reference data file
        reference_data_file = os.path.join(reference_dir, f"{model_name}.refpt")
        if not os.path.exists(reference_data_file):
            raise FileNotFoundError(
                f"Reference data file not found: {reference_data_file}\n"
                f"Available models: {self._list_available_models(reference_dir)}"
            )

        logger.info(f"Loading reference data from {reference_data_file}")
        reference_data = torch.load(reference_data_file)

        # Extract data
        reference_tokens = reference_data["reference_tokens"]  # Shape: [1, total_length]
        full_top5_tokens = reference_data["top5_tokens"]  # Shape: [total_length, 5]

        # Split tokens: first half = prefill, second half = decode ground truth
        split_point = reference_tokens.shape[-1] // 2
        prefill_tokens = reference_tokens[0, :split_point]
        decode_tokens = reference_tokens[0, split_point:]

        # Apply limits: take last max_prefill_tokens from prefill, first max_decode_tokens from decode
        if max_prefill_tokens is not None and max_prefill_tokens < len(prefill_tokens):
            self.input_prompt = prefill_tokens[-max_prefill_tokens:]
            prefill_start_idx = len(prefill_tokens) - max_prefill_tokens
        else:
            self.input_prompt = prefill_tokens
            prefill_start_idx = 0

        if max_decode_tokens is not None and max_decode_tokens < len(decode_tokens):
            self.reference_tokens = decode_tokens[:max_decode_tokens]
        else:
            self.reference_tokens = decode_tokens

        # Adjust top5_tokens to match the selected decode window
        # Top5 tokens at position i predict token at position i+1
        # Decode tokens always start at split_point, regardless of prefill window
        # So for decode tokens at split_point onwards, we need top5 from split_point-1
        top5_start_idx = split_point - 1
        top5_end_idx = top5_start_idx + len(self.reference_tokens)
        self.top5_tokens = full_top5_tokens[top5_start_idx:top5_end_idx, :]

        self.maxindex = len(self.reference_tokens) - 1
        logger.info(
            f"Loaded {len(self.input_prompt)} input tokens (from position {prefill_start_idx}), "
            f"{len(self.reference_tokens)} ground truth tokens"
        )

    def prepare_ref_tokens(self, tokenizer) -> str:
        """
        Decode input prompt tokens back to text.

        Args:
            tokenizer: HuggingFace tokenizer

        Returns:
            Text string for input prompt
        """
        text_data = tokenizer.decode(self.input_prompt.tolist())
        return text_data

    def collect_predicted_tokens(self, predicted_token: int) -> torch.Tensor:
        """
        Store predicted token and return ground truth token for teacher forcing.

        This implements teacher forcing: the model's prediction is stored for
        accuracy calculation, but the GROUND TRUTH token is returned to be
        used as input for the next iteration. This prevents error accumulation
        and allows independent accuracy measurement at each position.

        Args:
            predicted_token: Token predicted by the model (scalar integer)

        Returns:
            Ground truth token tensor with shape [1, 1] for next iteration input
        """
        self.store_predicted_tokens.append(predicted_token)
        self.gt_pos += 1

        # Return ground truth token (not the prediction!)
        gt_token = self.reference_tokens[min(self.gt_pos, self.maxindex)]
        return gt_token.unsqueeze(-1).unsqueeze(-1)  # Shape: [1, 1]

    def compute_accuracy(self) -> tuple[float, float]:
        """
        Compute TOP1 and TOP5 token accuracy.

        TOP1: Percentage of positions where predicted token (on TT device) matches predicted token of reference model
        TOP5: Percentage of positions where predicted token (on TT device) is within the top 5 predicted tokens of reference model

        Returns:
            Tuple of (top1_accuracy, top5_accuracy) as floats in range [0.0, 1.0]
        """
        count_top1 = 0
        count_top5 = 0
        matching_sz = min(len(self.reference_tokens), len(self.store_predicted_tokens))

        for i in range(matching_sz):
            # TOP1: Check if prediction matches first entry in top5_tokens
            if self.top5_tokens[i, 0].item() == self.store_predicted_tokens[i]:
                count_top1 += 1

            # TOP5: Check if prediction is in any of the top 5
            if self.store_predicted_tokens[i] in self.top5_tokens[i, :]:
                count_top5 += 1

        accuracy_top1 = count_top1 / matching_sz if matching_sz > 0 else 0.0
        accuracy_top5 = count_top5 / matching_sz if matching_sz > 0 else 0.0

        return accuracy_top1, accuracy_top5

    @staticmethod
    def _list_available_models(reference_dir: str) -> list[str]:
        """List available .refpt models in reference directory."""
        if not os.path.exists(reference_dir):
            return []
        files = [f.replace(".refpt", "") for f in os.listdir(reference_dir) if f.endswith(".refpt")]
        return files

    @staticmethod
    def get_model_name_from_variant(model_loader, variant) -> str:
        """
        Convert model variant to reference file name format.

        Args:
            model_loader: Model loader instance
            variant: Model variant enum

        Returns:
            Model name string matching .refpt file naming convention

        Example:
            LLAMA_3_2_1B_INSTRUCT -> "Llama-3.2-1B-Instruct"
        """
        # Get the config for this variant from _VARIANTS dict
        config = model_loader._VARIANTS.get(variant)
        if config is None:
            raise ValueError(f"Variant {variant} not found in model loader")

        # Get pretrained model name from config
        pretrained_name = config.pretrained_model_name

        # Extract model name from HuggingFace format
        # e.g., "meta-llama/Llama-3.2-1B-Instruct" -> "Llama-3.2-1B-Instruct"
        if "/" in pretrained_name:
            model_name = pretrained_name.split("/")[-1]
        else:
            model_name = pretrained_name

        return model_name
