# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Model wrapper utilities for extracting and wrapping transformer blocks/layers.

Separate wrappers for:
- Vision Transformers (ViT, Swin, SegFormer)
- Encoders (BERT-like)
- LLMs (decoder-based like LLaMA)
"""

import torch
import torch.nn as nn


# ============================================================================
# Decoder Wrappers (decoder-only models like LLaMA, Qwen, Falcon, etc.)
# ============================================================================


class DecoderBlockWrapper(nn.Module):
    """Wrapper for a single decoder block.

    Input: hidden_states [batch, seq_len, hidden_size]
    Output: hidden_states [batch, seq_len, hidden_size]
    """

    def __init__(self, block, rotary_emb, hidden_size):
        super().__init__()
        self.block = block
        self.rotary_emb = rotary_emb
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        block_output = self.block(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        )

        if isinstance(block_output, tuple):
            return block_output[0]
        return block_output


def extract_decoder_block(model, block_idx: int = 0):
    """Extract a single decoder block from a decoder-only model.

    Args:
        model: Full decoder model (LLaMA, Qwen, Falcon, etc.)
        block_idx: Which block to extract

    Returns:
        DecoderBlockWrapper
    """
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError(f"Cannot find decoder layers in model: {type(model)}")

    block = model.model.layers[block_idx]
    rotary_emb = model.model.rotary_emb
    hidden_size = model.config.hidden_size

    wrapper = DecoderBlockWrapper(block, rotary_emb, hidden_size)
    wrapper.eval()
    return wrapper


def make_decoder_single_layer(model, layer_idx: int = 0):
    """Modify a decoder model in-place to use only one transformer layer.

    Simulates num_layers=1 by replacing the layers list with a single layer.
    Reuses the entire model structure (embed, rotary, norm, lm_head).

    Args:
        model: Full decoder model (LLaMA, Qwen, Falcon, etc.)
        layer_idx: Which layer to keep

    Returns:
        The same model, modified to have only one layer
    """
    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise ValueError(f"Cannot find decoder layers in model: {type(model)}")

    original_layers = model.model.layers
    model.model.layers = nn.ModuleList([original_layers[layer_idx]])

    # Update config if it has num_hidden_layers
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = 1

    return model


# ============================================================================
# Encoder Wrappers (BERT-like models)
# ============================================================================


class EncoderBlockWrapper(nn.Module):
    """Wrapper for a single encoder transformer block.

    Input: hidden_states [batch, seq_len, hidden_size]
    Output: hidden_states [batch, seq_len, hidden_size]
    """

    def __init__(self, block, hidden_size):
        super().__init__()
        self.block = block
        self.hidden_size = hidden_size

    def forward(self, hidden_states, attention_mask=None):
        try:
            block_output = self.block(hidden_states, attention_mask)
        except TypeError:
            block_output = self.block(hidden_states)

        if isinstance(block_output, tuple):
            return block_output[0]
        return block_output


def make_encoder_single_layer(model, layer_idx: int = 0):
    """Modify an encoder model in-place to use only one transformer layer.

    Simulates a single-layer encoder by replacing the layers list with a single layer.
    Reuses the entire model structure (embeddings, pooler, etc.).

    Args:
        model: Full encoder model (BERT, etc.)
        layer_idx: Which layer to keep

    Returns:
        The same model, modified to have only one layer
    """
    # Find and modify encoder layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        original_layers = model.encoder.layer
        model.encoder.layer = nn.ModuleList([original_layers[layer_idx]])
    elif hasattr(model, "model") and hasattr(model.model, "encoder"):
        original_layers = model.model.encoder.layer
        model.model.encoder.layer = nn.ModuleList([original_layers[layer_idx]])
    else:
        raise ValueError(f"Cannot find encoder layers in model: {type(model)}")

    # Update config if it has num_hidden_layers
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = 1

    return model


def extract_encoder_block(model, block_idx: int = 0):
    """Extract a single block from an encoder model.

    Args:
        model: Full encoder model (BERT, etc.)
        block_idx: Which block to extract

    Returns:
        EncoderBlockWrapper
    """
    config = getattr(model, "config", None)

    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        block = model.encoder.layer[block_idx]
    elif hasattr(model, "model") and hasattr(model.model, "encoder"):
        block = model.model.encoder.layer[block_idx]
    elif hasattr(model, "layers"):
        block = model.layers[block_idx]
    else:
        raise ValueError(f"Cannot find encoder blocks in model: {type(model)}")

    hidden_size = config.hidden_size if config else 768
    wrapper = EncoderBlockWrapper(block, hidden_size)
    wrapper.eval()
    return wrapper


# ============================================================================
# Vision Transformer Wrappers (ViT)
# ============================================================================


class VisionBlockWrapper(nn.Module):
    """Wrapper for a single vision transformer block.

    Supports ViT, Swin, and SegFormer block types.
    Input: hidden_states [batch, seq_len, hidden_size]
    Output: hidden_states [batch, seq_len, hidden_size]
    """

    def __init__(self, block, hidden_size, block_type="vit"):
        super().__init__()
        self.block = block
        self.hidden_size = hidden_size
        self.block_type = block_type

    def forward(self, hidden_states, attention_mask=None):
        if self.block_type == "swin":
            # Swin: expects 4D input (B, H, W, C)
            batch_size, seq_len, hidden = hidden_states.shape
            hw = int(seq_len**0.5)
            x = hidden_states.view(batch_size, hw, hw, hidden)
            block_output = self.block(x)
            block_output = block_output.view(batch_size, -1, hidden)
        elif self.block_type == "segformer":
            # SegFormer: needs height and width
            batch_size, seq_len, _ = hidden_states.shape
            hw = int(seq_len**0.5)
            block_output = self.block(hidden_states, hw, hw)
        else:
            # ViT: standard forward
            block_output = self.block(hidden_states)

        if isinstance(block_output, tuple):
            return block_output[0]
        return block_output


def extract_vision_block(model, block_idx: int = 0):
    """Extract a single block from a vision transformer model.

    Auto-detects model type (ViT, Swin, SegFormer) and extracts the appropriate block.

    Args:
        model: Full vision model (ViT, Swin, or SegFormer)
        block_idx: Which block to extract

    Returns:
        VisionBlockWrapper with appropriate block_type
    """
    config = getattr(model, "config", None)

    # ViT (HuggingFace)
    if hasattr(model, "vit"):
        block = model.vit.encoder.layer[block_idx]
        hidden_size = config.hidden_size if config else 768
        block_type = "vit"

    # Swin (torchvision)
    elif hasattr(model, "features") and hasattr(model, "head"):
        block = model.features[1][block_idx]
        hidden_size = 96
        block_type = "swin"

    # SegFormer (HuggingFace)
    elif hasattr(model, "segformer"):
        block = model.segformer.encoder.block[0][block_idx]
        hidden_size = config.hidden_sizes[0] if config else 32
        block_type = "segformer"

    else:
        raise ValueError(f"Unknown vision model type: {type(model)}")

    wrapper = VisionBlockWrapper(block, hidden_size, block_type=block_type)
    wrapper.eval()
    return wrapper


class SingleLayerVisionWrapper(nn.Module):
    """Wrapper for single-layer vision model testing.

    Extracts patch_embed + 1 transformer layer + norm, outputs features.
    No classification head to avoid dimension mismatches.

    Input: image [batch, channels, height, width]
    Output: features [batch, ...]
    """

    def __init__(self, model, layer_idx=0):
        super().__init__()
        self.model_type = self._detect_type(model)
        self._setup(model, layer_idx)

    def _detect_type(self, model):
        if hasattr(model, "vit"):
            return "vit_hf"
        elif hasattr(model, "features") and hasattr(model, "head"):
            return "swin_torchvision"
        elif hasattr(model, "segformer"):
            return "segformer"
        else:
            raise ValueError(f"Unknown vision model type: {type(model)}")

    def _setup(self, model, layer_idx):
        if self.model_type == "vit_hf":
            # Prune to 1 encoder layer, keep embeddings + layernorm + classifier
            model.vit.encoder.layer = nn.ModuleList([model.vit.encoder.layer[layer_idx]])
            self.model = model

        elif self.model_type == "swin_torchvision":
            # Prune to patch_embed + 1 block, skip other stages/norm/head
            model.features = nn.Sequential(
                model.features[0],  # patch_embed
                nn.Sequential(model.features[1][layer_idx])  # 1 block from stage 0
            )
            self.features = model.features

        elif self.model_type == "segformer":
            # Prune to 1 stage with 1 block, skip decode head
            encoder = model.segformer.encoder
            keep_block_idx = min(1, len(encoder.block[0]) - 1)  # Use block 1 for SegformerDropPath
            encoder.patch_embeddings = nn.ModuleList([encoder.patch_embeddings[0]])
            encoder.block = nn.ModuleList([nn.ModuleList([encoder.block[0][keep_block_idx]])])
            encoder.layer_norm = nn.ModuleList([encoder.layer_norm[0]])
            self.encoder = encoder

    def forward(self, x):
        if self.model_type == "vit_hf":
            return self.model(x).logits

        elif self.model_type == "swin_torchvision":
            return self.features(x)

        elif self.model_type == "segformer":
            return self.encoder(x).last_hidden_state


def extract_vision_single_layer_model(model, layer_idx: int = 0):
    """Extract a single-layer vision model for testing.

    Creates a wrapper with patch_embed + 1 layer + norm.
    Outputs features (no classification head to avoid dimension mismatches).

    Supports ViT, Swin, and SegFormer architectures.

    Args:
        model: Full vision transformer model
        layer_idx: Which layer to use

    Returns:
        SingleLayerVisionWrapper
    """
    wrapper = SingleLayerVisionWrapper(model, layer_idx)
    wrapper.eval()
    return wrapper


# ============================================================================
# Generic extraction function (auto-detects model type)
# ============================================================================


def extract_single_block(model, block_idx: int = 0):
    """Extract a single block from any transformer model.

    Auto-detects model type and uses appropriate wrapper.

    Args:
        model: Full transformer model
        block_idx: Which block to extract

    Returns:
        Appropriate block wrapper for the model type
    """
    # Decoder (LLaMA, Qwen, Falcon, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return extract_decoder_block(model, block_idx)

    # Vision models (ViT, Swin, SegFormer)
    if hasattr(model, "vit") or hasattr(model, "segformer") or (hasattr(model, "features") and hasattr(model, "head")):
        return extract_vision_block(model, block_idx)

    # Encoder (BERT-like)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return extract_encoder_block(model, block_idx)
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return extract_encoder_block(model, block_idx)
    if hasattr(model, "layers"):
        return extract_encoder_block(model, block_idx)

    raise ValueError(f"Cannot detect model type for: {type(model)}")
