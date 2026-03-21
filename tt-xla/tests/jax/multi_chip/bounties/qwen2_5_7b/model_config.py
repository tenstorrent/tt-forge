# SPDX-License-Identifier: MIT

from typing import NamedTuple, Tuple, Dict, Any
import jax.numpy as jnp


class Qwen25Config(NamedTuple):
    """Configuration class for Qwen2.5-7B model with tensor parallelism settings."""

    # Model architecture parameters
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rope_theta: float = 1000000.0

    # Activation and normalization
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6

    # Dropout and regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Tensor parallelism configuration
    tensor_parallel_size: int = 8
    mesh_shape: Tuple[int, int] = (1, 8)  # (dp, tp)

    # Sharding specifications
    param_sharding: Dict[str, Any] = None

    # Compute and precision settings
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16

    # Training related
    use_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.param_sharding is None:
            object.__setattr__(self, 'param_sharding', self._get_default_sharding_specs())

    def _get_default_sharding_specs(self) -> Dict[str, Any]:
        """Define default parameter sharding specifications for tensor parallelism."""
        return {
            # Embedding layers
            'token_embeddings': ('vocab', None),
            'position_embeddings': (None, None),

            # Attention layers
            'attention_wq': ('fsdp', 'tp'),
            'attention_wk': ('fsdp', 'tp'),
            'attention_wv': ('fsdp', 'tp'),
            'attention_wo': ('tp', 'fsdp'),

            # MLP layers
            'mlp_gate_proj': ('fsdp', 'tp'),
            'mlp_up_proj': ('fsdp', 'tp'),
            'mlp_down_proj': ('tp', 'fsdp'),

            # Layer norms
            'input_layernorm': ('fsdp',),
            'post_attention_layernorm': ('fsdp',),
            'final_layernorm': ('fsdp',),

            # Output layer
            'lm_head': ('fsdp', 'tp'),
        }

    @property
    def kv_heads_per_tp(self) -> int:
        """Number of KV heads per tensor parallel device."""
        return self.num_key_value_heads // self.tensor_parallel_size

    @property
    def q_heads_per_tp(self) -> int:
        """Number of query heads per tensor parallel device."""
        return self.num_attention_heads // self.tensor_parallel_size

    @property
    def hidden_size_per_tp(self) -> int:
        """Hidden size per tensor parallel device."""
        return self.hidden_size // self.tensor_parallel_size

    @property
    def intermediate_size_per_tp(self) -> int:
        """Intermediate size per tensor parallel device."""
        return self.intermediate_size // self.tensor_parallel_size


def get_qwen25_7b_config(
    tensor_parallel_size: int = 8,
    mesh_shape: Tuple[int, int] = (1, 8),
    dtype: jnp.dtype = jnp.bfloat16
) -> Qwen25Config:
    """Get Qwen2.5-7B configuration with specified tensor parallelism settings."""
    return Qwen25Config(
        tensor_parallel_size=tensor_parallel_size,
        mesh_shape=mesh_shape,
        dtype=dtype,
        param_dtype=dtype
    )
