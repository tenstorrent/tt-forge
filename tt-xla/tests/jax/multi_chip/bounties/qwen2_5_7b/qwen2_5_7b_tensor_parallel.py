# SPDX-License-Identifier: MIT

import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit
import flax.linen as nn
from flax.core import freeze
import numpy as np
from typing import Optional, Tuple, Any, Dict
import math


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.hidden_size,))

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = jnp.power(hidden_states.astype(jnp.float32), 2).mean(-1, keepdims=True)
        hidden_states = hidden_states / jnp.sqrt(variance + self.eps)
        return (self.weight * hidden_states).astype(input_dtype)


class RotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000.0

    def setup(self):
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        self.inv_freq = inv_freq

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        cos = jnp.cos(emb)[None, None, :, :]
        sin = jnp.sin(emb)[None, None, :, :]
        return cos, sin


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    cos = cos[:, :, :q.shape[-2], :]
    sin = sin[:, :, :q.shape[-2], :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    hidden_size: int
    intermediate_size: int

    def setup(self):
        self.gate_proj = nn.Dense(self.intermediate_size, use_bias=False,
                                 kernel_init=nn.initializers.normal(stddev=0.02))
        self.up_proj = nn.Dense(self.intermediate_size, use_bias=False,
                               kernel_init=nn.initializers.normal(stddev=0.02))
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False,
                                 kernel_init=nn.initializers.normal(stddev=0.02))

    def __call__(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = nn.silu(gate) * up
        return self.down_proj(intermediate)


class Qwen2Attention(nn.Module):
    hidden_size: int
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0

    def setup(self):
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=True,
                              kernel_init=nn.initializers.normal(stddev=0.02))
        self.k_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=True,
                              kernel_init=nn.initializers.normal(stddev=0.02))
        self.v_proj = nn.Dense(self.num_key_value_heads * self.head_dim, use_bias=True,
                              kernel_init=nn.initializers.normal(stddev=0.02))
        self.o_proj = nn.Dense(self.hidden_size, use_bias=False,
                              kernel_init=nn.initializers.normal(stddev=0.02))

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None,
                 past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Repeat key and value states for GQA
        if self.num_key_value_heads != self.num_heads:
            key_states = jnp.repeat(key_states, self.num_heads // self.num_key_value_heads, axis=1)
            value_states = jnp.repeat(value_states, self.num_heads // self.num_key_value_heads, axis=1)

        attn_weights = jnp.matmul(query_states, key_states.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.dropout(attn_weights, rate=self.attention_dropout, deterministic=not self.training)

        attn_output = jnp.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen2DecoderLayer(nn.Module):
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    attention_dropout: float = 0.0

    def setup(self):
        self.input_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            attention_dropout=self.attention_dropout
        )
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen25Model(nn.Module):
    vocab_size: int = 152064
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False

    def setup(self):
        self.embed_tokens = nn.Embed(self.vocab_size, self.hidden_size,
                                    embedding_init=nn.initializers.normal(stddev=0.02))

        self.layers = [
            Qwen2DecoderLayer(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                rms_norm_eps=self.rms_norm_eps,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                attention_dropout=self.attention_dropout
            ) for _ in range(self.num_hidden_layers)
        ]

        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            position_ids = jnp.arange(seq_length)[None, :]

        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_length))

        causal_mask = jnp.tril(jnp.ones((seq_length, seq_length)))
        causal_mask = jnp.where(causal_mask == 0, -jnp.inf, 0.0)
        attention_mask = attention_mask[:, None, None, :] * causal_mask[None, None, :, :]

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen25ForCausalLM(nn.Module):
    config: Dict[str, Any]

    def setup(self):
        self.model = Qwen25Model(**self.config)
        if not self.config.get('tie_word_embeddings', False):
            self.lm_head = nn.Dense(self.config['vocab_size'], use_bias=False,
                                   kernel_init=nn.initializers.normal(stddev=0.02))

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        if self.config.get('tie_word_embeddings', False):
            shared_kernel = self.model.embed_tokens.variables['params']['embedding'].T
            logits = jnp.dot(hidden_states, shared_kernel)
        else:
            logits = self.lm_head(hidden_states)

        return logits


def create_mesh_and_sharding(mesh_shape):
    """Create device mesh and sharding specifications for tensor parallelism."""
    devices = mesh_utils.create_device_mesh(mesh_shape)
    mesh = jax.sharding.Mesh(devices, ('data', 'model'))

    # Define sharding specs for different components
    sharding_specs = {
        'embed_tokens': P('data', 'model'),
        'attention': {
            'q_proj': P('data', 'model'),
            'k_proj': P('data', 'model'),
            'v_proj': P('data', 'model'),
            'o_proj': P('model', 'data'),
        },
        'mlp': {
            'gate_proj': P('data', 'model'),
            'up_proj': P('data', 'model'),
            'down_proj': P('model', 'data'),
        },
        'norm': P('data'),
        'lm_head': P('data', 'model'),
        'activations': P('data', None),
    }

    return mesh, sharding_specs


def create_qwen25_7b_config():
    """Create configuration for Qwen2.5-7B model."""
    return {
        'vocab_size': 152064,
        'hidden_size': 4096,
        'intermediate_size': 22016,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'num_key_value_heads': 32,
        'head_dim': 128,
        'max_position_embeddings': 32768,
        'rope_theta': 1000000.0,
        'rms_norm_eps': 1e-6,
        'attention_dropout': 0.0,
        'tie_word_embeddings': False,
    }


def initialize_model_parallel(mesh_shape=(2, 4), batch_size=1, seq_length=512):
    """Initialize tensor parallel Qwen2.5-7B model."""
    config = create_qwen25_7b_config()
    mesh, sharding_specs = create_mesh_and_sharding(mesh_shape)

    model = Qwen25ForCausalLM(config=config)

    # Initialize parameters with proper sharding
    key = random.PRNGKey(42)
    input_shape = (batch_size, seq_length)
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)

    @pjit(
        in_shardings=(P('data', None),),
        out_shardings=P('data', None)
    )
    def init_fn(dummy_input):
        return model.init(key, dummy_input)

    with mesh:
        params = init_fn(dummy_input)

    @pjit(
        in_shardings=(None, P('data', None)),
        out_shardings=P('data', None)
    )
    def forward_fn(params, input_ids):
        return model.apply(params, input_ids)

    return model, params, forward_fn, mesh


def run_tensor_parallel_inference(mesh_shape=(2, 4), batch_size=2, seq_length=128):
    """Run inference with tensor parallelism on different mesh configurations."""
    print(f"Initializing Qwen2.5-7B with mesh shape {mesh_shape}")

    model, params, forward_fn, mesh = initialize_model_parallel(
        mesh_shape=mesh_shape,
        batch_size=batch_size,
        seq_length=seq_length
    )

    # Create sample input
    key = random.PRNGKey(123)
    input_ids = random.randint(key, (batch_size, seq_length), 0, 152064)

    with mesh:
        print("Running forward pass...")
        logits = forward_fn(params, input_ids)
        print(f"Output shape: {logits.shape}")
        print(f"Output dtype: {logits.dtype}")

        # Get predictions for last token
        last_token_logits = logits[:, -1, :]
        predictions = jnp.argmax(last_token_logits, axis=-1)
        print(f"Predictions for last token: {predictions}")

    return logits


def test_multiple_mesh_configurations():
    """Test different mesh configurations for tensor parallelism."""
    mesh_configs = [
        (2, 4),   # 2x4 = 8 devices
        (1, 8),   # 1x8 = 8 devices
        (1, 32),  # 1x32 = 32 devices (if available)
        (8, 4),   # 8x4 = 32 devices (if available)
    ]

    for mesh_shape in mesh_configs:
        try:
            total_devices = mesh_shape[0] * mesh_shape[1]
            available_devices = jax.device_count()

            if total_devices <= available_devices:
                print(f"\n{'='*50}")
                print(f"Testing mesh shape: {mesh_shape}")
                print(f"Total devices needed: {total_devices}")
                print(f"Available devices: {available_devices}")

                logits = run_tensor_parallel_inference(
                    mesh_shape=mesh_shape,
                    batch_size=2,
                    seq_length=64
                )
                print(f"✓ Successfully ran with mesh shape {mesh_shape}")
            else:
                print(f"\n⚠️  Skipping mesh shape {mesh_shape}: needs {total_devices} devices, have {available_devices}")

        except Exception as e:
            print(f"❌ Error with mesh shape {mesh_shape}: {e}")


if __name__ == "__main__":
    print("Qwen2.5-7B Tensor Parallel Implementation")
    print(f"Available JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")

    # Test different mesh configurations
    test_multiple_mesh_configurations()

    # Run a single configuration for detailed testing
    print(f"\n{'='*50}")
    print("Running detailed test with 2x4 mesh")
    try:
        logits = run_tensor_parallel_inference(
            mesh_shape=(2, 4),
            batch_size=4,
            seq_length=256
        )
        print("✓ Detailed test completed successfully")
    except Exception as e:
        print(f"❌ Detailed test failed: {e}")
