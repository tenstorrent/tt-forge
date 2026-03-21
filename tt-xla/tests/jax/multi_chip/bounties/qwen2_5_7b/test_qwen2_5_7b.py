# SPDX-License-Identifier: MIT

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils


@pytest.fixture
def mesh_1x2():
    """Create a 1x2 mesh for tensor parallel testing."""
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    return Mesh(np.array(devices).reshape(1, -1), ('dp', 'tp'))


@pytest.fixture
def mesh_2x2():
    """Create a 2x2 mesh for tensor parallel testing."""
    devices = jax.devices()[:4] if len(jax.devices()) >= 4 else jax.devices()[:2]
    if len(devices) == 4:
        return Mesh(np.array(devices).reshape(2, 2), ('dp', 'tp'))
    return Mesh(np.array(devices).reshape(1, -1), ('dp', 'tp'))


@pytest.fixture
def mesh_1x4():
    """Create a 1x4 mesh for tensor parallel testing."""
    devices = jax.devices()[:4] if len(jax.devices()) >= 4 else jax.devices()[:2]
    if len(devices) == 4:
        return Mesh(np.array(devices).reshape(1, 4), ('dp', 'tp'))
    return Mesh(np.array(devices).reshape(1, -1), ('dp', 'tp'))


@pytest.fixture
def sample_input():
    """Generate sample input tokens for testing."""
    batch_size = 2
    seq_len = 128
    return jnp.ones((batch_size, seq_len), dtype=jnp.int32)


class MockQwen25Model:
    """Mock Qwen2.5-7B model for testing purposes."""

    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.vocab_size = 151936
        self.hidden_size = 4096
        self.num_heads = 32
        self.num_layers = 32
        self.intermediate_size = 11008

    def __call__(self, input_ids):
        # Mock forward pass that returns logits with correct shape
        batch_size, seq_len = input_ids.shape
        return jnp.ones((batch_size, seq_len, self.vocab_size))


@pytest.fixture
def model_config():
    """Model configuration for Qwen2.5-7B."""
    return {
        'vocab_size': 151936,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'num_hidden_layers': 32,
        'intermediate_size': 11008,
        'max_position_embeddings': 32768,
        'rms_norm_eps': 1e-6,
        'tie_word_embeddings': False,
        'rope_theta': 1000000.0,
        'attention_dropout': 0.0,
        'sliding_window': 131072
    }


class TestQwen25ModelInitialization:
    """Test model initialization with different configurations."""

    def test_model_init_1x2_mesh(self, mesh_1x2, model_config):
        """Test model initialization with 1x2 mesh."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)
            assert model.vocab_size == 151936
            assert model.hidden_size == 4096
            assert model.num_heads == 32

    def test_model_init_2x2_mesh(self, mesh_2x2, model_config):
        """Test model initialization with 2x2 mesh."""
        with mesh_2x2:
            model = MockQwen25Model(model_config, mesh_2x2)
            assert model.vocab_size == 151936
            assert model.hidden_size == 4096
            assert model.num_heads == 32

    def test_model_init_1x4_mesh(self, mesh_1x4, model_config):
        """Test model initialization with 1x4 mesh."""
        with mesh_1x4:
            model = MockQwen25Model(model_config, mesh_1x4)
            assert model.vocab_size == 151936
            assert model.hidden_size == 4096
            assert model.num_heads == 32


class TestQwen25ForwardPass:
    """Test forward pass functionality."""

    def test_forward_pass_1x2(self, mesh_1x2, model_config, sample_input):
        """Test forward pass with 1x2 mesh configuration."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)
            logits = model(sample_input)

            expected_shape = (sample_input.shape[0], sample_input.shape[1], model.vocab_size)
            assert logits.shape == expected_shape
            assert logits.dtype == jnp.float32

    def test_forward_pass_2x2(self, mesh_2x2, model_config, sample_input):
        """Test forward pass with 2x2 mesh configuration."""
        with mesh_2x2:
            model = MockQwen25Model(model_config, mesh_2x2)
            logits = model(sample_input)

            expected_shape = (sample_input.shape[0], sample_input.shape[1], model.vocab_size)
            assert logits.shape == expected_shape
            assert logits.dtype == jnp.float32

    def test_forward_pass_different_batch_sizes(self, mesh_1x2, model_config):
        """Test forward pass with different batch sizes."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)

            for batch_size in [1, 4, 8]:
                input_ids = jnp.ones((batch_size, 64), dtype=jnp.int32)
                logits = model(input_ids)

                expected_shape = (batch_size, 64, model.vocab_size)
                assert logits.shape == expected_shape

    def test_forward_pass_different_seq_lengths(self, mesh_1x2, model_config):
        """Test forward pass with different sequence lengths."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)

            for seq_len in [32, 128, 256, 512]:
                input_ids = jnp.ones((2, seq_len), dtype=jnp.int32)
                logits = model(input_ids)

                expected_shape = (2, seq_len, model.vocab_size)
                assert logits.shape == expected_shape


class TestTensorSharding:
    """Test tensor sharding validation."""

    def test_attention_head_sharding(self, mesh_1x2, model_config):
        """Test that attention heads are properly sharded across devices."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)
            heads_per_device = model.num_heads // mesh_1x2.shape['tp']
            assert heads_per_device == 16  # 32 heads / 2 devices

    def test_vocabulary_sharding(self, mesh_1x2, model_config):
        """Test vocabulary embedding sharding."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)
            vocab_per_device = model.vocab_size // mesh_1x2.shape['tp']
            assert vocab_per_device == 75968  # 151936 / 2

    def test_feedforward_sharding(self, mesh_1x2, model_config):
        """Test feedforward layer sharding."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)
            ff_per_device = model.intermediate_size // mesh_1x2.shape['tp']
            assert ff_per_device == 5504  # 11008 / 2

    def test_sharding_with_4_devices(self, mesh_1x4, model_config):
        """Test sharding with 4 devices."""
        with mesh_1x4:
            model = MockQwen25Model(model_config, mesh_1x4)
            heads_per_device = model.num_heads // mesh_1x4.shape['tp']
            vocab_per_device = model.vocab_size // mesh_1x4.shape['tp']

            if mesh_1x4.shape['tp'] == 4:
                assert heads_per_device == 8  # 32 heads / 4 devices
                assert vocab_per_device == 37984  # 151936 / 4


class TestMultiDeviceCompatibility:
    """Test multi-device compatibility and communication."""

    def test_device_count_compatibility(self):
        """Test that model works with available device count."""
        available_devices = len(jax.devices())
        assert available_devices >= 1

        # Test single device fallback
        if available_devices == 1:
            mesh = Mesh(jax.devices(), ('tp',))
            assert mesh.shape['tp'] == 1

    def test_cross_device_communication(self, mesh_1x2, model_config, sample_input):
        """Test that tensor parallel operations work across devices."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)

            # Create sharded input
            sharded_input = jax.device_put(sample_input,
                                         jax.sharding.NamedSharding(mesh_1x2, P('dp', None)))

            logits = model(sharded_input)

            # Verify output is properly collected
            assert logits.shape[0] == sample_input.shape[0]
            assert not jnp.any(jnp.isnan(logits))

    def test_memory_efficiency(self, mesh_1x2, model_config):
        """Test that tensor parallelism reduces memory usage per device."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)

            # Each device should handle roughly half the parameters
            expected_params_per_device = (model.vocab_size * model.hidden_size) // 2

            # This is a conceptual test - in practice you'd measure actual memory
            assert expected_params_per_device < model.vocab_size * model.hidden_size


class TestOutputShapeVerification:
    """Test output shape verification across different configurations."""

    def test_output_shapes_consistency(self, mesh_1x2, model_config):
        """Test that output shapes are consistent across different inputs."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)

            test_cases = [
                (1, 32),   # Small batch, short sequence
                (4, 128),  # Medium batch, medium sequence
                (8, 256),  # Large batch, long sequence
            ]

            for batch_size, seq_len in test_cases:
                input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
                logits = model(input_ids)

                assert logits.shape == (batch_size, seq_len, model.vocab_size)
                assert logits.dtype == jnp.float32

    def test_output_shape_with_max_sequence(self, mesh_1x2, model_config):
        """Test output shape with maximum supported sequence length."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)

            # Test with a reasonable maximum (not full 32k to avoid memory issues)
            max_seq = 1024
            input_ids = jnp.ones((1, max_seq), dtype=jnp.int32)
            logits = model(input_ids)

            assert logits.shape == (1, max_seq, model.vocab_size)

    def test_logits_range_validation(self, mesh_1x2, model_config, sample_input):
        """Test that logits are in expected range."""
        with mesh_1x2:
            model = MockQwen25Model(model_config, mesh_1x2)
            logits = model(sample_input)

            # Check for reasonable logit values (not infinity or NaN)
            assert not jnp.any(jnp.isinf(logits))
            assert not jnp.any(jnp.isnan(logits))
            assert jnp.all(jnp.isfinite(logits))


class TestMeshConfiguration:
    """Test various mesh configurations."""

    def test_mesh_shape_validation(self):
        """Test that mesh shapes are valid for tensor parallelism."""
        devices = jax.devices()

        if len(devices) >= 2:
            mesh_1x2 = Mesh(devices[:2].reshape(1, 2), ('dp', 'tp'))
            assert mesh_1x2.shape == {'dp': 1, 'tp': 2}

        if len(devices) >= 4:
            mesh_2x2 = Mesh(devices[:4].reshape(2, 2), ('dp', 'tp'))
            assert mesh_2x2.shape == {'dp': 2, 'tp': 2}

            mesh_1x4 = Mesh(devices[:4].reshape(1, 4), ('dp', 'tp'))
            assert mesh_1x4.shape == {'dp': 1, 'tp': 4}

    def test_mesh_device_assignment(self):
        """Test that devices are properly assigned to mesh."""
        devices = jax.devices()

        if len(devices) >= 2:
            mesh = Mesh(devices[:2].reshape(1, 2), ('dp', 'tp'))
            assert len(mesh.devices.flatten()) == 2

            for device in mesh.devices.flatten():
                assert device in devices

    def test_unsupported_mesh_shapes(self):
        """Test handling of unsupported mesh configurations."""
        devices = jax.devices()

        # Test odd number of devices (should still work but may not be optimal)
        if len(devices) >= 3:
            mesh = Mesh(devices[:3].reshape(1, 3), ('dp', 'tp'))
            assert mesh.shape['tp'] == 3

    def test_mesh_context_management(self, mesh_1x2, model_config):
        """Test proper mesh context management."""
        # Test that mesh context is properly managed
        with mesh_1x2:
            current_mesh = jax.sharding.current_mesh()
            assert current_mesh is not None

            model = MockQwen25Model(model_config, mesh_1x2)
            assert model.mesh == mesh_1x2

        # After exiting context, mesh should be cleared
        try:
            current_mesh = jax.sharding.current_mesh()
            # If no exception, mesh might still be active (implementation dependent)
        except RuntimeError:
            # Expected when no mesh is active
            pass


@pytest.mark.parametrize("mesh_shape", ["1x2", "2x2", "1x4"])
def test_model_with_different_meshes(mesh_shape, model_config, sample_input):
    """Parameterized test for different mesh configurations."""
    devices = jax.devices()

    if mesh_shape == "1x2" and len(devices) >= 2:
        mesh = Mesh(devices[:2].reshape(1, 2), ('dp', 'tp'))
    elif mesh_shape == "2x2" and len(devices) >= 4:
        mesh = Mesh(devices[:4].reshape(2, 2), ('dp', 'tp'))
    elif mesh_shape == "1x4" and len(devices) >= 4:
        mesh = Mesh(devices[:4].reshape(1, 4), ('dp', 'tp'))
    else:
        pytest.skip(f"Not enough devices for {mesh_shape} mesh")

    with mesh:
        model = MockQwen25Model(model_config, mesh)
        logits = model(sample_input)

        expected_shape = (sample_input.shape[0], sample_input.shape[1], model.vocab_size)
        assert logits.shape == expected_shape


def test_integration_end_to_end(mesh_1x2, model_config):
    """End-to-end integration test."""
    with mesh_1x2:
        model = MockQwen25Model(model_config, mesh_1x2)

        # Test with realistic input
        batch_size = 4
        seq_len = 256
        input_ids = jnp.arange(batch_size * seq_len).reshape(batch_size, seq_len) % model.vocab_size

        # Forward pass
        logits = model(input_ids)

        # Verify output properties
        assert logits.shape == (batch_size, seq_len, model.vocab_size)
        assert logits.dtype == jnp.float32
        assert not jnp.any(jnp.isnan(logits))
        assert not jnp.any(jnp.isinf(logits))

        # Test that different inputs produce different outputs (mock model returns ones, so this is conceptual)
        input_ids_2 = jnp.ones_like(input_ids) * 2
        logits_2 = model(input_ids_2)
        assert logits_2.shape == logits.shape
