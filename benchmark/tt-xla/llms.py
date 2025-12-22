# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Type
from loguru import logger

from benchmark.utils import sanitize_filename
from llm_benchmark import benchmark_llm_torch_xla

import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
import numpy as np


# =============================================================================
# Layer-Only Wrapper (handles rotary embeddings properly)
# =============================================================================

import torch
import torch.nn as nn

class DecoderLayerWrapper(nn.Module):
    """Wrapper that runs a decoder layer with proper rotary embeddings.
    
    Handles the complexity of computing position_embeddings (cos/sin) that
    modern transformer layers require.
    """
    def __init__(self, decoder_layer, rotary_emb, config):
        super().__init__()
        self.layer = decoder_layer
        self.rotary_emb = rotary_emb
        self.config = config
        self.hidden_size = config.hidden_size
        
    def forward(self, hidden_states, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Compute rotary embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # Run the decoder layer
        layer_output = self.layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=None,
            use_cache=False,
        )
        
        # layer_output is a tuple, first element is hidden_states
        return layer_output[0]


def _supports_num_layers(loader_class: Type) -> bool:
    """Check if a model loader class supports the num_layers parameter."""
    import inspect
    try:
        sig = inspect.signature(loader_class.__init__)
        return "num_layers" in sig.parameters
    except (ValueError, TypeError):
        return False


def _create_single_block_loader(BaseLoaderClass: Type, layer_idx: int = 0) -> Type:
    """Create a model loader that returns only the decoder layer (no embedding/lm_head).
    
    Returns a DecoderLayerWrapper that handles rotary embeddings properly.
    Input: hidden_states [batch, seq_len, hidden_size]
    Output: hidden_states [batch, seq_len, hidden_size]
    
    If num_layers is supported, loads with 1 layer for efficiency.
    Otherwise, loads full model and extracts the first layer.
    """
    use_num_layers = _supports_num_layers(BaseLoaderClass)
    
    class LayerOnlyModelLoader(BaseLoaderClass):
        def __init__(self, variant=None):
            if use_num_layers:
                # Load with only 1 layer to avoid loading unnecessary layers
                super().__init__(variant=variant, num_layers=1)
            else:
                # Load full model - will extract first layer later
                print(f"Note: {BaseLoaderClass.__name__} doesn't support num_layers - loading full model")
                super().__init__(variant=variant)
        
        def load_model(self, dtype_override=None):
            # Load model
            full_model = super().load_model(dtype_override=dtype_override)
            
            # Extract decoder layer and rotary_emb
            decoder_layer = full_model.model.layers[layer_idx]
            rotary_emb = full_model.model.rotary_emb
            config = full_model.config
            
            # Create wrapper
            wrapper = DecoderLayerWrapper(decoder_layer, rotary_emb, config)
            wrapper.eval()
            
            logger.info(f"🔧 Single block mode: returning wrapped layer {layer_idx}")
            return wrapper
        
        def load_tokenizer(self):
            # We don't need a tokenizer for layer-only mode
            return None
    
    return LayerOnlyModelLoader


# =============================================================================
# Defaults for all LLMs
# =============================================================================

DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_MEMORY_LAYOUT_ANALYSIS = False
DEFAULT_TRACE_ENABLED = False
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOOP_COUNT = 1
DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_MEASURE_CPU = False
DEFAULT_TASK = "text-generation"
DEFAULT_EXPERIMENTAL_COMPILE = True
DEFAULT_ENABLE_WEIGHT_BFP8_CONVERSION = True
DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION = False
DEFAULT_REQUIRED_PCC = 0.95


def default_read_logits_fn(output):
    return output.logits


def test_llm(
    ModelLoaderModule,
    variant,
    output_file,
    single_block=False,
    single_layer=False,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
    data_format=DEFAULT_DATA_FORMAT,
    measure_cpu=DEFAULT_MEASURE_CPU,
    task=DEFAULT_TASK,
    experimental_compile=DEFAULT_EXPERIMENTAL_COMPILE,
    enable_weight_bfp8_conversion=DEFAULT_ENABLE_WEIGHT_BFP8_CONVERSION,
    experimental_enable_permute_matmul_fusion=DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION,
    read_logits_fn=default_read_logits_fn,
    mesh_config_fn=None,
    shard_spec_fn=None,
    arch=None,
    required_pcc=DEFAULT_REQUIRED_PCC,
):
    """Test LLM model with the given variant and optional configuration overrides.

    Args:
        variant: Model variant identifier
        output_file: Path to save benchmark results as JSON
        single_block: If True, compile and export decoder block only (no embedding, no lm_head)
        single_layer: If True, compile and export single layer model (full model with 1 layer)
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_sequence_length: Input sequence length
        data_format: Data format
        measure_cpu: Measure CPU FPS
        task: Task type
        experimental_compile: Enable experimental compile
        enable_weight_bfp8_conversion: Enable BFP8 weight conversion
        experimental_enable_permute_matmul_fusion: Enable permute matmul fusion optimization
        read_logits_fn: Function to extract logits from model output
        required_pcc: Required PCC threshold
    """
    # Apply layer modifications if requested
    if single_block:
        # Block-only: just the decoder layer, no embedding/lm_head
        # If num_layers not supported, loads full model then extracts first layer
        ModelLoaderModule = _create_single_block_loader(ModelLoaderModule)
        model_loader = ModelLoaderModule(variant=variant)
    elif single_layer:
        # Single-layer: full model with 1 layer (or full model if num_layers not supported)
        if _supports_num_layers(ModelLoaderModule):
            model_loader = ModelLoaderModule(variant=variant, num_layers=1)
        else:
            print(f"Note: {ModelLoaderModule.__name__} doesn't support num_layers - loading full model")
            model_loader = ModelLoaderModule(variant=variant)
    else:
        model_loader = ModelLoaderModule(variant=variant)
    # Sanitize variant name for safe filesystem usage
    sanitized_variant = sanitize_filename(str(variant))
    ttnn_perf_metrics_output_file = f"tt_xla_{sanitized_variant}_perf_metrics"

    print(f"Running LLM benchmark for variant: {variant}")
    print(
        f"""Configuration:
    single_block={single_block}
    single_layer={single_layer}
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    batch_size={batch_size}
    loop_count={loop_count}
    input_sequence_length={input_sequence_length}
    data_format={data_format}
    measure_cpu={measure_cpu}
    task={task}
    experimental_compile={experimental_compile}
    enable_weight_bfp8_conversion={enable_weight_bfp8_conversion}
    experimental_enable_permute_matmul_fusion={experimental_enable_permute_matmul_fusion}
    required_pcc={required_pcc}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_llm_torch_xla(
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        model_loader=model_loader,
        model_variant=variant,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        measure_cpu=measure_cpu,
        input_sequence_length=input_sequence_length,
        training=False,
        experimental_compile=experimental_compile,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        read_logits_fn=read_logits_fn,
        mesh_config_fn=mesh_config_fn,
        shard_spec_fn=shard_spec_fn,
        arch=arch,
        required_pcc=required_pcc,
        single_block=single_block,
        single_layer=single_layer,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_loader.get_model_info(variant=variant).name

        # LLM-specific perf metrics handling: Use only decode graph (second file)
        # LLMs split into 2 graphs: prefill (index 0) and decode (index 1)
        # Only decode is relevant for throughput
        base_name = os.path.basename(ttnn_perf_metrics_output_file)
        perf_files = [f for f in os.listdir(".") if f.startswith(base_name) and f.endswith(".json")]
        perf_files = sorted(perf_files)

        if len(perf_files) == 2:
            # Use only the decode graph (second file)
            decode_perf_file = perf_files[1]
            print(f"Using decode graph perf metrics from: {decode_perf_file}")

            with open(decode_perf_file, "r") as f:
                perf_metrics_data = json.load(f)

            if "summary" in perf_metrics_data and isinstance(perf_metrics_data["summary"], dict):
                summary = perf_metrics_data["summary"]
                results["config"]["ttnn_total_ops"] = summary.get("total_ops", 0)
                results["config"]["ttnn_total_shardable_ops"] = summary.get("total_shardable_ops", 0)
                results["config"]["ttnn_effectively_sharded_ops"] = summary.get("effectively_sharded_ops", 0)
                results["config"]["ttnn_system_memory_ops"] = summary.get("system_memory_ops", 0)
                results["config"]["ttnn_effectively_sharded_percentage"] = summary.get(
                    "effectively_sharded_percentage", 0.0
                )
                results["config"]["ttnn_num_graphs"] = 2  # prefill + decode
        else:
            logger.warning(
                f"Expected 2 perf metrics files (prefill + decode) for LLM, but found {len(perf_files)}: {perf_files}. "
                f"Skipping perf metrics."
            )
            results["config"]["ttnn_num_graphs"] = len(perf_files)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_llm_tp(ModelLoaderModule, variant, output_file):
    # Need to define arch since get_xla_device_arch() doesn't work when spmd is enabled
    arch = "wormhole_llmbox"
    mesh_config_fn = ModelLoaderModule.get_mesh_config
    shard_spec_fn = ModelLoaderModule.load_shard_spec

    test_llm(
        ModelLoaderModule=ModelLoaderModule,
        variant=variant,
        output_file=output_file,
        mesh_config_fn=mesh_config_fn,
        shard_spec_fn=shard_spec_fn,
        batch_size=32,
        input_sequence_length=128,
        arch=arch,
    )


# =============================================================================
# LLAMA Tests
# =============================================================================

def test_llama_3_2_1b(output_file, single_block, single_layer):
    """Test Llama 3.2 1B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.LLAMA_3_2_1B_INSTRUCT,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_llama_3_2_3b(output_file, single_block, single_layer):
    """Test Llama 3.2 3B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.LLAMA_3_2_3B_INSTRUCT,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


# =============================================================================
# GEMMA Tests
# =============================================================================

def test_gemma_1_1_2b(output_file, single_block, single_layer):
    """Test Gemma 1.1 2B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.GEMMA_1_1_2B_IT,
             output_file=output_file, experimental_compile=False, single_block=single_block, single_layer=single_layer)


def test_gemma_2_2b(output_file, single_block, single_layer):
    """Test Gemma 2 2B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.GEMMA_2_2B_IT,
             output_file=output_file, experimental_compile=False, single_block=single_block, single_layer=single_layer)


# =============================================================================
# PHI Tests
# =============================================================================

def test_phi_1(output_file, single_block, single_layer):
    """Test Phi 1 model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.PHI_1,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_phi_1_5(output_file, single_block, single_layer):
    """Test Phi 1.5 model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.phi1_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.PHI_1_5,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_phi_2(output_file, single_block, single_layer):
    """Test Phi 2 model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.phi2.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.PHI_2,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


# =============================================================================
# FALCON Tests
# =============================================================================

# Falcon returns tuple format: (logits, past_key_values, ...)
_falcon_read_logits_fn = lambda output: output[0]


def test_falcon_3_1b(output_file, single_block, single_layer):
    """Test Falcon 3 1B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.FALCON_3_1B,
             output_file=output_file, read_logits_fn=_falcon_read_logits_fn, single_block=single_block, single_layer=single_layer)


def test_falcon_3_3b(output_file, single_block, single_layer):
    """Test Falcon 3 3B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.FALCON_3_3B,
             output_file=output_file, read_logits_fn=_falcon_read_logits_fn, single_block=single_block, single_layer=single_layer)


# =============================================================================
# QWEN Tests
# =============================================================================

def test_qwen_2_5_0_5b(output_file, single_block, single_layer):
    """Test Qwen 2.5 0.5B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.QWEN_2_5_0_5B_INSTRUCT,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_qwen_2_5_1_5b(output_file, single_block, single_layer):
    """Test Qwen 2.5 1.5B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.QWEN_2_5_1_5B_INSTRUCT,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_qwen_3_0_6b(output_file, single_block, single_layer):
    """Test Qwen 3 0.6B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.QWEN_3_0_6B,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_qwen_3_1_7b(output_file, single_block, single_layer):
    """Test Qwen 3 1.7B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.QWEN_3_1_7B,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_qwen_3_4b(output_file, single_block, single_layer):
    """Test Qwen 3 4B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.QWEN_3_4B,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_qwen_2_5_3b(output_file, single_block, single_layer):
    """Test Qwen 2.5 3B model. Use --generate-block-test for single decode block, or --generate-layer-test for single layer (prefill, decode)."""
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.QWEN_2_5_3B_INSTRUCT,
             output_file=output_file, single_block=single_block, single_layer=single_layer)


def test_qwen_3_8b(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_8B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_qwen_2_5_7b(output_file):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_7B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: KeyError: "L['self'].model.lifted_tensor_0"
def test_gemma_1_1_7b(output_file):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_1_1_7B_IT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: TypeError: Phi3ForCausalLM.forward() got an unexpected keyword argument 'cache_position'
def test_phi_3_mini(output_file):
    from third_party.tt_forge_models.phi3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINI_4K
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: KeyError: 'lifted_tensor_0'
def test_phi_3_5_mini(output_file):
    from third_party.tt_forge_models.phi3.phi_3_5.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINI_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: AttributeError: 'MambaConfig' object has no attribute 'num_attention_heads'
def test_mamba_2_8b(output_file):
    from third_party.tt_forge_models.mamba.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MAMBA_2_8B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: ValueError: Asking to pad but the tokenizer does not have a padding token
def test_falcon_3_7b(output_file):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant
    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.FALCON_3_7B,
             output_file=output_file, read_logits_fn=_falcon_read_logits_fn)


def test_mistral_7b(output_file):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_7B_INSTRUCT_V03
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_ministral_8b(output_file):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINISTRAL_8B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_llama_3_1_8b(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_falcon3_7b_tp(output_file):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_7B
    test_llm_tp(ModelLoader, variant, output_file)


def test_falcon3_10b_tp(output_file):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_10B
    test_llm_tp(ModelLoader, variant, output_file)


def test_llama_3_1_8b_instruct_tp(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file)


def test_mistral_7b_tp(output_file):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_7B_INSTRUCT_V03
    test_llm_tp(ModelLoader, variant, output_file)


def test_ministral_8b_tp(output_file):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINISTRAL_8B
    test_llm_tp(ModelLoader, variant, output_file)


def test_mistral_nemo_instruct_2407_tp(output_file):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_NEMO_INSTRUCT_2407
    test_llm_tp(ModelLoader, variant, output_file)


def test_qwen_2_5_14b_instruct_tp(output_file):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_14B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file)


def test_qwen_3_0_6b_tp(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_0_6B
    test_llm_tp(ModelLoader, variant, output_file)


def test_qwen_3_1_7b_tp(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_1_7B
    test_llm_tp(ModelLoader, variant, output_file)


def test_qwen_3_8b_tp(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_8B
    test_llm_tp(ModelLoader, variant, output_file)


def test_qwen_3_14b_tp(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_14B
    test_llm_tp(ModelLoader, variant, output_file)


def test_llama_3_8b_instruct_tp(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_8B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file)


def test_llama_3_1_8b_tp(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_8B
    test_llm_tp(ModelLoader, variant, output_file)


def test_llama_3_8b_tp(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_8B
    test_llm_tp(ModelLoader, variant, output_file)
