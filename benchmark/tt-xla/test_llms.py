# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
from loguru import logger

from benchmark.utils import sanitize_filename
from llm_benchmark import benchmark_llm_torch_xla
from utils import create_model_loader, resolve_display_name

import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
import numpy as np

# Defaults for all llms
DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_MEMORY_LAYOUT_ANALYSIS = False
DEFAULT_TRACE_ENABLED = False
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOOP_COUNT = 1
DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_DATA_FORMAT = "bfloat16"
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
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
    data_format=DEFAULT_DATA_FORMAT,
    task=DEFAULT_TASK,
    experimental_compile=DEFAULT_EXPERIMENTAL_COMPILE,
    enable_weight_bfp8_conversion=DEFAULT_ENABLE_WEIGHT_BFP8_CONVERSION,
    experimental_enable_permute_matmul_fusion=DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION,
    read_logits_fn=default_read_logits_fn,
    mesh_config_fn=None,
    shard_spec_fn=None,
    arch=None,
    required_pcc=DEFAULT_REQUIRED_PCC,
    num_layers=None,
    request=None,
):
    """Test LLM model with the given variant and optional configuration overrides.

    Args:
        variant: Model variant identifier
        output_file: Path to save benchmark results as JSON
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_sequence_length: Input sequence length
        data_format: Data format
        task: Task type
        experimental_compile: Enable experimental compile
        enable_weight_bfp8_conversion: Enable BFP8 weight conversion
        experimental_enable_permute_matmul_fusion: Enable permute matmul fusion optimization
        read_logits_fn: Function to extract logits from model output
        required_pcc: Required PCC threshold
    """
    model_loader = create_model_loader(ModelLoaderModule, num_layers=num_layers, variant=variant)
    if num_layers is not None and model_loader is None:
        pytest.fail("num_layers override requested but ModelLoader does not support it.")
    model_info_name = model_loader.get_model_info(variant=variant).name
    display_name = resolve_display_name(request=request, fallback=model_info_name)

    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    print(f"Running LLM benchmark for variant: {variant}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    batch_size={batch_size}
    loop_count={loop_count}
    input_sequence_length={input_sequence_length}
    data_format={data_format}
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
        display_name=display_name,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
        experimental_compile=experimental_compile,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        read_logits_fn=read_logits_fn,
        mesh_config_fn=mesh_config_fn,
        shard_spec_fn=shard_spec_fn,
        arch=arch,
        required_pcc=required_pcc,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

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


def test_llm_tp(ModelLoaderModule, variant, output_file, num_layers=None, request=None, **kwargs):
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
        num_layers=num_layers,
        request=request,
        **kwargs,
    )


def test_llama_3_2_1b(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_2_1B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_llama_3_2_3b(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_2_3B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_gemma_1_1_2b(output_file, num_layers, request):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_1_1_2B_IT
    experimental_compile = False
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        experimental_compile=experimental_compile,
        num_layers=num_layers,
        request=request,
    )


def test_gemma_2_2b(output_file, num_layers, request):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_2_2B_IT
    experimental_compile = False
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        experimental_compile=experimental_compile,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
    )


def test_phi1(output_file, num_layers, request):
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI1
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_phi1_5(output_file, num_layers, request):
    from third_party.tt_forge_models.phi1_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI1_5
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_phi2(output_file, num_layers, request):
    from third_party.tt_forge_models.phi2.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI2
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
    )


def test_falcon3_1b(output_file, num_layers, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_1B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        read_logits_fn=read_logits_fn,
        num_layers=num_layers,
        request=request,
    )


def test_falcon3_3b(output_file, num_layers, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_3B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        read_logits_fn=read_logits_fn,
        num_layers=num_layers,
        request=request,
    )


def test_qwen_2_5_0_5b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_0_5B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        required_pcc=0.94,
        num_layers=num_layers,
        request=request,
    )


def test_qwen_3_0_6b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_0_6B
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_qwen_3_1_7b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_1_7B
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_qwen_3_4b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_4B
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_qwen_2_5_1_5b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_1_5B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_qwen_2_5_3b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_3B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_qwen_3_8b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_8B
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_qwen_2_5_7b(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_7B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


# FAILED: KeyError: "L['self'].model.lifted_tensor_0"
def test_gemma_1_1_7b(output_file, num_layers, request):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_1_1_7B_IT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


# FAILED: TypeError: Phi3ForCausalLM.forward() got an unexpected keyword argument 'cache_position'
def test_phi3_mini(output_file, num_layers, request):
    from third_party.tt_forge_models.phi3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINI_4K
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


# FAILED: KeyError: 'lifted_tensor_0'
def test_phi3_5_mini(output_file, num_layers, request):
    from third_party.tt_forge_models.phi3.phi_3_5.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINI_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


# FAILED: AttributeError: 'MambaConfig' object has no attribute 'num_attention_heads'
def test_mamba_2_8b(output_file, num_layers, request):
    from third_party.tt_forge_models.mamba.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MAMBA_2_8B
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_falcon3_7b(output_file, num_layers, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_7B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        read_logits_fn=read_logits_fn,
        num_layers=num_layers,
        request=request,
    )


def test_mistral_7b(output_file, num_layers, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_7B_INSTRUCT_V03
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_ministral_8b(output_file, num_layers, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINISTRAL_8B
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_llama_3_1_8b(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, num_layers=num_layers, request=request
    )


def test_falcon3_7b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_7B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_falcon3_10b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_10B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_llama_3_1_8b_instruct_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_mistral_7b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_7B_INSTRUCT_V03
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_ministral_8b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINISTRAL_8B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_mistral_nemo_instruct_2407_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_NEMO_INSTRUCT_2407
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_mistral_small_24b_instruct_2501_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_SMALL_24B_INSTRUCT_2501
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_2_5_14b_instruct_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_14B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_2_5_32b_instruct_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_32B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_2_5_coder_32b_instruct_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_2_5_coder.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_3_0_6b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_0_6B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_3_1_7b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_1_7B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_3_8b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_8B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_3_14b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_14B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_qwen_3_32b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_32B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_llama_3_8b_instruct_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_8B_INSTRUCT
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_llama_3_1_8b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_8B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_llama_3_8b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_8B
    test_llm_tp(ModelLoader, variant, output_file, num_layers=num_layers, request=request)


def test_llama_3_1_70b_tp(output_file, num_layers, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_70B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        required_pcc=-1.0,
    )  # https://github.com/tenstorrent/tt-xla/issues/2976
