# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import os

from llm_benchmark import benchmark_llm_torch_xla

# Defaults for all llms
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
    measure_cpu=DEFAULT_MEASURE_CPU,
    task=DEFAULT_TASK,
    experimental_compile=DEFAULT_EXPERIMENTAL_COMPILE,
    enable_weight_bfp8_conversion=DEFAULT_ENABLE_WEIGHT_BFP8_CONVERSION,
    experimental_enable_permute_matmul_fusion=DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION,
    read_logits_fn=default_read_logits_fn,
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
        measure_cpu: Measure CPU FPS
        task: Task type
        experimental_compile: Enable experimental compile
        enable_weight_bfp8_conversion: Enable BFP8 weight conversion
        experimental_enable_permute_matmul_fusion: Enable permute matmul fusion optimization
        read_logits_fn: Function to extract logits from model output
    """
    model_loader = ModelLoaderModule(variant=variant)
    ttnn_perf_metrics_output_file = f"{variant}.json"

    print(f"Running LLM benchmark for variant: {variant}")
    print(
        f"""Configuration:
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
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_loader.get_model_info(variant=variant).name

        if os.path.exists(ttnn_perf_metrics_output_file):
            with open(ttnn_perf_metrics_output_file, "r") as f:
                perf_metrics_data = json.load(f)
            if "summary" in perf_metrics_data and isinstance(perf_metrics_data["summary"], dict):
                results["config"]["ttnn_total_ops"] = perf_metrics_data["summary"]["total_ops"]
                results["config"]["ttnn_total_shardable_ops"] = perf_metrics_data["summary"]["total_shardable_ops"]
                results["config"]["ttnn_effectively_sharded_ops"] = perf_metrics_data["summary"][
                    "effectively_sharded_ops"
                ]
                results["config"]["ttnn_effectively_sharded_percentage"] = perf_metrics_data["summary"][
                    "effectively_sharded_percentage"
                ]
                results["config"]["ttnn_system_memory_ops"] = perf_metrics_data["summary"]["system_memory_ops"]

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_llama_3_2_1b(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_2_1B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_llama_3_2_3b(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_2_3B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_gemma_1_1_2b(output_file):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_1_1_2B_IT
    experimental_compile = False
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        experimental_compile=experimental_compile,
    )


def test_gemma_2_2b(output_file):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_2_2B_IT
    experimental_compile = False
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        experimental_compile=experimental_compile,
        output_file=output_file,
    )


def test_phi1(output_file):
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI1
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_phi1_5(output_file):
    from third_party.tt_forge_models.phi1_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI1_5
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_phi2(output_file):
    from third_party.tt_forge_models.phi2.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI2
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
    )


def test_falcon3_1b(output_file):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_1B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, read_logits_fn=read_logits_fn)


def test_falcon3_3b(output_file):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_3B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        read_logits_fn=read_logits_fn,
    )


def test_qwen_2_5_0_5b(output_file):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_0_5B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_qwen_3_0_6b(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_0_6B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_qwen_3_1_7b(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_1_7B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_qwen_3_4b(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_4B
    # Disable BFP8 weight conversion due to OOM failure
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_qwen_2_5_1_5b(output_file):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_1_5B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


def test_qwen_2_5_3b(output_file):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_3B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: Out of Memory: Not enough space to allocate 100663296 B DRAM buffer across 12 banks
def test_qwen_3_8b(output_file):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_8B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: Out of Memory: Not enough space to allocate 135790592 B DRAM buffer across 12 banks
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
def test_phi3_mini(output_file):
    from third_party.tt_forge_models.phi3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINI_4K
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: KeyError: 'lifted_tensor_0'
def test_phi3_5_mini(output_file):
    from third_party.tt_forge_models.phi3.phi_3_5.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINI_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: AttributeError: 'MambaConfig' object has no attribute 'num_attention_heads'
def test_mamba_2_8b(output_file):
    from third_party.tt_forge_models.mamba.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MAMBA_2_8B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: ValueError: Asking to pad but the tokenizer does not have a padding token
def test_falcon3_7b(output_file):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_7B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file, read_logits_fn=read_logits_fn)


# FAILED: ValueError: Asking to pad but the tokenizer does not have a padding token
def test_mistral_7b(output_file):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MISTRAL_7B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: ValueError: Asking to pad but the tokenizer does not have a padding token
def test_ministral_8b(output_file):
    from third_party.tt_forge_models.mistral.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MINISTRAL_8B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)


# FAILED: Out of Memory: Not enough space to allocate 117440512 B DRAM buffer across 12 banks
def test_llama_3_1_8b(output_file):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output_file=output_file)
