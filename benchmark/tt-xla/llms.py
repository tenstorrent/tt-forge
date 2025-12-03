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


def default_read_logits_fn(output):
    return output.logits


def test_llm(
    ModelLoaderModule,
    variant,
    output,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
    data_format=DEFAULT_DATA_FORMAT,
    measure_cpu=DEFAULT_MEASURE_CPU,
    task=DEFAULT_TASK,
    experimental_compile=DEFAULT_EXPERIMENTAL_COMPILE,
    read_logits_fn=default_read_logits_fn,
):
    """Test LLM model with the given variant and optional configuration overrides.

    Args:
        variant: Model variant identifier
        output: Path to save benchmark results as JSON
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_sequence_length: Input sequence length
        data_format: Data format
        measure_cpu: Measure CPU FPS
        task: Task type
        experimental_compile: Enable experimental compile
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
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        read_logits_fn=read_logits_fn,
    )

    if output:
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

        with open(output, "w") as file:
            json.dump(results, file, indent=2)


def test_llama_3_2_1b(output):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_2_1B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, optimization_level=0, variant=variant, output=output)


def test_llama_3_2_3b(output):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.LLAMA_3_2_3B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output)


def test_gemma_1_1_2b(output):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_1_1_2B_IT
    experimental_compile = False
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output, experimental_compile=experimental_compile)


def test_gemma_2_2b(output):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.GEMMA_2_2B_IT
    experimental_compile = False
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output, experimental_compile=experimental_compile)


def test_phi1(output):
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI1
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output)


def test_phi1_5(output):
    from third_party.tt_forge_models.phi1_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI1_5
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output)


def test_phi2(output):
    from third_party.tt_forge_models.phi2.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.PHI2
    # Disable optimizer for phi2 due to PCC issue
    test_llm(ModelLoaderModule=ModelLoader, optimization_level=0, variant=variant, output=output)


def test_falcon3_1b(output):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_1B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output, read_logits_fn=read_logits_fn)


def test_falcon3_3b(output):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.FALCON_3B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output, read_logits_fn=read_logits_fn)


def test_qwen_2_5_0_5b(output):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_2_5_0_5B_INSTRUCT
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output)


def test_qwen_3_0_6b(output):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_0_6B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output)


def test_qwen_3_1_7b(output):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_1_7B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output)


def test_qwen_3_4b(output):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_4B
    test_llm(ModelLoaderModule=ModelLoader, variant=variant, output=output)
