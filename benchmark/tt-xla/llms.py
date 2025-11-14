# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import os

from llm_benchmark import benchmark_llm_torch_xla

# Defaults for all llms
DEFAULT_OPTIMIZER_ENABLED = False
DEFAULT_MEMORY_LAYOUT_ANALYSIS = False
DEFAULT_TRACE_ENABLED = False
DEFAULT_BATCH_SIZE = 1
DEFAULT_LOOP_COUNT = 1
DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_MEASURE_CPU = False
DEFAULT_TASK = "text-generation"
DEFAULT_EXPERIMENTAL_COMPILE = True


def test_llm(
    ModelLoaderModule,
    variant,
    output,
    optimizer_enabled=DEFAULT_OPTIMIZER_ENABLED,
    memory_layout_analysis=DEFAULT_MEMORY_LAYOUT_ANALYSIS,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
    data_format=DEFAULT_DATA_FORMAT,
    measure_cpu=DEFAULT_MEASURE_CPU,
    task=DEFAULT_TASK,
    experimental_compile=DEFAULT_EXPERIMENTAL_COMPILE,
):
    """Test LLM model with the given variant and optional configuration overrides.

    Args:
        variant: Model variant identifier
        output: Path to save benchmark results as JSON
        optimizer_enabled: Enable optimizer (overrides config)
        memory_layout_analysis: Enable memory layout analysis (overrides config)
        trace_enabled: Enable trace (overrides config)
        batch_size: Batch size (overrides config)
        loop_count: Number of benchmark iterations (overrides config)
        input_sequence_length: Input sequence length (overrides config)
        data_format: Data format (overrides config)
        measure_cpu: Measure CPU FPS (overrides config)
        task: Task type (overrides config)
    """
    model_loader = ModelLoaderModule(variant=variant)

    print(f"Running LLM benchmark for variant: {variant}")
    print(
        f"""Configuration:
    optimizer_enabled={optimizer_enabled}
    memory_layout_analysis={memory_layout_analysis}
    trace_enabled={trace_enabled}
    batch_size={batch_size}
    loop_count={loop_count}
    input_sequence_length={input_sequence_length}
    data_format={data_format}
    measure_cpu={measure_cpu}
    task={task}
    experimental_compile={experimental_compile}
    """
    )

    results = benchmark_llm_torch_xla(
        optimizer_enabled=optimizer_enabled,
        memory_layout_analysis=memory_layout_analysis,
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
    )

    if output:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_loader.get_model_info(variant=variant).name

        with open(output, "w") as file:
            json.dump(results, file, indent=2)


def test_llama_3_2_1B(output):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.LLAMA_3_2_1B_INSTRUCT, output=output)


def test_llama_3_2_3B(output):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.LLAMA_3_2_3B_INSTRUCT, output=output)


def test_gemma_1_1_2b(output):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    test_llm(
        ModelLoaderModule=ModelLoader, variant=ModelVariant.GEMMA_1_1_2B_IT, output=output, experimental_compile=False
    )


def test_gemma_2_2b(output):
    from third_party.tt_forge_models.gemma.pytorch.loader import ModelLoader, ModelVariant

    test_llm(
        ModelLoaderModule=ModelLoader, variant=ModelVariant.GEMMA_2_2B_IT, output=output, experimental_compile=False
    )


def test_phi1(output):
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.PHI1, output=output)


def test_phi1_5(output):
    from third_party.tt_forge_models.phi1_5.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.PHI1_5, output=output)


def test_phi2(output):
    from third_party.tt_forge_models.phi2.causal_lm.pytorch.loader import ModelLoader, ModelVariant

    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.PHI2, output=output)


def test_falcon3_1b(output):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.FALCON_1B, output=output)


def test_falcon3_3b(output):
    from third_party.tt_forge_models.falcon.pytorch.loader import ModelLoader, ModelVariant

    test_llm(ModelLoaderModule=ModelLoader, variant=ModelVariant.FALCON_3B, output=output)
