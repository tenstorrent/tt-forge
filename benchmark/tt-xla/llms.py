# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import os

from llm_benchmark import benchmark_llm_torch_xla

# Defaults for all llms
OPTIMIZATION_LEVEL = 0
TRACE_ENABLED = False
BATCH_SIZE = 32
LOOP_COUNT = 1
INPUT_SEQUENCE_LENGTH = 128
DATA_FORMAT = "bfloat16"
MEASURE_CPU = False
TASK = "text-generation"

# Path to the JSON config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../.github/workflows/perf-bench-matrix.json")


def load_llm_configs():
    """Load LLM model configurations from the JSON file."""
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)

    # Find the tt-xla project
    tt_xla_project = None
    for project in data:
        if project.get("project") == "tt-xla":
            tt_xla_project = project
            break

    if not tt_xla_project:
        raise ValueError("tt-xla project not found in perf-bench-matrix.json")

    # Extract LLM configs from tests that have model_config
    llm_configs = {}
    for test in tt_xla_project.get("tests", []):
        if "model_config" in test and "variant" in test:
            variant = test["variant"]
            llm_configs[variant] = test["model_config"]

    return llm_configs


# Load configurations from JSON file
LLM_MODEL_CONFIGS = load_llm_configs()


def test_llm(
    variant,
    output,
    optimization_level,
    trace_enabled,
    batch_size,
    loop_count,
    input_sequence_length,
    data_format,
    measure_cpu,
    task,
    experimental_compile,
):
    """Test LLM model with the given variant and optional configuration overrides.

    Args:
        variant: Model variant identifier
        output: Path to save benchmark results as JSON
        optimization_level: Optimization level (0, 1, or 2) (overrides config)
        trace_enabled: Enable trace (overrides config)
        batch_size: Batch size (overrides config)
        loop_count: Number of benchmark iterations (overrides config)
        input_sequence_length: Input sequence length (overrides config)
        data_format: Data format (overrides config)
        measure_cpu: Measure CPU FPS (overrides config)
        task: Task type (overrides config)
    """
    if variant is None:
        raise ValueError("Model variant must be specified with --variant <variant_name>")

    variant_config = LLM_MODEL_CONFIGS.get(variant, None)
    if not variant_config:
        raise ValueError(f"Variant {variant} is not available in LLM_MODEL_CONFIGS.")

    module_path = variant_config["model_loader_module"]
    model_loader_module = __import__(module_path, fromlist=["ModelLoader", "ModelVariant"])
    ModelLoader = model_loader_module.ModelLoader
    ModelVariant = model_loader_module.ModelVariant

    if ModelVariant(variant) not in ModelLoader.query_available_variants():
        raise ValueError(f"Variant {variant} is not available for the specified model.")
    model_variant = ModelVariant(variant)
    model_loader = ModelLoader(variant=model_variant)

    # Get config values for the model in the following order of precedence:
    # 1. Command line argument (if provided)
    # 2. Variant-specific configuration from CI JSON file
    # 3. Default constant defined at the top of this file
    optimization_level = (
        optimization_level
        if optimization_level is not None
        else variant_config.get("optimization_level", OPTIMIZATION_LEVEL)
    )
    trace_enabled = trace_enabled if trace_enabled is not None else variant_config.get("trace_enabled", TRACE_ENABLED)
    batch_size = batch_size if batch_size is not None else variant_config.get("batch_size", BATCH_SIZE)
    loop_count = loop_count if loop_count is not None else variant_config.get("loop_count", LOOP_COUNT)
    input_sequence_length = (
        input_sequence_length
        if input_sequence_length is not None
        else variant_config.get("input_sequence_length", INPUT_SEQUENCE_LENGTH)
    )
    data_format = data_format if data_format is not None else variant_config.get("data_format", DATA_FORMAT)
    measure_cpu = measure_cpu if measure_cpu is not None else variant_config.get("measure_cpu", MEASURE_CPU)
    task = task if task is not None else variant_config.get("task", TASK)
    experimental_compile = (
        experimental_compile if experimental_compile is not None else variant_config.get("experimental_compile", True)
    )

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
    """
    )

    results = benchmark_llm_torch_xla(
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        model_loader=model_loader,
        model_variant=model_variant,
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
        results["model_rawname"] = model_loader.get_model_info(variant=model_variant).name

        with open(output, "w") as file:
            json.dump(results, file, indent=2)
