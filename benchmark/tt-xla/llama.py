# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import pytest

from llm_benchmark import benchmark_llm_torch_xla


def llm_benchmark(model_loader, output):
    """Benchmark LLaMA model with given version.

    Args:
        version (str): Hugging Face model identifier for LLaMA.
        output_file (str, optional): Path to save benchmark results as JSON. Defaults to None.
    """
    results = benchmark_llm_torch_xla(
        training=False,
        batch_size=1,
        loop_count=1,
        task="text-generation",
        data_format="bfloat16",
        measure_cpu=False,
        input_sequence_length=128,
        model_loader=model_loader,
    )

    if output:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_loader.get_model_info().name

        with open(output, "w") as file:
            json.dump(results, file, indent=2)


def test_llama(variant, output):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader as LLamaLoader,
        ModelVariant as LLamaVariant,
    )

    if variant is None:
        raise ValueError("Model variant must be specified with --variant <variant_name>")

    if LLamaVariant(variant) not in LLamaLoader.query_available_variants():
        raise ValueError(f"Variant {variant} is not available for LLaMA model.")

    model_loader = LLamaLoader(variant=LLamaVariant(variant))
    llm_benchmark(model_loader, output)
