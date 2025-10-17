# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import pytest

from llm_benchmark import benchmark_llm_torch_xla


@pytest.mark.parametrize(
    "version",
    [
        "meta-llama/Llama-3.2-1B",
    ],
)
def test_llama(version, output):
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
        huggingface_id=version,
    )

    if output:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = version

        with open(output, "w") as file:
            json.dump(results, file, indent=2)
