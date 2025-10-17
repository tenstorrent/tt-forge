# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from llm_benchmark import benchmark_llm_torch_xla


def test_llama_3_2_3B():
    llama_version = "meta-llama/Llama-3.2-3B"

    return benchmark_llm_torch_xla(
        training=False,
        batch_size=1,
        loop_count=1,
        task="text-generation",
        data_format="bfloat16",
        measure_cpu=False,
        input_sequence_length=128,
        huggingface_id=llama_version,
    )


def test_llama_3_2_1B():
    llama_version = "meta-llama/Llama-3.2-1B"

    return benchmark_llm_torch_xla(
        training=False,
        batch_size=1,
        loop_count=1,
        task="text-generation",
        data_format="bfloat16",
        measure_cpu=False,
        input_sequence_length=128,
        huggingface_id=llama_version,
    )


def test_llama_3_1_8B():
    llama_version = "meta-llama/Llama-3.1-8B"

    return benchmark_llm_torch_xla(
        training=False,
        batch_size=1,
        loop_count=1,
        task="text-generation",
        data_format="bfloat16",
        measure_cpu=False,
        input_sequence_length=128,
        huggingface_id=llama_version,
    )
