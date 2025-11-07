# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import pytest
import socket
from typing import List, Union, Optional, Dict, Literal
from collections import defaultdict

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import numpy as np
from FlagEmbedding import BGEM3FlagModel

from benchmark.utils import get_xla_device_arch

from .utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants
OPTIMIZER_ENABLED = False
PROGRAM_CACHE_ENABLED = True
MEMORY_LAYOUT_ANALYSIS_ENABLED = False
TRACE_ENABLED = False

if PROGRAM_CACHE_ENABLED:
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"

# Machine learning task
TASK = [
    "embedding",
]

# Batch size configurations
BATCH_SIZE = [
    1,
    2,
    4,
]

# Data format configurations
DATA_FORMAT = [
    "float32",
]

# Loop count configurations
LOOP_COUNT = [1, 2, 4, 8, 16, 32]


# Adapted from FlagEmbedding's BGEM3 encode function, removing overhead handled by TT backend
def encode(
    model,
    sentences: Union[List[str], str],
    device,
    input_sequence_length: int,
    return_dense: Optional[bool] = None,
    return_sparse: Optional[bool] = None,
    return_colbert_vecs: Optional[bool] = None,
) -> Dict[
    Literal["dense_vecs", "lexical_weights", "colbert_vecs"],
    Union[np.ndarray, List[Dict[str, float]], List[np.ndarray]],
]:

    # Tokenize the input sentences and move to device
    text_input = model.tokenizer(
        sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=input_sequence_length
    )
    text_input.to(device)
    inputs = {
        "text_input": text_input,
        "return_dense": return_dense,
        "return_sparse": return_sparse,
        "return_colbert_vecs": return_colbert_vecs,
    }

    outputs = model(**inputs)

    # Process outputs to expected format
    def _process_token_weights(token_weights: np.ndarray, input_ids: list):
        result = defaultdict(int)
        unused_tokens = set()
        for _token in ["cls_token", "eos_token", "pad_token", "unk_token"]:
            if _token in model.tokenizer.special_tokens_map:
                _token_id = model.tokenizer.convert_tokens_to_ids(model.tokenizer.special_tokens_map[_token])
                unused_tokens.add(_token_id)
        for w, idx in zip(token_weights, input_ids):
            if idx not in unused_tokens and w > 0:
                idx = str(idx)
                if w > result[idx]:
                    result[idx] = w
        return result

    def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
        tokens_num = np.sum(attention_mask)
        return colbert_vecs[: tokens_num - 1]

    all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []

    batch_size = text_input["input_ids"].shape[0]
    length_sorted_idx = np.argsort([-len(text_input["input_ids"][i]) for i in range(batch_size)])

    all_dense_embeddings.append(outputs["dense_vecs"].cpu().detach())
    all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
    all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]

    token_weights = outputs["sparse_vecs"].squeeze(-1)
    all_lexical_weights.extend(
        list(
            map(
                _process_token_weights,
                token_weights.cpu().detach().numpy(),
                text_input["input_ids"].cpu().detach().numpy().tolist(),
            )
        )
    )
    all_lexical_weights = [all_lexical_weights[i] for i in np.argsort(length_sorted_idx)]

    all_colbert_vecs.extend(
        list(
            map(
                _process_colbert_vecs,
                outputs["colbert_vecs"].cpu().detach().numpy(),
                text_input["attention_mask"].cpu().detach().numpy(),
            )
        )
    )
    all_colbert_vecs = [all_colbert_vecs[i] for i in np.argsort(length_sorted_idx)]

    # return the embeddings
    return {
        "dense_vecs": all_dense_embeddings,
        "lexical_weights": all_lexical_weights,
        "colbert_vecs": all_colbert_vecs,
    }


def get_benchmark_sentences(batch_size: int) -> List[str]:
    """
    Get benchmark sentences for BGE-M3 encoding.
    Returns a list of sentences, repeating as needed to match batch_size.
    """
    # Base sentences from bge_m3_encode_cpu_test.py
    base_sentences = [
        # English
        "The quick brown fox jumps over the lazy dog while the sun shines brightly in the clear blue sky, creating beautiful shadows on the green grass below.",
        "Machine learning has revolutionized the way we process and understand vast amounts of data, enabling computers to learn patterns and make predictions without being explicitly programmed for each specific task.",
        "Climate change represents one of the most pressing challenges of our time, requiring coordinated global action to reduce greenhouse gas emissions and transition to sustainable energy sources.",
        "The development of quantum computing promises to solve complex problems that are currently beyond the reach of classical computers, potentially transforming fields like drug discovery and cryptography.",
        "In the depths of the ocean, scientists continue to discover new species that have adapted to extreme conditions, revealing the incredible diversity and resilience of life on Earth.",
        # Japanese
        "人工知能システムは医療分野にますます統合されており、医師が病気を診断し、患者の結果を予測し、個々の遺伝的プロファイルに基づいて治療計画をパーソナライズするのを支援しています。",
        "人間の脳には約860億個のニューロンが含まれており、それぞれが他のニューロンと数千の接続を形成し、意識、思考、感情を生み出す複雑なネットワークを作り出しています。",
        "ブロックチェーン技術は、データの保存と検証への革命的なアプローチとして登場し、暗号通貨を超えたさまざまな産業に分散型で透明なソリューションを提供しています。",
        # Korean
        "기후 변화는 우리 시대의 가장 시급한 과제 중 하나로, 온실가스 배출을 줄이고 지속 가능한 에너지원으로 전환하기 위한 전 세계적인 협력 행동이 필요합니다.",
        "양자 컴퓨팅의 발전은 현재 고전 컴퓨터의 능력을 넘어서는 복잡한 문제를 해결할 것을 약속하며, 잠재적으로 약물 발견과 암호화 같은 분야를 변화시킬 수 있습니다.",
    ]

    # Extend to match batch size by repeating sentences
    sentences = []
    for i in range(batch_size):
        sentences.append(base_sentences[i % len(base_sentences)])

    return sentences


def bge_m3_warmup(compiled_model, sentences_list, encode_kwargs, loop_count, device, input_sequence_length):
    """
    Warmup the BGE-M3 model for a given number of loop_count.

    Parameters:
    ----------
    compiled_model: torch.nn.Module
        The compiled model to warmup.
    sentences_list: list of list of str
        List of sentence batches to process.
    encode_kwargs: dict
        Encoding parameters (return_dense, return_sparse, return_colbert_vecs).
    loop_count: int
        The number of iterations to warmup the model.
    device: torch.device
        The device to run on.
    input_sequence_length: int
        Maximum sequence length for tokenization.
    """
    print("Warming up the device...")

    if len(sentences_list) != loop_count:
        raise ValueError("Number of sentence batches must be equal to loop count.")

    with torch.no_grad():
        for i in range(loop_count):
            _ = encode(
                compiled_model,
                sentences_list[i],
                device=device,
                input_sequence_length=input_sequence_length,
                **encode_kwargs,
            )

    print("Warming up completed.")


def bge_m3_measure_fps(compiled_model, sentences_list, encode_kwargs, loop_count, device, input_sequence_length):
    if len(sentences_list) != loop_count:
        raise ValueError("Number of sentence batches must be equal to loop count.")

    print("Starting benchmark loop...")

    predictions = []
    iteration_times = []

    with torch.no_grad():
        for i in range(loop_count):
            start_time = time.perf_counter_ns()

            # Run encoding
            output = encode(
                compiled_model,
                sentences_list[i],
                device=device,
                input_sequence_length=input_sequence_length,
                **encode_kwargs,
            )
            predictions.append(output)

            end_time = time.perf_counter_ns()
            iteration_times.append(end_time - start_time)

            print(f"Iteration\t{i+1}/{loop_count}\ttook {iteration_times[-1] / 1e6:.04} ms")

    total_time = sum(iteration_times)

    # Convert to seconds
    total_time /= 1e9
    return predictions, total_time


@pytest.mark.parametrize("batch_size", BATCH_SIZE, ids=[f"batch_size={item}" for item in BATCH_SIZE])
@pytest.mark.parametrize("loop_count", LOOP_COUNT, ids=[f"loop_count={item}" for item in LOOP_COUNT])
@pytest.mark.parametrize("task", TASK, ids=[f"task={item}" for item in TASK])
@pytest.mark.parametrize("data_format", DATA_FORMAT, ids=[f"data_format={item}" for item in DATA_FORMAT])
def test_bge_m3_encode_torch_xla(
    training, batch_size, loop_count, task, data_format, model_name, measure_cpu, input_sequence_length
):
    """
    This function creates a BGE-M3 encode model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if not input_sequence_length:
        raise ValueError("Input sequence length must be a positive integer.")

    if training:
        pytest.skip("Training is not supported")

    # Get input sentences
    sentences = get_benchmark_sentences(batch_size)

    # Prepare inputs for loop_count iterations
    inputs = [sentences] * loop_count
    warmup_inputs = [sentences] * loop_count

    # Load model
    bge_m3 = BGEM3FlagModel("BAAI/bge-m3")
    model = bge_m3.model
    model.eval()

    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)

    # Encode kwargs
    encode_kwargs = {
        "return_dense": True,
        "return_sparse": True,
        "return_colbert_vecs": True,
    }

    # For CPU baseline, run once
    cpu_fps = -1.0
    if measure_cpu:
        cpu_sentences = [sentences[0]]  # Single sentence for CPU measurement

        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            _ = bge_m3.encode(cpu_sentences, return_dense=True, return_sparse=True, return_colbert_vecs=True)
        elapsed = time.time() - start_time
        cpu_fps = num_runs / elapsed

    # Get golden output for PCC verification using bge_m3.encode()
    golden_output = bge_m3.encode(sentences, return_dense=True, return_sparse=True, return_colbert_vecs=True)

    options = {
        "enable_optimizer": OPTIMIZER_ENABLED,
        "enable_memory_layout_analysis": MEMORY_LAYOUT_ANALYSIS_ENABLED,
        "enable_l1_interleaved": False,
        "enable_fusing_conv2d_with_multiply_pattern": True,
    }

    torch_xla.set_custom_compile_options(options)

    # Compile model with TT backend
    compiled_model = torch.compile(model, backend="tt", options={"tt_experimental_compile": True})

    device = xm.xla_device()
    # Send actual device to model encode

    if data_format == "bfloat16":
        compiled_model = compiled_model.to(device, dtype=torch.bfloat16)
    else:
        compiled_model = compiled_model.to(device)

    # Warmup using custom warmup function
    bge_m3_warmup(compiled_model, warmup_inputs, encode_kwargs, loop_count, device, input_sequence_length)

    # Benchmark using custom fps measurement function
    predictions, total_time = bge_m3_measure_fps(
        compiled_model, inputs, encode_kwargs, loop_count, device, input_sequence_length
    )

    # PCC verification with first prediction
    pcc_value = compute_pcc(
        torch.tensor(predictions[0]["dense_vecs"]), torch.tensor(golden_output["dense_vecs"]), required_pcc=0.99
    )
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = f"BGE-M3 Encode Torch-XLA ({input_sequence_length})"
    model_type = "embedding"
    dataset_name = "Random Data"
    num_layers = 24  # BGE-M3 has 24 layers

    input_size = (1024,)

    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title="BGE-M3 Encode Torch-XLA",
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        cpu_samples_per_sec=cpu_fps,
        evaluation_score=pcc_value,
        batch_size=batch_size,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
    )

    result = create_benchmark_result(
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        num_layers=num_layers,
        batch_size=batch_size,
        input_size=input_size,
        loop_count=loop_count,
        data_format=data_format,
        training=training,
        total_time=total_time,
        total_samples=total_samples,
        evaluation_score=pcc_value,
        custom_measurements=custom_measurements,
        optimizer_enabled=OPTIMIZER_ENABLED,
        program_cache_enabled=PROGRAM_CACHE_ENABLED,
        memory_layout_analysis_enabled=MEMORY_LAYOUT_ANALYSIS_ENABLED,
        trace_enabled=TRACE_ENABLED,
        model_info="BAAI/bge-m3",
        torch_xla_enabled=True,
        backend="tt",
        input_is_image=False,
        input_sequence_length=input_sequence_length,
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
    )

    return result


def benchmark(config: dict):
    """
    Run the bge_m3_encode torch-xla benchmark.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    task = config["task"]
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]
    input_sequence_length = config["input_sequence_length"]

    return test_bge_m3_encode_torch_xla(
        training=training,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        model_name=model_name,
        measure_cpu=measure_cpu,
        input_sequence_length=input_sequence_length,
    )
