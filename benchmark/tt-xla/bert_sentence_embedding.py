# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import time
import pytest
import socket
from typing import List

# Third-party modules
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm

from benchmark.utils import get_xla_device_arch

from .utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
)

# Import from tt_forge_models
from third_party.tt_forge_models.bert.sentence_embedding_generation import ModelLoader

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

# Common constants
OPTIMIZATION_LEVEL = 2
PROGRAM_CACHE_ENABLED = True
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

MODULE_EXPORT_PATH = "modules"


def get_benchmark_sentences(batch_size: int) -> List[str]:
    """
    Get benchmark sentences for Turkish BERT sentence embedding.
    Returns a list of Turkish sentences, repeating as needed to match batch_size.
    """
    # Turkish sentences for testing
    base_sentences = [
        "Bu örnek bir cümle",
        "Yapay zeka sistemleri tıp alanında giderek daha fazla kullanılıyor",
        "İklim değişikliği zamanımızın en acil sorunlarından biridir",
        "Kuantum bilgisayarlar karmaşık problemleri çözmeyi vaat ediyor",
        "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur",
        "Makine öğrenimi algoritmaları büyük veri setlerinden öğrenebilir",
        "Doğal dil işleme teknolojileri hızla gelişiyor",
        "Derin öğrenme modelleri görüntü tanıma görevlerinde başarılıdır",
        "Blok zinciri teknolojisi merkezi olmayan sistemler oluşturur",
        "Nöral ağlar insan beyninden ilham alır",
    ]

    # Extend to match batch size by repeating sentences
    sentences = []
    for i in range(batch_size):
        sentences.append(base_sentences[i % len(base_sentences)])

    return sentences


def encode_sentences(model, tokenizer, sentences: List[str], device, max_length: int) -> torch.Tensor:
    """
    Encode sentences using BERT model with mean pooling.

    Args:
        model: BERT model instance
        tokenizer: BERT tokenizer instance
        sentences: List of sentences to encode
        device: Device to run on
        max_length: Maximum sequence length for tokenization

    Returns:
        torch.Tensor: Sentence embeddings with shape [batch_size, hidden_size]
    """
    # Tokenize the input sentences
    inputs = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get model outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Mean pooling: mask out padding tokens and compute mean
    token_embeddings = outputs[0]  # Last hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

    return sentence_embeddings


def bert_warmup(compiled_model, tokenizer, sentences_list, loop_count, device, max_length):
    """
    Warmup the BERT model for a given number of loop_count.

    Parameters:
    ----------
    compiled_model: torch.nn.Module
        The compiled model to warmup.
    tokenizer: transformers.BertTokenizer
        The tokenizer instance.
    sentences_list: list of list of str
        List of sentence batches to process.
    loop_count: int
        The number of iterations to warmup the model.
    device: torch.device
        The device to run on.
    max_length: int
        Maximum sequence length for tokenization.
    """
    print("Warming up the device...")

    if len(sentences_list) != loop_count:
        raise ValueError("Number of sentence batches must be equal to loop count.")

    with torch.no_grad():
        for i in range(loop_count):
            _ = encode_sentences(
                compiled_model,
                tokenizer,
                sentences_list[i],
                device=device,
                max_length=max_length,
            )

    print("Warming up completed.")


def bert_measure_fps(compiled_model, tokenizer, sentences_list, loop_count, device, max_length):
    """
    Measure FPS for BERT sentence embedding generation.

    Returns:
        predictions: List of sentence embeddings for each batch
        total_time: Total time taken for all iterations (in seconds)
    """
    if len(sentences_list) != loop_count:
        raise ValueError("Number of sentence batches must be equal to loop count.")

    print("Starting benchmark loop...")

    predictions = []
    iteration_times = []

    with torch.no_grad():
        for i in range(loop_count):
            start_time = time.perf_counter_ns()

            # Run encoding
            output = encode_sentences(
                compiled_model,
                tokenizer,
                sentences_list[i],
                device=device,
                max_length=max_length,
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
def test_bert_sentence_embedding_torch_xla(
    training, batch_size, loop_count, task, data_format, model_name, measure_cpu, input_sequence_length
):
    """
    This function creates a BERT sentence embedding model using PyTorch and torch-xla.
    It is used for benchmarking purposes.
    """

    if not input_sequence_length:
        input_sequence_length = 16  # Default from loader.py

    if training:
        pytest.skip("Training is not supported")

    # Get input sentences
    sentences = get_benchmark_sentences(batch_size)

    # Prepare inputs for loop_count iterations
    inputs = [sentences] * loop_count
    warmup_inputs = [sentences] * loop_count

    # Load model from tt_forge_models
    loader = ModelLoader()
    model = loader.load_model()
    tokenizer = loader.tokenizer
    max_length = input_sequence_length

    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)

    # For CPU baseline, run once
    cpu_fps = -1.0
    if measure_cpu:
        cpu_sentences = [sentences[0]]  # Single sentence for CPU measurement

        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = encode_sentences(model, tokenizer, cpu_sentences, device="cpu", max_length=max_length)
        elapsed = time.time() - start_time
        cpu_fps = num_runs / elapsed

    # Get golden output for PCC verification
    with torch.no_grad():
        golden_output = encode_sentences(model, tokenizer, sentences, device="cpu", max_length=max_length)

    options = {
        "optimization_level": OPTIMIZATION_LEVEL,
        "export_path": MODULE_EXPORT_PATH,
    }

    torch_xla.set_custom_compile_options(options)

    # Compile model with TT backend
    compiled_model = torch.compile(model, backend="tt", options={"tt_experimental_compile": True})

    device = xm.xla_device()

    if data_format == "bfloat16":
        compiled_model = compiled_model.to(device, dtype=torch.bfloat16)
    else:
        compiled_model = compiled_model.to(device)

    # Warmup using custom warmup function
    bert_warmup(compiled_model, tokenizer, warmup_inputs, loop_count, device, max_length)

    # Benchmark using custom fps measurement function
    predictions, total_time = bert_measure_fps(compiled_model, tokenizer, inputs, loop_count, device, max_length)

    # PCC verification with first prediction
    pcc_value = compute_pcc(predictions[0].cpu(), golden_output, required_pcc=0.99)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = f"BERT Sentence Embedding Torch-XLA ({input_sequence_length})"
    model_type = "embedding"
    dataset_name = "Turkish Sentences"
    num_layers = 12  # BERT-base has 12 layers

    input_size = (input_sequence_length,)

    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title="BERT Sentence Embedding Torch-XLA",
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
        optimization_level=OPTIMIZATION_LEVEL,
        program_cache_enabled=PROGRAM_CACHE_ENABLED,
        trace_enabled=TRACE_ENABLED,
        model_info="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
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
    Run the bert_sentence_embedding torch-xla benchmark.
    """

    training = config["training"]
    batch_size = config["batch_size"]
    loop_count = config["loop_count"]
    data_format = config["data_format"]
    task = config["task"]
    model_name = config["model"]
    measure_cpu = config["measure_cpu"]
    input_sequence_length = config.get("input_sequence_length", 16)

    return test_bert_sentence_embedding_torch_xla(
        training=training,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        model_name=model_name,
        measure_cpu=measure_cpu,
        input_sequence_length=input_sequence_length,
    )
