# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import socket
import time
from typing import List

# Third-party modules
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from benchmark.utils import get_xla_device_arch
from utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
)

xr.set_device_type("TT")
os.environ["XLA_STABLEHLO_COMPILE"] = "1"

MIN_STEPS = 16

MODULE_EXPORT_PATH = "modules"


# Default benchmark sentences for different model types
BERT_SENTENCES = [
    "Bu örnek bir cümle",
    "Yapay zeka sistemleri tıp alanında giderek daha fazla kullanılıyor",
    "İklim değişikliği zamanımızın en acil sorunlarından biridir",
    "Kuantum bilgisayarlar karmaşık problemleri çözmeyi vaat ediyor",
    "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur",
    "Makine öğrenimi algoritmaları büyük veri setlerinden öğrenebilir",
    "Doğal dil işleme teknolojileri hızla gelişiyor",
    "Derin öğrenme modelleri görüntü tanıma görevlerinde başarılıdır",
]

QWEN_SENTENCES = [
    "What is the capital of China?",
    "Explain quantum computing in simple terms",
    "How does machine learning work?",
    "What are the benefits of renewable energy?",
    "Describe the process of photosynthesis",
    "What is artificial intelligence?",
    "How do neural networks learn?",
    "What is climate change?",
]

MULTILINGUAL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog while the sun shines brightly.",
    "Machine learning has revolutionized the way we process data.",
    "Climate change represents one of the most pressing challenges of our time.",
    "人工知能システムは医療分野にますます統合されています。",
    "기후 변화는 우리 시대의 가장 시급한 과제입니다.",
    "La inteligencia artificial está transformando muchas industrias.",
    "L'apprentissage automatique change notre façon de comprendre les données.",
    "Die künstliche Intelligenz entwickelt sich rasant weiter.",
]


def setup_model(model_loader, model_variant=None, data_format="float32") -> tuple[torch.nn.Module, str, object]:
    """
    Instantiate model and tokenizer.

    Args:
        model_loader: Loader of the model.
        model_variant: Specific variant of the model (optional).
        data_format: Data format (bfloat16 or float32).

    Returns:
        Tuple of (model, model_info_name, tokenizer)
    """
    if model_variant:
        print(f"Loading model {model_loader.get_model_info(variant=model_variant).name}...")
        model = model_loader.load_model()
        model_info = model_loader.get_model_info(model_variant).name
    else:
        print(f"Loading model {model_loader.get_model_info().name}...")
        model = model_loader.load_model()
        model_info = model_loader.get_model_info().name

    # Get tokenizer
    tokenizer = model_loader.tokenizer if hasattr(model_loader, "tokenizer") else None

    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    elif data_format == "float32":
        model = model.to(torch.float32)

    model = model.eval()

    return model, model_info, tokenizer


def get_benchmark_sentences(batch_size: int, model_info: str) -> List[str]:
    """
    Get benchmark sentences for encoder models.
    Returns a list of sentences, repeating as needed to match batch_size.

    Args:
        batch_size: Number of sentences to return
        model_info: Model info string to determine sentence type
    """
    # Select sentences based on model type
    model_info_lower = model_info.lower()
    if "bert" in model_info_lower and "turkish" in model_info_lower:
        base_sentences = BERT_SENTENCES
    elif "qwen" in model_info_lower:
        base_sentences = QWEN_SENTENCES
    else:
        base_sentences = MULTILINGUAL_SENTENCES

    # Extend to match batch size by repeating sentences
    sentences = []
    for i in range(batch_size):
        sentences.append(base_sentences[i % len(base_sentences)])

    return sentences


def mean_pool_encode(model, tokenizer, sentences: List[str], device, max_length: int) -> torch.Tensor:
    """
    Encode sentences using mean pooling over token embeddings.

    Args:
        model: Encoder model instance
        tokenizer: Tokenizer instance
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
    if hasattr(outputs, "last_hidden_state"):
        token_embeddings = outputs.last_hidden_state
    else:
        token_embeddings = outputs[0]  # Last hidden state

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

    return sentence_embeddings


def last_token_pool_encode(model, tokenizer, sentences: List[str], device, max_length: int) -> torch.Tensor:
    """
    Encode sentences using last token pooling (for models like Qwen3).

    Args:
        model: Encoder model instance
        tokenizer: Tokenizer instance
        sentences: List of sentences to encode
        device: Device to run on
        max_length: Maximum sequence length for tokenization

    Returns:
        torch.Tensor: Sentence embeddings with shape [batch_size, hidden_size]
    """
    # Tokenize the input sentences
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get model outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Get last hidden state
    if hasattr(outputs, "last_hidden_state"):
        last_hidden_states = outputs.last_hidden_state
    else:
        last_hidden_states = outputs[0]

    # Last token pooling
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        embeddings = last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        embeddings = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    return embeddings


def construct_inputs(
    tokenizer,
    batch_size: int,
    input_sequence_length: int,
    loop_count: int,
    model_info: str,
) -> list:
    """
    Construct sentence inputs for the encoder model.

    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        input_sequence_length: Maximum sequence length
        loop_count: Number of loops
        model_info: Model info string to determine sentence type

    Returns:
        List of sentence lists for each iteration
    """
    inputs = []
    for _ in range(loop_count):
        sentences = get_benchmark_sentences(batch_size, model_info)
        inputs.append(sentences)

    return inputs


def benchmark_encoder_torch_xla(
    model_loader,
    model_variant,
    optimization_level,
    trace_enabled,
    training,
    batch_size,
    input_sequence_length,
    loop_count,
    data_format,
    measure_cpu,
    experimental_compile,
    ttnn_perf_metrics_output_file,
    required_pcc=0.97,
    enable_weight_bfp8_conversion=False,
):
    """
    Benchmark an encoder model using PyTorch and torch-xla.

    This function loads an encoder model, compiles it with torch-xla for the Tenstorrent backend,
    and measures its inference performance. It performs warmup runs, collects metrics,
    and validates output correctness via PCC (Pearson Correlation Coefficient).

    Args:
        model_loader: Model loader instance for loading the encoder model
        model_variant: Specific variant/version of the model to benchmark
        optimization_level: tt-mlir optimization level for compilation
        trace_enabled: Whether to enable tracing
        training: Whether to run in training mode (not supported)
        batch_size: Batch size for inference
        input_sequence_length: Maximum sequence length for tokenization
        loop_count: Number of inference iterations to benchmark
        data_format: Data precision format
        measure_cpu: Whether to measure CPU baseline performance
        experimental_compile: Whether to use experimental compilation features
        ttnn_perf_metrics_output_file: Path to save TTNN performance metrics
        required_pcc: Minimum PCC threshold for output validation
        enable_weight_bfp8_conversion: Whether to enable bfp8 weight conversion

    Returns:
        Benchmark result containing performance metrics and model information
    """
    if training:
        raise ValueError("Training is not supported for encoder benchmarks")

    if not input_sequence_length:
        input_sequence_length = 128  # Default sequence length

    xr.set_device_type("TT")

    # Load model and tokenizer
    framework_model, model_info, tokenizer = setup_model(model_loader, model_variant, data_format)

    if tokenizer is None:
        raise ValueError("Model loader must provide a tokenizer for encoder benchmarks")

    # Determine encoding function based on model type
    model_info_lower = model_info.lower()
    if "qwen" in model_info_lower:
        encode_fn = last_token_pool_encode
    else:
        encode_fn = mean_pool_encode

    # Construct inputs
    sentences_list = construct_inputs(
        tokenizer=tokenizer,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        loop_count=loop_count,
        model_info=model_info,
    )

    warmup_sentences_list = construct_inputs(
        tokenizer=tokenizer,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        loop_count=loop_count,
        model_info=model_info,
    )

    # Measure CPU performance
    if measure_cpu:
        print("Measuring CPU performance...")
        cpu_sentences = [sentences_list[0][0]]  # Single sentence for CPU measurement

        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = encode_fn(framework_model, tokenizer, cpu_sentences, device="cpu", max_length=input_sequence_length)
        elapsed = time.time() - start_time
        cpu_fps = num_runs / elapsed
        print(f"CPU samples per second: {cpu_fps:.2f}")
    else:
        cpu_fps = -1.0

    # Generate golden output for PCC calculation
    print("Generating golden output on CPU...")
    with torch.no_grad():
        golden_output = encode_fn(
            framework_model, tokenizer, sentences_list[0], device="cpu", max_length=input_sequence_length
        )

    # Set XLA compilation options
    options = {
        "optimization_level": optimization_level,
        "export_path": MODULE_EXPORT_PATH,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "enable_trace": trace_enabled,
        "experimental_enable_weight_bfp8_conversion": enable_weight_bfp8_conversion,
    }

    torch_xla.set_custom_compile_options(options)

    # Compile model
    print("Compiling model for TT backend...")
    framework_model.compile(backend="tt", options={"tt_experimental_compile": experimental_compile})

    device = torch_xla.device()

    if data_format == "bfloat16":
        framework_model = framework_model.to(device, dtype=torch.bfloat16)
    elif data_format == "float32":
        framework_model = framework_model.to(device, dtype=torch.float32)
    else:
        framework_model = framework_model.to(device)

    # Warmup
    print("Warming up the device...")
    with torch.no_grad():
        for i in range(loop_count):
            _ = encode_fn(
                framework_model, tokenizer, warmup_sentences_list[i], device=device, max_length=input_sequence_length
            )
    print("Warming up completed.")

    # Benchmark
    print("\nStarting benchmark loop...")
    predictions = []
    iteration_times = []

    with torch.no_grad():
        for i in range(loop_count):
            start_time = time.perf_counter_ns()
            output = encode_fn(
                framework_model, tokenizer, sentences_list[i], device=device, max_length=input_sequence_length
            )
            predictions.append(output)
            end_time = time.perf_counter_ns()

            iteration_times.append(end_time - start_time)
            print(f"Iteration\t{i+1}/{loop_count}\ttook {iteration_times[-1] / 1e6:.04} ms")

    total_time = sum(iteration_times) / 1e9  # Convert to seconds

    # Evaluate PCC
    pcc_value = compute_pcc(predictions[0].cpu(), golden_output, required_pcc=required_pcc)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")
    evaluation_score = pcc_value

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = f"{model_info}"
    model_type = "Encoder, Text Embedding"
    dataset_name = "Benchmark Sentences"
    num_layers = -1

    custom_measurements = [
        {
            "measurement_name": "cpu_fps",
            "value": cpu_fps,
            "target": -1,
        }
    ]

    print_benchmark_results(
        model_title=full_model_name,
        full_model_name=full_model_name,
        model_type=model_type,
        dataset_name=dataset_name,
        date=metadata["date"],
        machine_name=metadata["machine_name"],
        total_time=total_time,
        total_samples=total_samples,
        samples_per_sec=samples_per_sec,
        cpu_samples_per_sec=cpu_fps,
        evaluation_score=evaluation_score,
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
        input_size=(input_sequence_length,),
        loop_count=loop_count,
        data_format=data_format,
        training=training,
        total_time=total_time,
        total_samples=total_samples,
        evaluation_score=evaluation_score,
        custom_measurements=custom_measurements,
        optimization_level=optimization_level,
        program_cache_enabled=True,
        trace_enabled=trace_enabled,
        model_info=model_info,
        torch_xla_enabled=True,
        backend="tt",
        device_name=socket.gethostname(),
        arch=get_xla_device_arch(),
        input_is_image=False,
        input_sequence_length=input_sequence_length,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
    )

    return result
