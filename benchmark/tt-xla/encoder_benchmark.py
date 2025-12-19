# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import socket
import time
from typing import List

# Third-party modules
import torch
import torch_xla
import torch_xla.runtime as xr

from benchmark.utils import get_xla_device_arch
from utils import (
    get_benchmark_metadata,
    print_benchmark_results,
    create_benchmark_result,
    compute_pcc,
)

xr.set_device_type("TT")

MIN_STEPS = 16  # Minimum warmup steps

MODULE_EXPORT_PATH = "modules"


def apply_mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Apply mean pooling over hidden states.

    Args:
        hidden_states: Token embeddings with shape [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask with shape [batch_size, seq_len]

    Returns:
        Sentence embeddings with shape [batch_size, hidden_size]
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sentence_embeddings = torch.sum(hidden_states * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return sentence_embeddings


def apply_last_token_pooling(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor, left_padding: bool
) -> torch.Tensor:
    """Apply last token pooling over hidden states.

    Args:
        hidden_states: Token embeddings with shape [batch_size, seq_len, hidden_size]
        attention_mask: Attention mask with shape [batch_size, seq_len]
        left_padding: Whether left padding was used (all sequences end with non-padding tokens)

    Returns:
        Sentence embeddings with shape [batch_size, hidden_size]
    """
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]


def encode_sentences(
    model,
    tokenized_inputs: dict,
    device,
    read_hidden_state_fn,
    use_mean_pooling: bool,
) -> torch.Tensor:
    """Encode sentences using the specified pooling strategy.

    Args:
        model: Encoder model instance
        tokenized_inputs: Pre-tokenized inputs dict with 'input_ids' and 'attention_mask'
        device: Device to run inference on
        read_hidden_state_fn: Function to extract hidden states from model output
        use_mean_pooling: If True, use mean pooling. If False, use last token pooling.

    Returns:
        torch.Tensor: Sentence embeddings with shape [batch_size, hidden_size]
    """
    # Check left padding before moving to device (needed for last token pooling)
    left_padding = None
    if not use_mean_pooling:
        left_padding = (
            tokenized_inputs["attention_mask"][:, -1].sum() == tokenized_inputs["attention_mask"].shape[0]
        ).item()

    # Move to device
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)

    # Get model outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Extract hidden states using the provided function
    hidden_states = read_hidden_state_fn(outputs)

    # Apply pooling strategy
    if use_mean_pooling:
        return apply_mean_pooling(hidden_states, attention_mask)
    else:
        return apply_last_token_pooling(hidden_states, attention_mask, left_padding)


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


def setup_model(model_loader, data_format, model_variant=None) -> tuple[torch.nn.Module, str, object]:
    """
    Instantiate model and tokenizer.

    Args:
        model_loader: Loader of the model.
        model_variant: Specific variant of the model (optional).
        data_format: Data format (bfloat16 or float32).

    Returns:
        Tuple of (model, model_info_name, tokenizer)

    Raises:
        ValueError: If model_loader does not provide a tokenizer
    """
    # Convert data_format string to torch.dtype
    dtype_override = None
    if data_format == "bfloat16":
        dtype_override = torch.bfloat16
    elif data_format == "float32":
        dtype_override = torch.float32

    if model_variant:
        print(f"Loading model {model_loader.get_model_info(variant=model_variant).name}...")
        model = model_loader.load_model(dtype_override=dtype_override)
        model_info = model_loader.get_model_info(model_variant).name
    else:
        print(f"Loading model {model_loader.get_model_info().name}...")
        model = model_loader.load_model(dtype_override=dtype_override)
        model_info = model_loader.get_model_info().name

    # Get tokenizer (required for encoder benchmarks)
    if not hasattr(model_loader, "tokenizer") or model_loader.tokenizer is None:
        raise ValueError("Model loader must provide a tokenizer for encoder benchmarks")
    tokenizer = model_loader.tokenizer

    model = model.eval()

    return model, model_info, tokenizer


def get_benchmark_sentences(batch_size: int) -> List[str]:
    """
    Get benchmark sentences for encoder models.
    Returns a list of sentences, repeating as needed to match batch_size.

    Args:
        batch_size: Number of sentences to return
    """
    base_sentences = MULTILINGUAL_SENTENCES

    # Extend to match batch size by repeating sentences
    sentences = []
    for i in range(batch_size):
        sentences.append(base_sentences[i % len(base_sentences)])

    return sentences


def construct_inputs(
    tokenizer,
    batch_size: int,
    input_sequence_length: int,
    loop_count: int,
    use_mean_pooling: bool = True,
) -> list:
    """
    Construct and pre-tokenize inputs for the encoder model.

    Args:
        tokenizer: Tokenizer instance
        batch_size: Batch size
        input_sequence_length: Maximum sequence length
        loop_count: Number of loops
        use_mean_pooling: If True, use max_length padding (for mean pooling).
                         If False, use dynamic padding (for last token pooling).

    Returns:
        List of pre-tokenized input dictionaries for each iteration
    """
    inputs = []
    for _ in range(loop_count):
        sentences = get_benchmark_sentences(batch_size)

        # Tokenize based on pooling strategy
        if use_mean_pooling:
            # Mean pooling uses max_length padding
            tokenized = tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=input_sequence_length,
                return_tensors="pt",
            )
        else:
            # Last token pooling uses dynamic padding
            tokenized = tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=input_sequence_length,
                return_tensors="pt",
            )

        inputs.append(tokenized)

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
    experimental_enable_permute_matmul_fusion=False,
    read_hidden_state_fn=None,
    use_mean_pooling=True,
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
        experimental_enable_permute_matmul_fusion: Whether to enable permute matmul fusion optimization
        read_hidden_state_fn: Function to extract hidden states from model output.
            Defaults to default_read_hidden_state_fn which handles most transformer models.
        use_mean_pooling: If True, use mean pooling. If False, use last token pooling.
            This also affects tokenization padding strategy.

    Returns:
        Benchmark result containing performance metrics and model information
    """
    if training:
        raise ValueError("Training is not supported for encoder benchmarks")

    # Load model and tokenizer
    framework_model, model_info, tokenizer = setup_model(model_loader, data_format, model_variant)

    # Construct and pre-tokenize inputs
    tokenized_inputs_list = construct_inputs(
        tokenizer=tokenizer,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        loop_count=loop_count,
        use_mean_pooling=use_mean_pooling,
    )

    warmup_inputs_list = construct_inputs(
        tokenizer=tokenizer,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        loop_count=min(MIN_STEPS, loop_count),
        use_mean_pooling=use_mean_pooling,
    )

    # Measure CPU performance
    if measure_cpu:
        print("Measuring CPU performance...")
        # Use the same full batch for CPU measurement
        cpu_inputs = tokenized_inputs_list[0]

        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = encode_sentences(framework_model, cpu_inputs, "cpu", read_hidden_state_fn, use_mean_pooling)
        elapsed = time.time() - start_time
        cpu_fps = (num_runs * batch_size) / elapsed
        print(f"CPU samples per second: {cpu_fps:.2f}")
    else:
        cpu_fps = -1.0

    # Generate golden output for PCC calculation
    print("Generating golden output on CPU...")
    with torch.no_grad():
        golden_output = encode_sentences(
            framework_model, tokenized_inputs_list[0], "cpu", read_hidden_state_fn, use_mean_pooling
        )

    # Set XLA compilation options
    options = {
        "optimization_level": optimization_level,
        "export_path": MODULE_EXPORT_PATH,
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": ttnn_perf_metrics_output_file,
        "enable_trace": trace_enabled,
        "experimental_enable_weight_bfp8_conversion": enable_weight_bfp8_conversion,
        "experimental_enable_permute_matmul_fusion": experimental_enable_permute_matmul_fusion,
    }

    torch_xla.set_custom_compile_options(options)

    # Compile model
    framework_model.compile(backend="tt", options={"tt_experimental_compile": experimental_compile})

    device = torch_xla.device()

    framework_model = framework_model.to(device)

    # Warmup
    print("Warming up the device...")
    warmup_count = len(warmup_inputs_list)
    with torch.no_grad():
        for i in range(warmup_count):
            output = encode_sentences(
                framework_model, warmup_inputs_list[i], device, read_hidden_state_fn, use_mean_pooling
            )
            _ = output.to("cpu")
    print("Warming up completed.")

    # Benchmark
    print("\nStarting benchmark loop...")
    predictions = []
    iteration_times = []
    outputs = []

    with torch.no_grad():
        for i in range(loop_count):
            start_time = time.perf_counter_ns()
            output = encode_sentences(
                framework_model, tokenized_inputs_list[i], device, read_hidden_state_fn, use_mean_pooling
            )
            outputs.append(output)
            end_time = time.perf_counter_ns()

            iteration_times.append(end_time - start_time)
            print(f"Iteration\t{i+1}/{loop_count}\ttook {iteration_times[-1] / 1e6:.04} ms")

        # Move all outputs to CPU, waits for model execution to finish
        output_start = time.perf_counter_ns()
        for output in outputs:
            cpu_output = output.to("cpu")
            predictions.append(cpu_output)
        output_end = time.perf_counter_ns()

        output_time = output_end - output_start
        print(f"Moving all outputs to CPU took {output_time / 1e6:.04} ms")

    total_time_iterations = sum(iteration_times)
    total_time = total_time_iterations + output_time
    # Convert to seconds
    total_time /= 1e9

    # Evaluate PCC
    pcc_value = compute_pcc(predictions[0], golden_output, required_pcc=required_pcc)
    print(f"PCC verification passed with PCC={pcc_value:.6f}")
    evaluation_score = pcc_value

    total_samples = batch_size * loop_count
    samples_per_sec = total_samples / total_time

    metadata = get_benchmark_metadata()

    full_model_name = model_info
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
