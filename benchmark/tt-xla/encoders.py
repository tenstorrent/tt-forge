# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List, Callable, Optional

import torch

from benchmark.utils import aggregate_ttnn_perf_metrics, sanitize_filename
from encoder_benchmark import benchmark_encoder_torch_xla
from utils import apply_mean_pooling, apply_last_token_pooling, extract_single_layer


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

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


def get_default_inputs(batch_size: int, sentences=MULTILINGUAL_SENTENCES) -> List[str]:
    """Get default benchmark sentences, repeating as needed to match batch_size."""
    return [sentences[i % len(sentences)] for i in range(batch_size)]


# =============================================================================
# Defaults for all encoder models
# =============================================================================

DEFAULT_OPTIMIZATION_LEVEL = 1
DEFAULT_TRACE_ENABLED = False
DEFAULT_BATCH_SIZE = 1
DEFAULT_LOOP_COUNT = 32
DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_MEASURE_CPU = False
DEFAULT_EXPERIMENTAL_COMPILE = True
DEFAULT_REQUIRED_PCC = 0.97
DEFAULT_ENABLE_WEIGHT_BFP8_CONVERSION = False
DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION = False


# =============================================================================
# Main test function
# =============================================================================


def test_encoder(
    model,
    model_nickname: str,
    output_file: str,
    tokenizer,
    output_processor_fn: Callable,
    single_layer: bool = False,
    # Tokenizer options
    padding: str = "max_length",
    padding_side: str = "right",
    # Benchmark options
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled: bool = DEFAULT_TRACE_ENABLED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    loop_count: int = DEFAULT_LOOP_COUNT,
    input_sequence_length: int = DEFAULT_INPUT_SEQUENCE_LENGTH,
    data_format: str = DEFAULT_DATA_FORMAT,
    measure_cpu: bool = DEFAULT_MEASURE_CPU,
    experimental_compile: bool = DEFAULT_EXPERIMENTAL_COMPILE,
    required_pcc: float = DEFAULT_REQUIRED_PCC,
    enable_weight_bfp8_conversion: bool = DEFAULT_ENABLE_WEIGHT_BFP8_CONVERSION,
    experimental_enable_permute_matmul_fusion: bool = DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION,
    # Optional overrides for non-standard models
    load_inputs_fn: Optional[Callable] = None,
    preprocess_fn: Optional[Callable] = None,
):
    """Test encoder model with automatic single_layer handling.

    Args:
        model: Loaded model instance in eval mode
        model_nickname: Short model name for file naming
        output_file: Path to save benchmark results as JSON
        tokenizer: Tokenizer for preprocessing (can be None if preprocess_fn provided)
        output_processor_fn: Function to process outputs -> embeddings
        single_layer: If True, extract and test single layer only
        padding: Tokenizer padding strategy
        padding_side: Tokenizer padding side
        load_inputs_fn: Custom input loader (default: get_default_inputs)
        preprocess_fn: Custom preprocessing (default: tokenize with options above)
    """
    # Handle single_layer mode
    if single_layer:
        print(f"🔧 Single layer mode: extracting layer 0 from {model_nickname}")
        hidden_size = model.config.hidden_size
        model = extract_single_layer(model, layer_idx=0)

        # Single layer uses hidden_states input instead of tokens
        load_inputs_fn = lambda bs: bs
        preprocess_fn = lambda bs, device: {
            "hidden_states": torch.randn(bs, input_sequence_length, hidden_size, dtype=DTYPE_MAP[data_format]).to(device),
            "attention_mask": None,
        }
        output_processor_fn = lambda out, inputs: out
    else:
        # Use defaults if not provided
        if load_inputs_fn is None:
            load_inputs_fn = get_default_inputs

        if preprocess_fn is None and tokenizer is not None:
            tokenizer.padding_side = padding_side
            preprocess_fn = lambda sentences, device: {
                k: v.to(device)
                for k, v in tokenizer(
                    sentences,
                    padding=padding,
                    truncation=True,
                    max_length=input_sequence_length,
                    return_tensors="pt",
                ).items()
            }

    # Run benchmark
    sanitized_name = sanitize_filename(model_nickname)
    ttnn_perf_metrics_file = f"tt_xla_{sanitized_name}_perf_metrics"

    print(f"Running encoder benchmark for: {model_nickname}")
    print(f"  single_layer={single_layer}, batch_size={batch_size}, seq_len={input_sequence_length}")

    results = benchmark_encoder_torch_xla(
        model=model,
        model_nickname=model_nickname,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        training=False,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        loop_count=loop_count,
        data_format=data_format,
        measure_cpu=measure_cpu,
        experimental_compile=experimental_compile,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        required_pcc=required_pcc,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
        single_layer=single_layer,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_nickname
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_file, results)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)


# =============================================================================
# BERT Tests
# =============================================================================


def test_bert(output_file, single_layer, attn_implementation="sdpa"):
    """Test BERT encoder model."""
    from third_party.tt_forge_models.bert.sentence_embedding_generation.pytorch.loader import ModelLoader

    loader = ModelLoader()
    print(f"\nLoading BERT with attention={attn_implementation}...")
    model = loader.load_model(dtype_override=DTYPE_MAP["bfloat16"], attn_implementation=attn_implementation)

    model_name = "bert" if attn_implementation == "sdpa" else f"bert_{attn_implementation}"

    test_encoder(
        model=model,
        model_nickname=model_name,
        output_file=output_file,
        tokenizer=loader.tokenizer,
        output_processor_fn=lambda out, inputs: apply_mean_pooling(out.last_hidden_state, inputs["attention_mask"]),
        single_layer=single_layer,
        batch_size=8,
        input_sequence_length=384,
        optimization_level=2,
    )


# =============================================================================
# Qwen3 Embedding Tests (decoder-based embeddings)
# =============================================================================


def test_qwen3_embedding_4b(output_file, single_layer):
    """Test Qwen3 Embedding 4B model."""
    from third_party.tt_forge_models.qwen_3.embedding.pytorch.loader import ModelLoader, ModelVariant

    loader = ModelLoader(variant=ModelVariant.QWEN_3_EMBEDDING_4B)
    print(f"\nLoading Qwen3 Embedding 4B...")
    model = loader.load_model(dtype_override=DTYPE_MAP["bfloat16"])

    test_encoder(
        model=model,
        model_nickname="qwen3_emb_4b",
        output_file=output_file,
        tokenizer=loader.tokenizer,
        output_processor_fn=lambda out, inputs: apply_last_token_pooling(out.last_hidden_state, inputs["attention_mask"]),
        single_layer=single_layer,
        padding="longest",
        batch_size=32,
        input_sequence_length=128,
        optimization_level=0,
    )


# [pytest.skip] Too large for single chip
def test_qwen3_embedding_8b(output_file, single_layer):
    """Test Qwen3 Embedding 8B model."""
    from third_party.tt_forge_models.qwen_3.embedding.pytorch.loader import ModelLoader, ModelVariant

    loader = ModelLoader(variant=ModelVariant.QWEN_3_EMBEDDING_8B)
    print(f"\nLoading Qwen3 Embedding 8B...")
    model = loader.load_model(dtype_override=DTYPE_MAP["bfloat16"])

    test_encoder(
        model=model,
        model_nickname="qwen3_emb_8b",
        output_file=output_file,
        tokenizer=loader.tokenizer,
        output_processor_fn=lambda out, inputs: apply_last_token_pooling(out.last_hidden_state, inputs["attention_mask"]),
        single_layer=single_layer,
        padding_side="left",
        batch_size=1,
        input_sequence_length=128,
    )


# =============================================================================
# BGE-M3 Test (complex postprocessing)
# =============================================================================


def test_bge_m3(output_file, single_layer):
    """Test BGE-M3 encoder model with custom postprocessing."""
    import numpy as np
    from collections import defaultdict
    from third_party.tt_forge_models.bge_m3.encode.pytorch.loader import ModelLoader
    from FlagEmbedding import BGEM3FlagModel

    input_sequence_length = 512

    loader = ModelLoader()
    print(f"\nLoading BGE-M3...")
    model = BGEM3FlagModel("BAAI/bge-m3").model
    model = model.eval()
    tokenizer = model.tokenizer

    def bge_preprocess(sentences, device):
        tokenized = tokenizer(sentences, padding="max_length", truncation=True,
                              max_length=input_sequence_length, return_tensors="pt")
        text_input = {k: v.to(device) for k, v in tokenized.items()}
        return {"text_input": text_input, "return_dense": True, "return_sparse": True,
                "return_colbert_vecs": True, "return_sparse_embedding": False}

    def bge_output_processor(outputs, model_inputs):
        text_input = model_inputs["text_input"]
        input_ids = text_input["input_ids"]
        batch_size = input_ids.shape[0]
        length_sorted_idx = np.argsort([-len(input_ids[i]) for i in range(batch_size)])
        dense_vecs = outputs["dense_vecs"].cpu().detach().numpy()
        dense_vecs = np.concatenate([dense_vecs], axis=0)[np.argsort(length_sorted_idx)]
        return torch.tensor(dense_vecs)

    test_encoder(
        model=model,
        model_nickname="bge_m3",
        output_file=output_file,
        tokenizer=None,  # Using custom preprocess
        output_processor_fn=bge_output_processor,
        single_layer=single_layer,
        preprocess_fn=bge_preprocess,
        data_format="float32",
        batch_size=4,
        input_sequence_length=input_sequence_length,
        optimization_level=0,
    )


# =============================================================================
# UNet Test (diffusion model)
# =============================================================================


def test_unet_for_conditional_generation(output_file, single_layer):
    """Test UNet for Stable Diffusion XL."""
    from third_party.tt_forge_models.unet_for_conditional_generation.pytorch.loader import ModelLoader

    def move_to_device(inputs, device):
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device)
            elif isinstance(v, dict):
                result[k] = move_to_device(v, device)
            else:
                result[k] = v
        return result

    loader = ModelLoader()
    print(f"\nLoading UNet SDXL...")
    model = loader.load_model(dtype_override=DTYPE_MAP["bfloat16"])

    test_encoder(
        model=model,
        model_nickname="unet_sdxl",
        output_file=output_file,
        tokenizer=None,
        output_processor_fn=lambda out, inputs: out.sample,
        single_layer=single_layer,
        load_inputs_fn=lambda bs: loader.load_inputs(batch_size=bs, dtype_override=DTYPE_MAP["bfloat16"]),
        preprocess_fn=lambda inputs, device: move_to_device(inputs, device),
        batch_size=1,
        input_sequence_length=77,
        loop_count=128,
        optimization_level=1,
    )
