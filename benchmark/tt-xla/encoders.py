# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import torch
import torch.nn as nn

from benchmark.utils import aggregate_ttnn_perf_metrics, sanitize_filename
from encoder_benchmark import benchmark_encoder_torch_xla
from utils import apply_mean_pooling, apply_last_token_pooling


def apply_identity_pooling(outputs, attention_mask):
    """No-op pooling for models that return pre-pooled embeddings."""
    return outputs.last_hidden_state.squeeze(1)


# Defaults for all encoder models
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
DEFAULT_PADDING_SIDE = "right"
DEFAULT_PADDING = "max_length"


def test_encoder(
    ModelLoaderModule,
    variant,
    output_file,
    output_processor_fn,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
    data_format=DEFAULT_DATA_FORMAT,
    measure_cpu=DEFAULT_MEASURE_CPU,
    experimental_compile=DEFAULT_EXPERIMENTAL_COMPILE,
    required_pcc=DEFAULT_REQUIRED_PCC,
    enable_weight_bfp8_conversion=DEFAULT_ENABLE_WEIGHT_BFP8_CONVERSION,
    experimental_enable_permute_matmul_fusion=DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION,
    padding_side=DEFAULT_PADDING_SIDE,
    padding=DEFAULT_PADDING,
):
    """Test encoder model with the given variant and optional configuration overrides.

    Args:
        ModelLoaderModule: Model loader class
        variant: Model variant identifier (can be None for models without variants)
        output_file: Path to save benchmark results as JSON
        output_processor_fn: Function to process model outputs into embeddings.
            Signature: fn(outputs, attention_mask) -> embeddings
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_sequence_length: Input sequence length
        data_format: Data format
        measure_cpu: Measure CPU FPS
        experimental_compile: Enable experimental compile
        required_pcc: Required PCC threshold
        enable_weight_bfp8_conversion: Enable BFP8 weight conversion
        experimental_enable_permute_matmul_fusion: Enable permute matmul fusion
        padding_side: Tokenizer padding side ("right" or "left")
        padding: Padding strategy ("max_length" or True for dynamic)
    """
    model_loader = ModelLoaderModule(variant=variant) if variant else ModelLoaderModule()
    model_info_name = model_loader.get_model_info(variant=variant).name

    # Sanitize model name for safe filesystem usage
    sanitized_model_name = sanitize_filename(model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{sanitized_model_name}_perf_metrics"

    print(f"Running encoder benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    batch_size={batch_size}
    loop_count={loop_count}
    input_sequence_length={input_sequence_length}
    data_format={data_format}
    measure_cpu={measure_cpu}
    experimental_compile={experimental_compile}
    required_pcc={required_pcc}
    enable_weight_bfp8_conversion={enable_weight_bfp8_conversion}
    experimental_enable_permute_matmul_fusion={experimental_enable_permute_matmul_fusion}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_encoder_torch_xla(
        model_loader=model_loader,
        model_variant=variant,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        training=False,
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        loop_count=loop_count,
        data_format=data_format,
        measure_cpu=measure_cpu,
        experimental_compile=experimental_compile,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        output_processor_fn=output_processor_fn,
        required_pcc=required_pcc,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
        padding_side=padding_side,
        padding=padding,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_bert(output_file):
    from third_party.tt_forge_models.bert.sentence_embedding_generation.pytorch.loader import ModelLoader

    test_encoder(
        ModelLoaderModule=ModelLoader,
        variant=None,
        output_file=output_file,
        output_processor_fn=lambda out, mask: apply_mean_pooling(out.last_hidden_state, mask),
        batch_size=8,
        input_sequence_length=384,
        loop_count=32,
        optimization_level=2,
    )


def test_qwen3_embedding_4b(output_file):
    from third_party.tt_forge_models.qwen_3.embedding.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_EMBEDDING_4B
    test_encoder(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        output_processor_fn=lambda out, mask: apply_last_token_pooling(out.last_hidden_state, mask),
        batch_size=32,
        input_sequence_length=128,
        loop_count=32,
        padding_side="left",
        padding="max_length",
        optimization_level=0,
    )


# [pytest.skip] Too large for single chip
def test_qwen3_embedding_8b(output_file):
    from third_party.tt_forge_models.qwen_3.embedding.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.QWEN_3_EMBEDDING_8B
    test_encoder(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        output_processor_fn=lambda out, mask: apply_last_token_pooling(out.last_hidden_state, mask),
        batch_size=1,
        input_sequence_length=128,
        loop_count=32,
        padding_side="left",
        padding="max_length",
    )


# BGE-M3 Wrapper Classes
# These adapt the BGE-M3 model interface to the standard encoder benchmark interface
class BGEM3Output:
    """Mimics standard encoder output structure for BGE-M3."""

    def __init__(self, dense_vecs):
        # dense_vecs is already [batch_size, hidden_size] - no pooling needed
        # Add seq_len=1 dim for compatibility with pooling functions
        self.last_hidden_state = dense_vecs.unsqueeze(1)


class BGEM3EncoderWrapper(nn.Module):
    """Wraps BGE-M3 model to match standard encoder interface."""

    def __init__(self, bge_model):
        super().__init__()
        self.model = bge_model

    def forward(self, input_ids, attention_mask):
        text_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        outputs = self.model(
            text_input=text_input,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return BGEM3Output(dense_vecs=outputs["dense_vecs"])


class BGEM3EncoderLoader:
    """Wrapper loader that adapts BGE-M3 to encoder benchmark interface."""

    def __init__(self, variant=None):
        from third_party.tt_forge_models.bge_m3.pytorch.loader import ModelLoader, ModelVariant

        self._inner_loader = ModelLoader(variant=variant or ModelVariant.BASE)
        self.tokenizer = None

    def get_model_info(self, variant=None):
        return self._inner_loader.get_model_info(variant=variant)

    def load_model(self, dtype_override=None):
        # Load the underlying BGE-M3 model (XLM-RoBERTa based)
        bge_model = self._inner_loader.load_model(dtype_override=dtype_override)
        # Get tokenizer from the loaded model
        self.tokenizer = self._inner_loader.model.tokenizer

        # Wrap it to adapt interface
        wrapper = BGEM3EncoderWrapper(bge_model)
        return wrapper


def test_bge_m3_encode(output_file):
    test_encoder(
        ModelLoaderModule=BGEM3EncoderLoader,
        variant=None,
        output_file=output_file,
        output_processor_fn=apply_identity_pooling,
        batch_size=4,
        input_sequence_length=512,
        loop_count=32,
        optimization_level=0,
        enable_weight_bfp8_conversion=True,
        data_format="float32",
    )
