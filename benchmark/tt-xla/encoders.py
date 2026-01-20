# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List, Type

import torch
from loguru import logger

from benchmark.utils import aggregate_ttnn_perf_metrics, sanitize_filename
from encoder_benchmark import benchmark_encoder_torch_xla
from utils import apply_mean_pooling, apply_last_token_pooling, supports_num_layers
from model_wrappers import extract_encoder_block, make_encoder_single_layer


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
    """
    Get default benchmark sentences for encoder models.
    Returns a list of sentences, repeating as needed to match batch_size.

    Args:
        batch_size: Number of sentences to return
    """
    inputs = []
    for i in range(batch_size):
        inputs.append(sentences[i % len(sentences)])
    return inputs


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


# =============================================================================
# Single Block Loader (for testing encoder blocks without embeddings/pooler)
# =============================================================================


def _create_single_block_loader(BaseLoaderClass: Type, block_idx: int = 0) -> Type:
    """Create a model loader that returns only the encoder block (no embeddings/pooler).

    Uses extract_encoder_block from model_wrappers which handles the block extraction properly.
    Input: hidden_states [batch, seq_len, hidden_size]
    Output: hidden_states [batch, seq_len, hidden_size]

    If num_layers is supported, loads with 1 layer for efficiency.
    Otherwise, loads full model and extracts the first block.
    """
    use_num_layers = supports_num_layers(BaseLoaderClass)

    class BlockOnlyModelLoader(BaseLoaderClass):
        def __init__(self, variant=None):
            if variant is not None:
                if use_num_layers:
                    super().__init__(variant=variant, num_layers=1)
                else:
                    print(f"Note: {BaseLoaderClass.__name__} doesn't support num_layers - loading full model")
                    super().__init__(variant=variant)
            else:
                if use_num_layers:
                    super().__init__(num_layers=1)
                else:
                    print(f"Note: {BaseLoaderClass.__name__} doesn't support num_layers - loading full model")
                    super().__init__()

        def load_model(self, dtype_override=None):
            full_model = super().load_model(dtype_override=dtype_override)
            wrapper = extract_encoder_block(full_model, block_idx)
            logger.info(f"🔧 Single block mode: returning wrapped block {block_idx}")
            return wrapper

        def load_tokenizer(self):
            return None

    return BlockOnlyModelLoader


def _create_single_layer_loader(BaseLoaderClass: Type, layer_idx: int = 0) -> Type:
    """Create a model loader that returns an encoder model with only one layer.

    Loads the full model then modifies it in-place to keep only one layer.
    Simulates num_layers=1 for loaders that don't support it.

    Input: tokenized input (input_ids, attention_mask, etc.)
    Output: model outputs (last_hidden_state, etc.)
    """

    class SingleLayerModelLoader(BaseLoaderClass):
        def __init__(self, variant=None):
            if variant is not None:
                super().__init__(variant=variant)
            else:
                super().__init__()

        def load_model(self, dtype_override=None):
            model = super().load_model(dtype_override=dtype_override)
            model = make_encoder_single_layer(model, layer_idx)
            logger.info(f"🔧 Single layer mode: modified model to use only layer {layer_idx}")
            return model

    return SingleLayerModelLoader


def create_encoder_loader(
    LoaderClass: Type,
    variant=None,
    single_block: bool = False,
    single_layer: bool = False,
):
    """Create the appropriate loader based on single_block/single_layer flags.

    Args:
        LoaderClass: The model loader class
        variant: Model variant (optional)
        single_block: If True, create block-only loader
        single_layer: If True, create single-layer loader (tries num_layers=1 first)

    Returns:
        Instantiated loader
    """
    if single_block:
        ModifiedLoaderClass = _create_single_block_loader(LoaderClass)
        return ModifiedLoaderClass(variant=variant) if variant else ModifiedLoaderClass()

    if single_layer:
        # Try num_layers=1 first if supported
        if supports_num_layers(LoaderClass):
            if variant:
                return LoaderClass(variant=variant, num_layers=1)
            return LoaderClass(num_layers=1)
        # Fallback: load full model and wrap to use only 1 layer
        print(f"Note: {LoaderClass.__name__} doesn't support num_layers - will wrap to 1 layer")
        ModifiedLoaderClass = _create_single_layer_loader(LoaderClass)
        return ModifiedLoaderClass(variant=variant) if variant else ModifiedLoaderClass()

    # Normal loader
    return LoaderClass(variant=variant) if variant else LoaderClass()


def test_encoder(
    model,
    model_info_name,
    output_file,
    load_inputs_fn,
    output_processor_fn,
    preprocess_fn,
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
    single_block=False,
    single_layer=False,
    model_nickname=None,
    dump_source=False,
):
    """Test encoder model with the given variant and optional configuration overrides.

    Args:
        model: Loaded model instance in eval mode
        model_info_name: Model information for identification and reporting
        output_file: Path to save benchmark results as JSON
        output_processor_fn: Function to process model outputs into embeddings.
            Signature: fn(outputs, model_inputs) -> embeddings
        preprocess_fn: Function to preprocess inputs (tokenization + device placement).
            Signature: fn(sentences, device) -> dict with model input kwargs
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_sequence_length: Length of input sentence
        data_format: Data format
        measure_cpu: Measure CPU FPS
        experimental_compile: Enable experimental compile
        required_pcc: Required PCC threshold
        enable_weight_bfp8_conversion: Enable BFP8 weight conversion
        experimental_enable_permute_matmul_fusion: Enable permute matmul fusion
        single_block: If True, compile and export encoder block only (no embeddings)
        single_layer: If True, compile and export single layer model (full model with 1 layer)
        model_nickname: Optional nickname for the model (used in export filenames)
        dump_source: If True, dump model source code to Python file
        load_inputs_fn: Optional function to load raw inputs.
            Signature: fn(batch_size) -> List[str]. Defaults to get_default_inputs.
    """
    # Sanitize model name for safe filesystem usage
    sanitized_model_name = sanitize_filename(model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{sanitized_model_name}_perf_metrics"

    print(f"Running encoder benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    single_block={single_block}
    single_layer={single_layer}
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
        model=model,
        model_info_name=model_info_name,
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
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        required_pcc=required_pcc,
        enable_weight_bfp8_conversion=enable_weight_bfp8_conversion,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
        single_block=single_block,
        single_layer=single_layer,
        model_nickname=model_nickname,
        dump_source=dump_source,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_bert(output_file, single_block, single_layer, dump_source):
    """Test BERT encoder model. Use --generate-block-test for single encoder block, or --generate-layer-test for single layer."""
    from third_party.tt_forge_models.bert.sentence_embedding_generation.pytorch.loader import ModelLoader

    # Configuration
    data_format = "bfloat16"
    input_sequence_length = 384

    loader = create_encoder_loader(ModelLoader, single_block=single_block, single_layer=single_layer)

    model_info_name = loader.get_model_info().name
    print(f"\nLoading model {model_info_name}...")
    model = loader.load_model(dtype_override=DTYPE_MAP[data_format])

    # Create function for loading raw inputs
    load_inputs_fn = get_default_inputs

    # Create input preprocessing function
    tokenizer = loader.tokenizer
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        preprocess_fn = lambda sentences, device: {
            k: v.to(device)
            for k, v in tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=input_sequence_length,
                return_tensors="pt",
            ).items()
        }
    else:
        # For single_block mode, tokenizer is None
        preprocess_fn = None

    # Create output processing function
    output_processor_fn = lambda out, inputs: apply_mean_pooling(out.last_hidden_state, inputs["attention_mask"])

    test_encoder(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        data_format=data_format,
        batch_size=8,
        input_sequence_length=input_sequence_length,
        loop_count=32,
        optimization_level=2,
        single_block=single_block,
        single_layer=single_layer,
        dump_source=dump_source,
        model_nickname="bert",
    )


def test_qwen_3_embedding_4b(output_file, single_block, single_layer, dump_source):
    """Test Qwen3 Embedding 4B model. Use --generate-layer-test for single layer."""
    from third_party.tt_forge_models.qwen_3.embedding.pytorch.loader import ModelLoader, ModelVariant

    # Block extraction doesn't work for Qwen3 embedding (rotary_emb returns None when extracted)
    if single_block:
        pytest.skip("Qwen3 embedding block extraction not supported - use --generate-layer-test instead")

    # Configuration
    data_format = "bfloat16"
    input_sequence_length = 128
    variant = ModelVariant.QWEN_3_EMBEDDING_4B

    loader = create_encoder_loader(ModelLoader, variant=variant, single_block=single_block, single_layer=single_layer)

    model_info_name = loader.get_model_info(variant=variant).name
    print(f"\nLoading model {model_info_name}...")
    model = loader.load_model(dtype_override=DTYPE_MAP[data_format])

    # Create function for loading raw inputs
    load_inputs_fn = get_default_inputs

    # Create input preprocessing function
    tokenizer = loader.tokenizer
    if tokenizer is not None:
        preprocess_fn = lambda sentences, device: {
            k: v.to(device)
            for k, v in tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=input_sequence_length,
                return_tensors="pt",
            ).items()
        }
    else:
        preprocess_fn = None

    # Create output processing function
    output_processor_fn = lambda out, inputs: apply_last_token_pooling(out.last_hidden_state, inputs["attention_mask"])

    test_encoder(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        data_format=data_format,
        batch_size=32,
        input_sequence_length=input_sequence_length,
        loop_count=32,
        optimization_level=0,
        single_block=single_block,
        single_layer=single_layer,
        dump_source=dump_source,
        model_nickname="qwen_3_embedding_4b",
    )


# [pytest.skip] Too large for single chip
def test_qwen_3_embedding_8b(output_file, single_block, single_layer, dump_source):
    """Test Qwen3 Embedding 8B model. Use --generate-block-test for single encoder block, or --generate-layer-test for single layer."""
    from third_party.tt_forge_models.qwen_3.embedding.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    input_sequence_length = 128
    variant = ModelVariant.QWEN_3_EMBEDDING_8B

    loader = create_encoder_loader(ModelLoader, variant=variant, single_block=single_block, single_layer=single_layer)

    model_info_name = loader.get_model_info(variant=variant).name
    print(f"\nLoading model {model_info_name}...")
    model = loader.load_model(dtype_override=DTYPE_MAP[data_format])

    # Create function for loading raw inputs
    load_inputs_fn = get_default_inputs

    # Create input preprocessing function
    tokenizer = loader.tokenizer
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        preprocess_fn = lambda sentences, device: {
            k: v.to(device)
            for k, v in tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=input_sequence_length,
                return_tensors="pt",
            ).items()
        }
    else:
        preprocess_fn = None

    # Create output processing function
    output_processor_fn = lambda out, inputs: apply_last_token_pooling(out.last_hidden_state, inputs["attention_mask"])

    test_encoder(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        output_processor_fn=output_processor_fn,
        preprocess_fn=preprocess_fn,
        data_format=data_format,
        batch_size=1,
        input_sequence_length=input_sequence_length,
        loop_count=32,
        single_block=single_block,
        single_layer=single_layer,
        dump_source=dump_source,
        model_nickname="qwen_3_embedding_8b",
    )


def test_bge_m3(output_file):
    """Test BGE-M3 encoder model with custom postprocessing.

    BGE-M3 has a unique architecture that produces dense, sparse, and colbert embeddings.
    This test includes all the necessary postprocessing but returns only dense_vecs for PCC calculation.
    Note: Single block/layer tests not supported - BGE-M3 loader uses FlagEmbedding which doesn't expose the model directly.
    """
    import torch
    import numpy as np
    from collections import defaultdict
    from third_party.tt_forge_models.bge_m3.encode.pytorch.loader import ModelLoader
    from FlagEmbedding import BGEM3FlagModel

    # Configuration
    data_format = "float32"
    input_sequence_length = 512

    loader = ModelLoader()
    model_info_name = loader.get_model_info().name
    print(f"\nLoading model {model_info_name}...")

    # BGE-M3 uses a special model loading path
    model = BGEM3FlagModel("BAAI/bge-m3").model
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    # Create function for loading raw inputs
    load_inputs_fn = get_default_inputs

    # Create bge-m3 preprocessing function
    tokenizer = getattr(model, "tokenizer", None)

    if tokenizer is not None:

        def bge_m3_preprocess(sentences, device):
            """Tokenize sentences for BGE-M3 and prepare model inputs."""
            tokenized = tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=input_sequence_length,
                return_tensors="pt",
            )
            # Move to device, convert to dtype, and wrap in text_input dict as expected by BGE-M3
            text_input = {k: v.to(device) for k, v in tokenized.items()}
            return {
                "text_input": text_input,
                "return_dense": True,
                "return_sparse": True,
                "return_colbert_vecs": True,
                "return_sparse_embedding": False,
            }

    else:
        bge_m3_preprocess = None

    # Create bge-m3 output processing function
    def bge_m3_output_processor(outputs, model_inputs):
        """Process BGE-M3 outputs with full postprocessing.

        This includes all postprocessing from bge_m3_encode.py:
        - Length-based sorting and unsorting
        - Processing dense vectors (normalization already done in model)
        - Processing sparse vectors (token weights)
        - Processing colbert vectors

        Returns only dense_vecs for PCC calculation.
        """
        # Extract input_ids and attention_mask from model_inputs
        text_input = model_inputs["text_input"]
        input_ids = text_input["input_ids"]
        attention_mask = text_input["attention_mask"]

        def _process_token_weights(token_weights: np.ndarray, input_ids_item: list):
            """Process token weights for sparse embeddings."""
            result = defaultdict(int)
            unused_tokens = set()
            for _token in ["cls_token", "eos_token", "pad_token", "unk_token"]:
                if _token in tokenizer.special_tokens_map:
                    _token_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map[_token])
                    unused_tokens.add(_token_id)
            for w, idx in zip(token_weights, input_ids_item):
                if idx not in unused_tokens and w > 0:
                    idx = str(idx)
                    if w > result[idx]:
                        result[idx] = w
            return result

        def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask_item: list):
            """Process colbert vectors."""
            tokens_num = np.sum(attention_mask_item)
            return colbert_vecs[: tokens_num - 1]

        # Initialize output containers (same as bge_m3_encode.py)
        all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []

        # Get batch size and create length-based sorting indices (same as bge_m3_encode.py)
        batch_size = input_ids.shape[0]
        length_sorted_idx = np.argsort([-len(input_ids[i]) for i in range(batch_size)])

        # Move all model outputs to CPU first (single sync point for the model graph)
        # Then do ALL post-processing on CPU to avoid extra XLA graphs
        dense_vecs_cpu = outputs["dense_vecs"].cpu().detach().numpy()
        sparse_vecs_cpu = outputs["sparse_vecs"].cpu().detach().numpy()
        colbert_vecs_cpu = outputs["colbert_vecs"].cpu().detach().numpy()
        input_ids_cpu = input_ids.cpu().detach().numpy()
        attention_mask_cpu = attention_mask.cpu().detach().numpy()

        # Post-processing on CPU (squeeze is now a numpy operation, not XLA)
        token_weights_cpu = sparse_vecs_cpu.squeeze(-1)

        # Process dense embeddings (same as bge_m3_encode.py)
        all_dense_embeddings.append(dense_vecs_cpu)
        all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
        all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]

        # Process sparse embeddings (lexical weights) (same as bge_m3_encode.py)
        all_lexical_weights.extend(
            list(
                map(
                    _process_token_weights,
                    token_weights_cpu,
                    input_ids_cpu.tolist(),
                )
            )
        )
        all_lexical_weights = [all_lexical_weights[i] for i in np.argsort(length_sorted_idx)]

        # Process colbert vectors (same as bge_m3_encode.py)
        all_colbert_vecs.extend(
            list(
                map(
                    _process_colbert_vecs,
                    colbert_vecs_cpu,
                    attention_mask_cpu,
                )
            )
        )
        all_colbert_vecs = [all_colbert_vecs[i] for i in np.argsort(length_sorted_idx)]

        # Return only dense_vecs for PCC calculation
        # The other embeddings (lexical_weights, colbert_vecs) were processed
        # to ensure their computation is included in the benchmark timing
        return torch.tensor(all_dense_embeddings)

    test_encoder(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=bge_m3_preprocess,
        output_processor_fn=bge_m3_output_processor,
        data_format=data_format,
        batch_size=4,
        input_sequence_length=input_sequence_length,
        loop_count=32,
        optimization_level=0,
        required_pcc=0.97,
        single_block=False,
        single_layer=False,
        dump_source=False,
        model_nickname="bge_m3",
    )


def test_unet_for_conditional_generation(output_file):
    """Test UNet for Conditional Generation model. This is a core component of the Stable Diffusion XL pipeline.

    Note: Single block/layer tests not supported - UNet is not a transformer encoder architecture.
    https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    """
    from third_party.tt_forge_models.unet_for_conditional_generation.pytorch.loader import ModelLoader

    def inputs_to_device(inputs, device):
        """Utility function to recursively move all tensors in nested dict to device."""
        result = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            elif isinstance(value, dict):
                result[key] = inputs_to_device(value, device)
            else:
                result[key] = value
        return result

    # Configuration
    data_format = "bfloat16"
    batch_size = 1
    unet_max_seqlen = 77

    loader = ModelLoader()
    model_info_name = loader.get_model_info().name
    print(f"\nLoading model {model_info_name}...")
    model = loader.load_model(dtype_override=DTYPE_MAP[data_format])

    load_inputs_fn = lambda batch_size: loader.load_inputs(batch_size=batch_size, dtype_override=DTYPE_MAP[data_format])
    preprocess_fn = lambda raw_inputs, device: inputs_to_device(raw_inputs, device)
    output_processor_fn = lambda out, inputs: out.sample

    test_encoder(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        data_format=data_format,
        batch_size=batch_size,
        input_sequence_length=unet_max_seqlen,  # for UNet it is always set to the max sequence length
        loop_count=128,
        optimization_level=1,
        single_block=False,
        single_layer=False,
        dump_source=False,
        model_nickname="unet_sdxl",
    )
