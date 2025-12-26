# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import torch

from benchmark.utils import aggregate_ttnn_perf_metrics, sanitize_filename
from vision_benchmark import benchmark_vision_torch_xla

# Defaults for all vision models
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TRACE_ENABLED = False
DEFAULT_BATCH_SIZE = 1
DEFAULT_LOOP_COUNT = 128
DEFAULT_INPUT_SIZE = (224, 224)
DEFAULT_CHANNEL_SIZE = 3
DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_MEASURE_CPU = False
DEFAULT_EXPERIMENTAL_COMPILE = True
DEFAULT_REQUIRED_PCC = 0.97
DEFAULT_READ_LOGITS_FN = lambda output: output


def test_vision(
    ModelLoaderModule,
    variant,
    output_file,
    load_inputs_fn,
    preprocess_fn,
    output_processor_fn,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_size=DEFAULT_INPUT_SIZE,
    channel_size=DEFAULT_CHANNEL_SIZE,
    data_format=DEFAULT_DATA_FORMAT,
    measure_cpu=DEFAULT_MEASURE_CPU,
    experimental_compile=DEFAULT_EXPERIMENTAL_COMPILE,
    required_pcc=DEFAULT_REQUIRED_PCC,
):
    """Test vision model with the given variant and optional configuration overrides.

    Args:
        ModelLoaderModule: Model loader class
        variant: Model variant identifier (can be None for models without variants)
        output_file: Path to save benchmark results as JSON
        load_inputs_fn: Function to load raw inputs. Signature: fn() -> torch.Tensor
        preprocess_fn: Function to preprocess inputs (move to device). Signature: fn(raw_inputs, device) -> torch.Tensor
        output_processor_fn: Function to process model outputs. Signature: fn(output) -> output
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_size: Input size tuple (height, width)
        channel_size: Number of channels
        data_format: Data format
        measure_cpu: Measure CPU FPS
        experimental_compile: Enable experimental compile
        required_pcc: Required PCC threshold
    """
    model_loader = ModelLoaderModule(variant=variant) if variant else ModelLoaderModule()
    model_info_name = (
        model_loader.get_model_info(variant=variant).name if variant else model_loader.get_model_info().name
    )
    # Sanitize model name for safe filesystem usage
    sanitized_model_name = sanitize_filename(model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{sanitized_model_name}_perf_metrics"

    print(f"Running vision benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    batch_size={batch_size}
    loop_count={loop_count}
    input_size={input_size}
    channel_size={channel_size}
    data_format={data_format}
    measure_cpu={measure_cpu}
    experimental_compile={experimental_compile}
    required_pcc={required_pcc}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_vision_torch_xla(
        model_loader=model_loader,
        model_variant=variant,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        training=False,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        loop_count=loop_count,
        data_format=data_format,
        measure_cpu=measure_cpu,
        experimental_compile=experimental_compile,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        required_pcc=required_pcc,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_efficientnet(output_file):
    from third_party.tt_forge_models.efficientnet.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.TIMM_EFFICIENTNET_B0
    batch_size = 8
    input_size = DEFAULT_INPUT_SIZE
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
    )


def test_mnist(output_file):
    from third_party.tt_forge_models.mnist.image_classification.pytorch.loader import ModelLoader

    batch_size = 32
    input_size = (28, 28)
    channel_size = 1
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=None,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
    )


def test_mobilenetv2(output_file):
    from third_party.tt_forge_models.mobilenetv2.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.MOBILENET_V2_TORCH_HUB
    batch_size = 12
    input_size = DEFAULT_INPUT_SIZE
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
    )


def test_resnet50(output_file):
    from third_party.tt_forge_models.resnet.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.RESNET_50_HF
    batch_size = 8
    input_size = DEFAULT_INPUT_SIZE
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output.logits

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        required_pcc=0.90,
    )


def test_segformer(output_file):
    from third_party.tt_forge_models.segformer.semantic_segmentation.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.B0_FINETUNED
    batch_size = 1
    input_size = (512, 512)
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output.logits

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
    )


def test_swin(output_file):
    from third_party.tt_forge_models.swin.image_classification.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.SWIN_S
    batch_size = 1
    input_size = (512, 512)
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        required_pcc=0.90,
    )


def test_ufld(output_file):
    from third_party.tt_forge_models.ultra_fast_lane_detection.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.TUSIMPLE_RESNET34
    model_loader = ModelLoader(variant)
    input_size = model_loader.config.input_size
    batch_size = 1
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
    )


def test_ufld_v2(output_file):
    from third_party.tt_forge_models.ultra_fast_lane_detection_v2.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.TUSIMPLE_RESNET34
    model_loader = ModelLoader(variant)
    input_size = (model_loader.config.input_height, model_loader.config.input_width)
    batch_size = 1
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output[0]

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
    )


def test_unet(output_file):
    from third_party.tt_forge_models.vgg19_unet.pytorch.loader import ModelLoader

    batch_size = 1
    input_size = (256, 256)
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=None,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
    )


def test_vit(output_file):
    from third_party.tt_forge_models.vit.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.BASE
    batch_size = 8
    input_size = DEFAULT_INPUT_SIZE
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output.logits

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
    )


def test_vovnet(output_file):
    from third_party.tt_forge_models.vovnet.pytorch.loader import ModelLoader, ModelVariant

    variant = ModelVariant.TIMM_VOVNET19B_DW_RAIN1K
    batch_size = 8
    input_size = DEFAULT_INPUT_SIZE
    channel_size = DEFAULT_CHANNEL_SIZE
    data_format = DEFAULT_DATA_FORMAT

    # Define 3 functions
    def load_inputs_fn():
        torch.manual_seed(1)
        inputs = torch.randn(batch_size, channel_size, *input_size)
        if data_format == "bfloat16":
            inputs = inputs.to(torch.bfloat16)
        return inputs

    preprocess_fn = lambda input_batch, device: input_batch.to(device)
    output_processor_fn = lambda output: output

    test_vision(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
    )
