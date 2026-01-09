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


def test_vision(
    model,
    model_info_name,
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
    """Test vision model with the given configuration.

    Args:
        model: Loaded model instance in eval mode
        model_info_name: Model name for identification and reporting
        output_file: Path to save benchmark results as JSON
        load_inputs_fn: Function to load raw inputs for the model.
            Signature: fn(batch_size, loop_count, channel_size, input_size) -> List[Tensor]
        preprocess_fn: Function to preprocess inputs (dtype conversion + device placement).
            Signature: fn(input_tensor, device, data_format) -> tensor on device
        output_processor_fn: Function to process model outputs (extract logits + move to CPU).
            Signature: fn(output) -> tensor on CPU
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_size: Input size tuple (height, width)
        channel_size: Number of input channels
        data_format: Data format
        measure_cpu: Measure CPU FPS
        experimental_compile: Enable experimental compile
        required_pcc: Required PCC threshold
    """
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
        model=model,
        model_info_name=model_info_name,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        training=False,
        batch_size=batch_size,
        loop_count=loop_count,
        input_size=input_size,
        channel_size=channel_size,
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

    # Configuration
    data_format = "bfloat16"
    batch_size = 8
    input_size = (224, 224)
    channel_size = 3

    # Load model
    variant = ModelVariant.TIMM_EFFICIENTNET_B0
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_mnist(output_file):
    from third_party.tt_forge_models.mnist.image_classification.pytorch.loader import ModelLoader

    # Configuration
    data_format = "bfloat16"
    batch_size = 32
    input_size = (28, 28)
    channel_size = 1

    # Load model
    loader = ModelLoader()
    model_info_name = loader.get_model_info().name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_mobilenetv2(output_file):
    from third_party.tt_forge_models.mobilenetv2.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 12
    input_size = (224, 224)
    channel_size = 3

    # Load model
    variant = ModelVariant.MOBILENET_V2_TORCH_HUB
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_resnet50(output_file):
    from third_party.tt_forge_models.resnet.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 8
    input_size = (224, 224)
    channel_size = 3

    # Load model
    variant = ModelVariant.RESNET_50_HF
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.logits.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
        required_pcc=0.90,
    )


def test_segformer(output_file):
    from third_party.tt_forge_models.segformer.semantic_segmentation.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 1
    input_size = (512, 512)
    channel_size = 3

    # Load model
    variant = ModelVariant.B0_FINETUNED
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.logits.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_swin(output_file):
    from third_party.tt_forge_models.swin.image_classification.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 1
    input_size = (512, 512)
    channel_size = 3

    # Load model
    variant = ModelVariant.SWIN_S
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
        required_pcc=0.90,
    )


def test_ufld(output_file):
    from third_party.tt_forge_models.ultra_fast_lane_detection.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 1
    channel_size = 3

    # Load model
    variant = ModelVariant.TUSIMPLE_RESNET34
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    input_size = loader.config.input_size
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_ufld_v2(output_file):
    from third_party.tt_forge_models.ultra_fast_lane_detection_v2.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 1
    channel_size = 3

    # Load model
    variant = ModelVariant.TUSIMPLE_RESNET34
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    input_size = (loader.config.input_height, loader.config.input_width)
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output[0].to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_unet(output_file):
    from third_party.tt_forge_models.vgg19_unet.pytorch.loader import ModelLoader

    # Configuration
    data_format = "bfloat16"
    batch_size = 1
    input_size = (256, 256)
    channel_size = 3

    # Load model
    loader = ModelLoader()
    model_info_name = loader.get_model_info().name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_vit(output_file):
    from third_party.tt_forge_models.vit.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 8
    input_size = (224, 224)
    channel_size = 3

    # Load model
    variant = ModelVariant.BASE
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.logits.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )


def test_vovnet(output_file):
    from third_party.tt_forge_models.vovnet.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = "bfloat16"
    batch_size = 8
    input_size = (224, 224)
    channel_size = 3

    # Load model
    variant = ModelVariant.TIMM_VOVNET19B_DW_RAIN1K
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    if data_format == "bfloat16":
        model = model.to(torch.bfloat16)
    model = model.eval()

    def load_inputs_fn(batch_size, loop_count, channel_size, input_size):
        torch.manual_seed(1)
        return [torch.randn(batch_size, channel_size, *input_size) for _ in range(loop_count)]

    def preprocess_fn(input_tensor, device, data_format):
        if data_format == "bfloat16":
            input_tensor = input_tensor.to(torch.bfloat16)
        return input_tensor.to(device)

    def output_processor_fn(output):
        return output.to("cpu")

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        load_inputs_fn=load_inputs_fn,
        preprocess_fn=preprocess_fn,
        output_processor_fn=output_processor_fn,
        batch_size=batch_size,
        input_size=input_size,
        channel_size=channel_size,
        data_format=data_format,
    )
