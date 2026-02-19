# TT-Forge Model Benchmarks

This directory contains benchmark implementations of popular deep learning models using TT-Forge (TT-Forge-ONNX,and TT-XLA). These benchmarks are designed to measure performance across various computer vision and natural language processing models.

## Running Benchmarks

Instructions for running performance benchmarks can be found [here](../docs/src/getting_started.md#running-performance-benchmark-tests).

## Available Benchmarks

| Model                    | Project        | Model Type     | Description                                                             | Benchmark Code                                         | Performance (FPS)* | Target Performance (FPS) |
|--------------------------|---------------|----------------|-------------------------------------------------------------------------|--------------------------------------------------------|--------------------|---------------------------|
| EfficientNet (Timm)      | TT-Forge-ONNX   | CNN            | Scalable and efficient CNN model for image classification               | [`tt-forge-onnx/efficientnet_timm.py`](tt-forge-onnx/efficientnet_timm.py)        | 40.94                |       na                 |
| MobileNetV2 Basic        | TT-Forge-ONNX   | CNN            | Lightweight CNN for mobile vision applications                          | [`tt-forge-onnx/mobilenetv2_basic.py`](tt-forge-onnx/mobilenetv2_basic.py)        | 168.55                | 70                       |
| ResNet-50 (HuggingFace)  | TT-Forge-ONNX   | CNN            | Deep residual network for image classification                          | [`tt-forge-onnx/resnet_hf.py`](tt-forge-onnx/resnet_hf.py)                        | 358.26                 | 500                       |
| SegFormer                | TT-Forge-ONNX   | Transformer, CNN    | Transformer-based model for image segmentation and classification       | [`tt-forge-onnx/segformer.py`](tt-forge-onnx/segformer.py)                        | 11.66                 | 40                       |
| ViT (Vision Transformer) | TT-Forge-ONNX   | Transformer, CNN    | Vision Transformer for image classification                             | [`tt-forge-onnx/vit.py`](tt-forge-onnx/vit.py)                                    | 23.58                 |           na              |
| VovNet (OSMR)            | TT-Forge-ONNX   | CNN            | Variety of Overparam Network for image classification                   | [`tt-forge-onnx/vovnet.py`](tt-forge-onnx/vovnet.py)                              | 248                | 240                       |
| ResNet-50                | TT-XLA        | CNN            | Deep residual network for image classification                   | [`tt-xla/resnet.py`](tt-xla/resnet.py)                                    | 0.14                 |             500            |
