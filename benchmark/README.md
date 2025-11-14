# TT-Forge Model Benchmarks

This directory contains benchmark implementations of popular deep learning models using TT-Forge (TT-Forge-FE, TT-XLA, and TT-Torch (deprecated)). These benchmarks are designed to measure performance across various computer vision and natural language processing models.

## Available Benchmarks

| Model                    | Project        | Model Type     | Description                                                             | Benchmark Code                                         | Performance (FPS)* | Target Performance (FPS) |
|--------------------------|---------------|----------------|-------------------------------------------------------------------------|--------------------------------------------------------|--------------------|---------------------------|
| EfficientNet (Timm)      | TT-Forge-FE   | CNN            | Scalable and efficient CNN model for image classification               | [`tt-forge-fe/efficientnet_timm.py`](tt-forge-fe/efficientnet_timm.py)        | 40.94                |       na                 |
| MobileNetV2 Basic        | TT-Forge-FE   | CNN            | Lightweight CNN for mobile vision applications                          | [`tt-forge-fe/mobilenetv2_basic.py`](tt-forge-fe/mobilenetv2_basic.py)        | 168.55                | 70                       |
| ResNet-50 (HuggingFace)  | TT-Forge-FE   | CNN            | Deep residual network for image classification                          | [`tt-forge-fe/resnet_hf.py`](tt-forge-fe/resnet_hf.py)                        | 358.26                 | 500                       |
| SegFormer                | TT-Forge-FE   | Transformer, CNN    | Transformer-based model for image segmentation and classification       | [`tt-forge-fe/segformer.py`](tt-forge-fe/segformer.py)                        | 11.66                 | 40                       |
| ViT (Vision Transformer) | TT-Forge-FE   | Transformer, CNN    | Vision Transformer for image classification                             | [`tt-forge-fe/vit.py`](tt-forge-fe/vit.py)                                    | 23.58                 |           na              |
| VovNet (OSMR)            | TT-Forge-FE   | CNN            | Variety of Overparam Network for image classification                   | [`tt-forge-fe/vovnet.py`](tt-forge-fe/vovnet.py)                              | 248                | 240                       |
| ResNet-50                | TT-XLA        | CNN            | Deep residual network for image classification                   | [`tt-xla/resnet.py`](tt-xla/resnet.py)                                    | 0.14                 |             500            |
| ResNet-50                | TT-Torch (deprecated)      | CNN            | Deep residual network for image classification        | [`tt-torch/resnet.py`](tt-torch/resnet.py)                                | 4.32                 |       500                  |

> *Last updated: July 31, 2025

## Running Benchmarks

Instructions for running performance benchmarks can be found [here](../docs/src/getting_started.md#running-performance-benchmark-tests).

## Additional Resources

- [TT-Forge-FE Documentation](https://docs.tenstorrent.com/tt-forge-fe/)
- [TT-Forge-FE GitHub Repository](https://github.com/tenstorrent/tt-forge-fe)

For issues or questions about benchmarks, please file them at [github.com/tenstorrent/tt-forge/issues](https://github.com/tenstorrent/tt-forge/issues).
