# TT-Forge-FE Model Benchmarks

This directory contains benchmark implementations of popular deep learning models using TT-Forge-FE. These benchmarks are designed to measure performance across various computer vision and natural language processing models.

## Available Benchmarks

| Model                    | Model Type     | Description                                                             | Benchmark Code                                         |   Performance (FPS)*                 |
|--------------------------|----------------|-------------------------------------------------------------------------|--------------------------------------------------------|-----------------------------|
| EfficientNet (Timm)      | CNN            | Scalable and efficient CNN model for image classification               | [`efficientnet_timm.py`](efficientnet_timm.py)        | 46.02                 |
| MNIST Linear             | MLP            | Simple linear neural network for handwritten digit classification       | [`mnist_linear.py`](mnist_linear.py)                  | 3521.22         |
| MobileNetV2 Basic        | CNN            | Lightweight CNN for mobile vision applications                          | [`mobilenetv2_basic.py`](mobilenetv2_basic.py)        | 170.91             |
| ResNet-50 (HuggingFace)  | CNN            | Deep residual network for image classification                          | [`resnet_hf.py`](resnet_hf.py)                        | 346.39             |
| SegFormer                | Transformer, CNN    | Transformer-based model for image segmentation and classification       | [`segformer.py`](segformer.py)                        | 6.95              |
| ViT (Vision Transformer) | Transformer, CNN    | Vision Transformer for image classification                             | [`vit.py`](vit.py)                                    | 10.90          |
| VovNet (OSMR)            | CNN            | Variety of Overparam Network for image classification                   | [`vovnet.py`](vovnet.py)                              | 185.18             |
| YOLOv4                   | CNN            | Real-time object detection model                                        | [`yolo_v4.py`](yolo_v4.py)                            | 7.83           |
| YOLOv8                   | CNN            | Latest YOLO version for object detection                                | [`yolo_v8.py`](yolo_v8.py)                            | 8.70           |
| YOLOv9                   | CNN            | Advanced YOLO model for object detection                                | [`yolo_v9.py`](yolo_v9.py)                            | 10.93              |
| YOLOv10                  | CNN            | Most recent YOLO iteration for object detection                         | [`yolo_v10.py`](yolo_v10.py)                            | 4.34             |
| Llama                    | LLM, Transformer           | Large language model for text generation and prefill benchmarking       | [`llama.py`](llama.py)                                | 220.60           |

*Performance column will be updated regularly.

## Running Benchmarks

Instructions for running performance benchmarks can be found [here](../../docs/src/getting_started.md).

## Additional Resources

- [TT-Forge-FE Documentation](https://docs.tenstorrent.com/tt-forge-fe/)
- [TT-Forge-FE GitHub Repository](https://github.com/tenstorrent/tt-forge-fe)

For issues or questions about benchmarks, please file them at [github.com/tenstorrent/tt-forge/issues](https://github.com/tenstorrent/tt-forge/issues).
