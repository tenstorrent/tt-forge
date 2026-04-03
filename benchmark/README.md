# TT-Forge Model Benchmarks

This directory contains benchmark implementations of popular deep learning models using TT-Forge (TT-Forge-ONNX,and TT-XLA). These benchmarks are designed to measure performance across various computer vision and natural language processing models.

## Running Benchmarks

Instructions for running performance benchmarks can be found [here](../docs/src/getting_started.md#running-performance-benchmark-tests).

## Available Benchmarks

| Model                    | Project        | Model Type     | Description                                                             | Benchmark Code                                         | Performance (FPS)* | Target Performance (FPS) |
|--------------------------|---------------|----------------|-------------------------------------------------------------------------|--------------------------------------------------------|--------------------|---------------------------|
| ResNet-50 HF ONNX        | TT-Forge-ONNX   | CNN            | Deep residual network for image classification (bfloat16, ONNX export) | [`tt-forge-onnx/resnet50_hf_onnx.py`](tt-forge-onnx/resnet50_hf_onnx.py)         | 895.02              | 800                       |
| ResNet-50                | TT-XLA        | CNN            | Deep residual network for image classification                   | [`tt-xla/resnet.py`](tt-xla/resnet.py)                                    | 0.14                 |             500            |
