# TT-Forge-ONNX Model Demos

This directory contains example implementations of popular deep learning models using TT-Forge-ONNX. These demos showcase how to use TT-Forge-ONNX to run inference on various computer vision and natural language processing models.

>**NOTE:** TT-Forge-ONNX does not support multi-chip configurations; it is for single-chip projects only.

## Directory Structure

The demos are organized by framework and model type:

- **`onnx/`** - Demos that convert PyTorch models to ONNX format
  - `cnn/` - Computer vision models (ResNet, etc.)

- **`paddlepaddle/`** - Demos using PaddlePaddle models
  - `cnn/` - Computer vision models (ResNet, AlexNet, DenseNet, GoogLeNet, MobileNetV2)
  - `multimodal/` - Multimodal models (BLIP)

## Available Demos

### ONNX Demos (PyTorch → ONNX)

| Model                    | Model Type | Description                                                             | Demo Code                                              |
|--------------------------|------------|-------------------------------------------------------------------------|--------------------------------------------------------|
| ResNet                   | CNN        | Deep residual network for image classification (converted to ONNX)      | [`onnx/cnn/resnet_demo.py`](onnx/cnn/resnet_demo.py)   |

### PaddlePaddle Demos

| Model                    | Model Type | Description                                                             | Demo Code                                              |
|--------------------------|------------|-------------------------------------------------------------------------|--------------------------------------------------------|
| AlexNet                  | CNN        | Classic deep CNN architecture for image classification                  | [`paddlepaddle/cnn/alexnet_demo.py`](paddlepaddle/cnn/alexnet_demo.py) |
| DenseNet                 | CNN        | Densely connected convolutional network for image classification        | [`paddlepaddle/cnn/densenet_demo.py`](paddlepaddle/cnn/densenet_demo.py) |
| GoogLeNet                | CNN        | Inception architecture for image classification                         | [`paddlepaddle/cnn/googlenet_demo.py`](paddlepaddle/cnn/googlenet_demo.py) |
| MobileNetV2              | CNN        | Lightweight CNN for mobile vision applications                          | [`paddlepaddle/cnn/mobilenetv2_demo.py`](paddlepaddle/cnn/mobilenetv2_demo.py) |
| ResNet                   | CNN        | Deep residual network for image classification                          | [`paddlepaddle/cnn/resnet_demo.py`](paddlepaddle/cnn/resnet_demo.py) |
| BLIP                     | Multimodal | Vision-language model for image captioning and understanding            | [`paddlepaddle/multimodal/blip_demo.py`](paddlepaddle/multimodal/blip_demo.py) |

## Running the Demos

For details about how to set up an environment and run a demo, please see the [tt-forge Getting Started](../../docs/src/getting-started.md) page.

If you encounter any issues or have questions, please file them at [github.com/tenstorrent/tt-forge/issues](https://github.com/tenstorrent/tt-forge/issues).

## Additional Resources

- [TT-Forge-ONNX Documentation](https://docs.tenstorrent.com/tt-forge-onnx/)
- [Getting Started Guide](https://docs.tenstorrent.com/tt-forge-onnx/getting-started.html)
- [TT-Forge-ONNX GitHub Repository](https://github.com/tenstorrent/tt-forge)
