# TT-Forge-FE Model Demos

This directory contains example implementations of popular deep learning models using TT-Forge-FE. These demos showcase how to use TT-Forge-FE to run inference on various computer vision and natural language processing models.

## Installation

Before running the demos, make sure you have TT-Forge-FE installed. You can install it using wheels from our latest release:

1. Download the latest TT-Forge-FE release (includes both TVM and tt-Forge-fe wheels):
   - Visit [TT-Forge Releases](https://github.com/tenstorrent/tt-forge/releases)

2. Install the wheels:
   ```bash
   pip install *.whl
   ```
   Note: Run this command from the directory where you downloaded the wheels.

## Available Demos

| Model | Model Type | Description | Demo Code |
|-------|------------|-------------|------------|
| MobileNetV2 | CNN | Lightweight convolutional neural network for efficient image classification | [`cnn/mobile_netv2_demo.py`](cnn/mobile_netv2_demo.py) |
| ResNet-50 | CNN | Deep residual network for image classification | [`cnn/resnet_50_demo.py`](cnn/resnet_50_demo.py) |
| ViLT | CNN | Vision-and-Language Transformer for vision-language tasks | [`cnn/vilt_demo.py`](cnn/vilt_demo.py) |
| BERT | NLP | Bidirectional Encoder Representations from Transformers for natural language understanding tasks | [`nlp/bert_demo.py`](nlp/bert_demo.py) |

## Running the Demos

Each demo can be run directly using Python. Navigate to the specific model directory and run the demo script:

```bash
python cnn/resnet_50_demo.py
```

If you encounter any issues or have questions, please file them at [github.com/tenstorrent/tt-forge/issues](https://github.com/tenstorrent/tt-forge/issues).

## Additional Resources

- [TT-Forge-FE Documentation](https://docs.tenstorrent.com/tt-forge-fe/)
- [Getting Started Guide](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html)
- [TT-Forge-FE GitHub Repository](https://github.com/tenstorrent/tt-forge)
