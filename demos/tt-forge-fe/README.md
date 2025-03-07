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
| ViLT | Vision-Language | Vision-and-Language Transformer for visual question answering | [`vilt_demo.py`](vilt_demo.py) |
| DeiT | Vision | Data-efficient Image Transformer for image classification | [`deit_demo.py`](deit_demo.py) |
| EfficientNet | Vision | Efficient CNN architecture using compound scaling (timm) | [`efficientnet_timm_demo.py`](efficientnet_timm_demo.py) |
| EfficientNet | Vision | Efficient CNN architecture using compound scaling (torchvision) | [`efficientnet_torchvision_demo.py`](efficientnet_torchvision_demo.py) |
| GPT-2 | NLP | Large language model for text generation | [`gpt2_demo.py`](gpt2_demo.py) |
| Falcon-3B | NLP | 3B parameter language model for text generation | [`falcon3_demo.py`](falcon3_demo.py) |

## Running the Demos

Each demo can be run directly using Python:

```bash
python vilt_demo.py                    # Run ViLT demo
python efficientnet_timm_demo.py       # Run EfficientNet (timm) demo
python gpt2_demo.py                    # Run GPT-2 demo
```

### Demo Status

Some demos may have limitations with the current version of TT-Forge-FE:

- **Fully Working**:
  - ViLT: Visual question answering
  - EfficientNet: Image classification (both timm and torchvision variants)

- **Partial Support** (Original model works, compilation in progress):
  - DeiT: Working on conv2d operator support
  - GPT-2: Working on attention mechanism support
  - Falcon-3B: Working on large model support

If you encounter any issues or have questions, please file them at [github.com/tenstorrent/tt-forge/issues](https://github.com/tenstorrent/tt-forge/issues).

## Additional Resources

- [TT-Forge-FE Documentation](https://docs.tenstorrent.com/tt-forge-fe/)
- [Getting Started Guide](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html)
- [TT-Forge-FE GitHub Repository](https://github.com/tenstorrent/tt-forge)
