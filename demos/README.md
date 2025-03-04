# TT-Forge Demos

This directory contains demonstration examples for three different frontends available for Tenstorrent hardware:

## Available Frontends

### [`tt-forge-fe/`](tt-forge-fe/)
The Frontend Engine (FE) provides a high-level interface for deploying popular deep learning models. It includes ready-to-use implementations of common CNN and NLP models like ResNet, MobileNet, and BERT.

### [`tt-torch/`](https://github.com/tenstorrent/tt-torch) (Coming soon)
A PyTorch-based frontend that enables seamless deployment of PyTorch models on Tenstorrent hardware. This frontend provides familiar PyTorch workflows while leveraging Tenstorrent's acceleration capabilities.

### [`tt-xla/`](https://github.com/tenstorrent/tt-xla) (Coming soon)
An XLA-based frontend that supports JAX and TensorFlow models. This enables users to deploy models from these frameworks directly to Tenstorrent hardware while maintaining their existing development workflow.

Each frontend is designed to support different ML frameworks and workflows. Choose the frontend that best matches your needs:
- Use `tt-forge-fe` for quick deployment of pre-optimized common models
- Use `tt-torch` for PyTorch model deployment
- Use `tt-xla` for JAX and TensorFlow model deployment

For more information, visit our [GitHub repositories](https://github.com/tenstorrent) or check the README in each frontend's directory.