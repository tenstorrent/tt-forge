# TT-Forge Demos

This directory contains demonstration examples for three different frontends available for Tenstorrent hardware:

## Available Frontends

### [`tt-forge-fe`](https://github.com/tenstorrent/tt-forge-fe)
The Frontend Engine (FE) provides a high-level interface for deploying popular deep learning models. It includes ready-to-use implementations of common CNN and NLP models like ResNet, MobileNet, and BERT.

### [`tt-xla`](https://github.com/tenstorrent/tt-xla)
An XLA-based frontend that natively supports JAX models and has support for PyTorch models through [PyTorch/XLA](https://pytorch.org/xla). This enables users to deploy models from these frameworks directly to Tenstorrent hardware while maintaining their existing development workflow.

### [`tt-torch`](https://github.com/tenstorrent/tt-torch) - (deprecated)
A PyTorch-based frontend that enables seamless deployment of PyTorch models on Tenstorrent hardware. This frontend provides familiar PyTorch workflows while leveraging Tenstorrent's acceleration capabilities.

Each frontend is designed to support different ML frameworks and workflows. Choose the frontend that best matches your needs:
- Use `tt-forge-fe` for quick deployment of pre-optimized common models (only single chip configurations are available)
- Use `tt-xla` for JAX model deployment as well as PyTorch (single and multi-chip configurations are available)
- Use `tt-torch` for PyTorch model deployment (deprecated, use `tt-xla`)


For more information, visit our [GitHub repositories](https://github.com/tenstorrent) or check the README in each frontend's directory.
