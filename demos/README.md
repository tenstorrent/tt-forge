# TT-Forge Demos

This directory contains demonstration examples for three different frontends available for Tenstorrent hardware:

## Available Frontends

- [TT-XLA](https://github.com/tenstorrent/tt-xla)
  - TT-XLA is the primary frontend for running PyTorch and JAX models. It leverages a PJRT interface to integrate JAX (and in the future other frameworks), TT-MLIR, and Tenstorrent hardware. It supports ingestion of JAX models via jit compile, providing StableHLO (SHLO) graph to TT-MLIR compiler. TT-XLA can be used for single and multi-chip projects.
  - See the [TT-XLA docs pages](https://docs.tenstorrent.com/tt-xla) for an overview and getting started guide.

- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
  - A TVM based graph compiler designed to optimize and transform computational graphs for deep learning models. Supports ingestion of ONNX, TensorFlow, PaddlePaddle and similar ML frameworks via TVM ([TT-TVM](https://github.com/tenstorrent/tt-tvm)). It also supports ingestion of PyTorch, however it is recommended that you use TT-XLA. TT-Forge-FE does not support multi-chip configurations; it is for single-chip projects only.
  - See the [TT-Forge-FE docs pages](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html) for an overview and getting started guide.

- [TT-Torch](https://github.com/tenstorrent/tt-torch) - (deprecated)
  - A MLIR-native, open-source, PyTorch 2.X and torch-mlir based front-end. It provides stableHLO (SHLO) graphs to TT-MLIR. Supports ingestion of PyTorch models via PT2.X compile and ONNX models via torch-mlir (ONNX->SHLO)
  - See the [TT-Torch docs pages](https://docs.tenstorrent.com/tt-torch) for an overview and getting started guide.

Each frontend is designed to support different ML frameworks and workflows. Choose the frontend that best matches your needs:
- Use TT-Forge-FE for quick deployment of pre-optimized common models for PaddlePaddle and ONNX. (Only single chip configurations are available. Also please note TT-Forge-FE can also be used for PyTorch, however it is recommended that you use TT-XLA for the best experience.)
- Use TT-XLA for JAX model deployment as well as PyTorch (single and multi-chip configurations are available).
- Use TT-Torch for PyTorch model deployment (deprecated, use TT-XLA).


For more information, visit our [GitHub repositories](https://github.com/tenstorrent) or check the README in each frontend's directory.
