# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT-Forge is Tenstorrent's MLIR-based compiler that integrates into various compiler technologies from AI/ML frameworks to enable running models and create custom kernel generation. This repository is the central hub for the tt-forge compiler project, bringing together various sub-projects into a cohesive product.

**Key Sub-Projects:**
- [TT-MLIR](https://github.com/tenstorrent/tt-mlir) - MLIR-based compiler framework
- [TT-XLA](https://github.com/tenstorrent/tt-xla) - Primary frontend for PyTorch and JAX via PJRT (single and multi-chip)
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe) - TVM-based graph compiler for ONNX, TensorFlow, PaddlePaddle (single-chip only)
- [TT-Torch](https://github.com/tenstorrent/tt-torch) - **(Deprecated)** Use TT-XLA for PyTorch

## Common Development Commands

### Running Tests

**Basic Tests** - Quick validation tests for frontends:
```bash
python basic_tests/tt-xla/demo_test.py
python basic_tests/tt-forge-fe/demo_test.py
```

### TT-XLA Benchmarks (pytest - recommended)

All TT-XLA models use pytest. The `benchmark.py` interface is deprecated for TT-XLA.

**LLM benchmarks:**
```bash
pytest -svv benchmark/tt-xla/llms.py::test_llama_3_2_1b
pytest -svv benchmark/tt-xla/llms.py::test_phi1
pytest -svv benchmark/tt-xla/llms.py::test_qwen_2_5_0_5b
pytest -svv benchmark/tt-xla/llms.py::test_falcon3_1b

# Save results to JSON
pytest -svv benchmark/tt-xla/llms.py::test_llama_3_2_1b --output results.json
```

**Vision model benchmarks:**
```bash
pytest -svv benchmark/tt-xla/vision_models.py::test_resnet50
pytest -svv benchmark/tt-xla/vision_models.py::test_vit
pytest -svv benchmark/tt-xla/vision_models.py::test_mobilenetv2
pytest -svv benchmark/tt-xla/vision_models.py::test_efficientnet
```

After running benchmarks, look for the `Sample per second:` line in the output for performance results.

### Performance Benchmark Workflow

Run performance benchmarks in CI with optional filtering:
```bash
gh workflow run "Performance benchmark" --ref <branch> -f test-filter="llama,phi,gemma" -f project-filter=tt-xla -f runs-on-filter=n150
```

### Device Reset

If the TT device hangs or gets stuck (e.g., "Timeout waiting for Ethernet" errors), reset it:
```bash
tt-smi --reset 0
```

## Architecture Overview

### Compiler Stack Flow

**Frontend Layer** → **TT-MLIR Compiler** → **TT-Metalium** → **Hardware**

1. **Frontend Layer**: Ingests models from ML frameworks
   - TT-XLA: JAX/PyTorch models via StableHLO
   - TT-Forge-FE: PyTorch/ONNX/TensorFlow via TVM → TTIR

2. **TT-MLIR Compiler**: Multi-dialect MLIR compiler
   - **TTIR Dialect**: Common IR for all frontends
   - **StableHLO-IR**: Standard MLIR representation from XLA/Torch frontends
   - **Graph Passes**: Optimization passes (layout transformation, op fusing, decomposition, sharding)
   - **TTNN Dialect**: Entry point to TTNN library ops
   - **TTMetal Dialect**: Direct access to tt-metalium kernels

3. **TT-Metalium Layer**: TTNN (neural network ops) + TTMetal (low-level programming)

4. **Hardware**: Wormhole (N150, N300), Blackhole (P150B)


## Coding Standards

### File Headers

All source files must include SPDX headers:

**Python files:**
```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
```

### Git Conventions

**Branch naming:**
```
<user>/<issue_number>[-optional-description]
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Documentation

- Main docs: https://docs.tenstorrent.com/tt-forge/
- [TT-XLA docs](https://docs.tenstorrent.com/tt-xla)
- [TT-Forge-FE docs](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html)
- [TT-MLIR coding guidelines](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/coding-guidelines.md)
