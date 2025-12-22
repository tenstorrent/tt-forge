# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT-Forge is Tenstorrent's MLIR-based compiler that integrates into various compiler technologies from AI/ML frameworks to enable running models and create custom kernel generation. This repository serves as the central hub for the tt-forge compiler project, bringing together various sub-projects into a cohesive product.

**Key Sub-Projects:**
- [TT-MLIR](https://github.com/tenstorrent/tt-mlir) - MLIR-based compiler framework
- [TT-XLA](https://github.com/tenstorrent/tt-xla) - Primary frontend for PyTorch and JAX via PJRT
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe) - TVM-based graph compiler for PyTorch, ONNX, TensorFlow, PaddlePaddle
- [TT-Torch](https://github.com/tenstorrent/tt-torch) - (Deprecated) MLIR-native PyTorch 2.X frontend

## Common Development Commands

### Running Tests

**Basic Tests** - Quick validation tests for frontends:
```bash
# Run basic test for a specific frontend
python basic_tests/tt-xla/demo_test.py
python basic_tests/tt-forge-fe/demo_test.py
```

**Benchmarks** - Performance and model testing:
```bash
# Run benchmark for a specific model
python benchmark/benchmark.py -p <project> -m <model> [options]

# Examples:
python benchmark/benchmark.py -p tt-xla -m resnet -bs 1 -lp 1
python benchmark/benchmark.py -p tt-forge-fe -m mobilenetv2_basic -bs 8 -lp 10
```

**Benchmark Options:**
- `-p, --project` - Project directory (tt-xla, tt-forge-fe, tt-torch)
- `-m, --model` - Model name (e.g., bert, resnet, mobilenetv2)
- `-c, --config` - Model configuration (e.g., tiny, base, large)
- `-bs, --batch_size` - Batch size (default: 1)
- `-lp, --loop_count` - Number of benchmark iterations (default: 1)
- `-o, --output` - Output JSON file for results
- `-df, --data_format` - Data format (default: float32)
- `-mc, --measure_cpu` - Measure CPU FPS

### Pre-commit Hooks

Pre-commit is used for code formatting and linting:
```bash
# Install pre-commit (one-time setup)
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Run on a single file
pre-commit run <hook_id>
```

## Architecture Overview

### Compiler Stack

The TT-Forge compiler stack follows this flow:

**Frontend Layer** → **TT-MLIR Compiler** → **TT-Metalium** → **Hardware**

1. **Frontend Layer**: Ingests models from ML frameworks
   - TT-XLA: JAX/PyTorch models via StableHLO
   - TT-Forge-FE: PyTorch/ONNX/TensorFlow via TVM → TTIR
   - TT-Torch (deprecated): PyTorch 2.X via StableHLO

2. **TT-MLIR Compiler**: Multi-dialect MLIR compiler with:
   - **TTIR Dialect**: Common IR for all frontends
   - **StableHLO-IR**: Standard MLIR representation from XLA/Torch frontends
   - **PyKernel**: Custom kernel definitions
   - **Graph Passes**: Optimization passes (layout transformation, op fusing, decomposition, sharding)
   - **TTNN Dialect**: Entry point to TTNN library ops
   - **TTMetal Dialect**: Direct access to tt-metalium kernels
   - **TTKernel-IR**: Low-level kernel representation

3. **TT-Metalium Layer**:
   - TTNN: Neural network operations library
   - TTMetal: Low-level programming model
   - Interfaces with LLK (Low-Level Kernels)

4. **Hardware**: Tenstorrent devices (Wormhole, Blackhole)

### Repository Structure

```
tt-forge/
├── benchmark/           # Benchmark infrastructure and model tests
│   ├── benchmark.py    # Main benchmark runner
│   ├── tt-xla/         # TT-XLA benchmarks (resnet, vit, etc.)
│   ├── tt-forge-fe/    # TT-Forge-FE benchmarks
│   └── tt-torch/       # TT-Torch benchmarks (deprecated)
├── basic_tests/        # Quick validation tests per frontend
│   ├── tt-xla/
│   ├── tt-forge-fe/
│   └── tt-torch/
├── demos/              # Demo applications and examples
├── docs/               # Documentation source
├── scripts/            # Release and automation scripts
├── third_party/        # External dependencies (tt-forge-models submodule)
└── .github/            # CI/CD workflows and release automation
```

### Benchmark Infrastructure

The benchmark system is extensible and project-agnostic:

1. **Dynamic Module Loading**: `benchmark.py` dynamically loads test modules from project directories
2. **Common Interface**: Each benchmark module must export a `benchmark(config)` function
3. **Pre/Post Hooks**: Projects can define `common.py` with `pre_test()` and `post_test()` functions for setup/teardown
4. **Results Format**: Benchmarks output JSON with measurements (total_samples, total_time, samples_per_sec, etc.)

## Release Process

This repository manages releases for all TT-Forge sub-projects using automated GitHub workflows.

### Release Types

1. **Nightly Builds**: `X.Y.0.devYYYYMMDD` - Daily builds from main branch
2. **Release Candidates**: `X.Y.0rcN` - Pre-releases on release branches
3. **Stable Releases**: `X.Y.Z` - Production releases

### Key Workflows

- **Daily Releaser** (`.github/workflows/daily-releaser.yml`): Automated nightly builds and RC/patch updates
- **Create Version Branches** (`.github/workflows/create-version-branches.yml`): Create new release branches
- **Promote Stable** (`.github/workflows/promote-stable.yml`): Promote RC to stable
- **Bump Version** (`.github/workflows/bump-version.yml`): Manual RC/patch version bumps
- **Release** (`.github/workflows/release.yml`): Core release orchestrator (build, test, publish)

### Release Artifacts

Releases produce:
- Python wheels (published to Tenstorrent PyPI)
- Docker images (tagged appropriately, 'latest' for stable)
- GitHub releases with documentation and artifacts
- MLIR artifacts (TTIR, TTNN) for debugging

## Coding Standards

### File Headers

All source files must include SPDX headers:

**C++ files:**
```cpp
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
```

**Python files:**
```python
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
```

### Error Messages

Follow these principles for error messages:
- **Be Specific**: Include actual values that caused the error
- **Explain the Issue**: Provide context on why the error occurred
- **Include Relevant Information**: Show variable values and context
- **Make it Actionable**: Provide clear guidance on how to fix the issue

Example:
```python
# Good
TT_FATAL(head_size % TILE_WIDTH == 0,
         "Invalid head size: {}. The head size must be a multiple of the tile width ({}). "
         "Please adjust the dimensions accordingly.",
         head_size, TILE_WIDTH);

# Bad
TT_FATAL(head_size % TILE_WIDTH != 0, "Head size is invalid.");
```

### Git Branch Naming

Branch names should follow:
```
<user>-<issue_number>[-optional_description]
<user>/<issue_number>[-optional-description]
```

Examples:
- `user-123`
- `user/123`
- `user-123_rename_method_x`
- `user/123-add-x-unit-test`

### Commit and PR Guidelines

- Use descriptive commit messages
- Each commit should be functional (compiles and passes tests)
- Link issues in PR descriptions under "Ticket" headline
- Avoid `git add -A` - be explicit about files added
- Use **squash and merge** or **rebase and merge** (no merge commits)
- Wait 24 hours after opening PR before merging
- Require at least 1 reviewer approval and green CI

## Testing Infrastructure

### CI/CD Workflows

- **Basic Tests** (`.github/workflows/basic-tests.yml`): Quick validation for each frontend
- **Demo Tests** (`.github/workflows/demo-tests.yml`): Comprehensive demo tests
- **Perf Benchmark** (`.github/workflows/perf-benchmark.yml`): Performance benchmarks with regression detection

### Performance Testing

Performance benchmarks:
- Run on dedicated hardware (N150, P150B)
- Generate detailed reports with MLIR artifacts
- Track metrics over time via Superset API
- Detect regressions >5% automatically
- Store TTIR and TTNN MLIR for debugging

## Hardware and Environment

### Supported Hardware

- Wormhole (N150, N300)
- Blackhole (P150B)

### Container Requirements

Tests run in Docker containers with:
- Device access: `/dev/tenstorrent`
- Huge pages: `/dev/hugepages`, `/dev/hugepages-1G`
- Kernel modules: `/lib/modules`
- Environment: `TRACY_NO_INVARIANT_CHECK=1`

## Important Notes

- **TT-Torch is deprecated** - Use TT-XLA for PyTorch models
- **Avoid interactive git commands** - Don't use `-i` flags (e.g., `git rebase -i`, `git add -i`)
- **Revert failing commits immediately** - Post-commit regressions must be reverted or fixed quickly
- **Linear history** - No merge commits allowed on main branch
- **Pre-commit hooks** - Must pass before commits
- **24-hour PR rule** - Wait at least 24 hours before merging to allow global team review

## Documentation

Main documentation: https://docs.tenstorrent.com/tt-forge/

Sub-project docs:
- [TT-XLA docs](https://docs.tenstorrent.com/tt-xla)
- [TT-Forge-FE docs](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html)
- [TT-MLIR coding guidelines](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/coding-guidelines.md)

## Bounty Program

This repository participates in the Tenstorrent Bounty Program. Issues tagged with "bounty" and difficulty levels are eligible for rewards. See: https://docs.tenstorrent.com/bounty_terms.html
