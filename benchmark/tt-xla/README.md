# TT-XLA Benchmarks

This directory contains benchmarks for running models via the TT-XLA frontend.

## Running Benchmarks

There are two ways to run benchmarks: using pytest (recommended) or using the legacy benchmark.py runner.

> **Note:** All models are transitioning to pytest. The benchmark.py interface will be deprecated in favor of pytest.

### Pytest (Recommended)

Each model has its own test function:

```bash
# LLMs
pytest -svv benchmark/tt-xla/llms.py::test_llama_3_2_1b
pytest -svv benchmark/tt-xla/llms.py::test_phi1

# Save results to JSON
pytest -svv benchmark/tt-xla/llms.py::test_llama_3_2_1b --output results.json
```

### Legacy Runner (benchmark.py)

Some vision models still use the legacy runner:

```bash
# ResNet
python benchmark/benchmark.py -p tt-xla -m resnet -bs 8 -lp 64 -df bfloat16

# VIT
python benchmark/benchmark.py -p tt-xla -m vit -bs 1 -lp 10

# MobileNetV2
python benchmark/benchmark.py -p tt-xla -m mobilenetv2 -bs 8 -lp 32 -df bfloat16
```

## Output

After running a benchmark, look for the `Sample per second:` line in the output to see the performance result.
