# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# Models we run on Wormhole B0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #


# ==================================================================== #
# Benchmark Arguments
# ==================================================================== #
# -m:  Model name
# -ts: Task type, for example, classification
# -bs: Batch size
# -df: Data format, for example, bfloat16
# -lp: Loop count, number of times to run the model
# -o:  Output file name
# ==================================================================== #


# ------------------------------------------------------- #
# TT-Forge-Fe Compiler
# ------------------------------------------------------- #
# MNIST Linear
python benchmark/benchmark.py -p tt-forge-fe -m mnist_linear -bs 32 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-mnist-linear.json

# Resnet HF
python benchmark/benchmark.py -p tt-forge-fe -m resnet_hf -bs 8 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-resnet50_hf.json

# EfficientNet Timm
python benchmark/benchmark.py -p tt-forge-fe -m efficientnet_timm -bs 1 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-efficientnet_timm.json

# Llama
python benchmark/benchmark.py -p tt-forge-fe -m llama -bs 1 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-llama.json

# MobileNetV2 Basic
python benchmark/benchmark.py -p tt-forge-fe -m mobilenetv2_basic -bs 1 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-mobilenetv2_basic.json


# ------------------------------------------------------- #
# TT-Torch Compiler
# ------------------------------------------------------- #
# Resnet
python benchmark/benchmark.py -p tt-torch -m resnet -bs 8 -lp 32 -o forge-benchmark-e2e-tt-torch-resnet50.json



# ------------------------------------------------------- #
# TT-XLA Compiler
# ------------------------------------------------------- #
