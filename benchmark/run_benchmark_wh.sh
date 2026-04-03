# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# Models we run on Wormhole B0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #


# ================================================================================================== #
# Benchmark Arguments
# ================================================================================================== #
# -m:  Model name
# -p:  Project name, actually frontend compiler name, for example, tt-forge-onnx, tt-torch, tt-xla
# -ts: Task type, for example, classification
# -bs: Batch size
# -df: Data format, for example, bfloat16
# -lp: Loop count, number of times to run the model
# -wc: warmup count, number of warmup iterations to run before the timed benchmark
# -o:  Output file name
# ================================================================================================== #


# ------------------------------------------------------- #
# TT-Forge-Onnx Compiler
# ------------------------------------------------------- #
# Resnet50 HF Onnx
python benchmark/benchmark.py -p tt-forge-onnx -m resnet50_hf_onnx -ts classification -bs 8 -df bfloat16 -lp 128 -wc 32 -o forge-benchmark-e2e-tt-forge-onnx-resnet50_hf_onnx.json

# ------------------------------------------------------- #
# TT-XLA Compiler
# ------------------------------------------------------- #
# Resnet JAX
python benchmark/benchmark.py -p tt-xla -m resnet_jax -bs 8 -lp 4 -o forge-benchmark-e2e-tt-xla-resnet_jax.json

# Resnet
python benchmark/benchmark.py -p tt-xla -m resnet -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-xla-resnet.json

# MobileNetV2
python benchmark/benchmark.py -p tt-xla -m mobilenetv2 -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-xla-mobilenetv2.json

# EfficientNet
python benchmark/benchmark.py -p tt-xla -m efficientnet -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-xla-efficientnet.json

# Segformer
python benchmark/benchmark.py -p tt-xla -m segformer -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-xla-segformer.json

# UNet
python benchmark/benchmark.py -p tt-xla -m unet -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-xla-unet.json

# ViT
python benchmark/benchmark.py -p tt-xla -m vit -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-xla-vit.json

# Vovnet
python benchmark/benchmark.py -p tt-xla -m vovnet -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-xla-vovnet.json
