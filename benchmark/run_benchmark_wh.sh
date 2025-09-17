# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #
# Models we run on Wormhole B0
# ------------------------------------------------------------------------------------------------------------------------------------------------------------ #


# ================================================================================================== #
# Benchmark Arguments
# ================================================================================================== #
# -m:  Model name
# -p:  Project name, actually frontend compiler name, for example, tt-forge-fe, tt-torch, tt-xla
# -ts: Task type, for example, classification
# -bs: Batch size
# -df: Data format, for example, bfloat16
# -lp: Loop count, number of times to run the model
# -o:  Output file name
# ================================================================================================== #


# ------------------------------------------------------- #
# TT-Forge-Fe Compiler
# ------------------------------------------------------- #
# Llama Prefill
python benchmark/benchmark.py -p tt-forge-fe -m llama_prefill -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-llama_prefill.json

# Llama Decode
python benchmark/benchmark.py -p tt-forge-fe -m llama_decode -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-llama_decode.json

# MNIST Linear
python benchmark/benchmark.py -p tt-forge-fe -m mnist_linear -bs 32 -df float32 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-mnist_linear.json

# Resnet HF
python benchmark/benchmark.py -p tt-forge-fe -m resnet_hf -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-resnet_hf.json

# MobileNetV2 Basic
python benchmark/benchmark.py -p tt-forge-fe -m mobilenetv2_basic -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-mobilenetv2_basic.json

# EfficientNet Timm
python benchmark/benchmark.py -p tt-forge-fe -m efficientnet_timm -ts classification -bs 6 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-efficientnet_timm.json

# Segformer
python benchmark/benchmark.py -p tt-forge-fe -m segformer -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-segformer.json

# ViT
python benchmark/benchmark.py -p tt-forge-fe -m vit -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-vit.json

# Vovnet
python benchmark/benchmark.py -p tt-forge-fe -m vovnet -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-vovnet.json

# YOLO v4
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v4 -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-yolo_v4.json

# YOLO v8
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v8 -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-yolo_v8.json

# YOLO v9
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v9 -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-yolo_v9.json

# YOLO v10
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v10 -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-yolo_v10.json

# ------------------------------------------------------- #
# TT-Torch Compiler
# ------------------------------------------------------- #
# Resnet
python benchmark/benchmark.py -p tt-torch -m resnet -bs 8 -lp 32 -o forge-benchmark-e2e-tt-torch-resnet.json

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
