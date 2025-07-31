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
# MNIST Linear
python benchmark/benchmark.py -p tt-forge-fe -m mnist_linear -bs 32 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-mnist-linear.json

# Resnet HF
python benchmark/benchmark.py -p tt-forge-fe -m resnet_hf -ts classification -bs 8 -df bfloat16 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-resnet50_hf.json

# EfficientNet Timm
python benchmark/benchmark.py -p tt-forge-fe -m efficientnet_timm -ts classification -bs 6 -df bfloat16 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-efficientnet_timm.json

# Llama
python benchmark/benchmark.py -p tt-forge-fe -m llama -bs 1 -df float32 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-llama.json

# MobileNetV2 Basic
python benchmark/benchmark.py -p tt-forge-fe -m mobilenetv2_basic -ts classification -bs 8 -df bfloat16 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-mobilenetv2_basic.json

# Segformer Classification
python benchmark/benchmark.py -p tt-forge-fe -m segformer -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-segformer.json

# ViT Base
python benchmark/benchmark.py -p tt-forge-fe -m vit -ts classification -bs 1 -df bfloat16 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-vit_base.json

# Vovnet OSMR
python benchmark/benchmark.py -p tt-forge-fe -m vovnet -ts classification -bs 8 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-vovnet_osmr.json

# Yolo4
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v4 -ts na -bs 1 -df bfloat16 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-yolo_v4.json

# Yolo8
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v8 -ts na -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-yolo_v8.json

# Yolo9
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v9 -ts na -bs 1 -df bfloat16 -lp 32 -o forge-benchmark-e2e-tt-forge-fe-yolo_v9.json

# Yolo10
python benchmark/benchmark.py -p tt-forge-fe -m yolo_v10 -ts na -bs 1 -df bfloat16 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-yolo_v10.json

# Unet
python benchmark/benchmark.py -p tt-forge-fe -m unet -ts na -bs 1 -df bfloat16 -lp 32 -o orge-benchmark-e2e-tt-forge-fe-unet.json

# ------------------------------------------------------- #
# TT-Torch Compiler
# ------------------------------------------------------- #
# Resnet
python benchmark/benchmark.py -p tt-torch -m resnet -bs 8 -lp 32 -o forge-benchmark-e2e-tt-torch-resnet50.json



# ------------------------------------------------------- #
# TT-XLA Compiler
# ------------------------------------------------------- #
# Resnet
python benchmark/benchmark.py -p tt-xla -m resnet -bs 8 -lp 4 -o forge-benchmark-e2e-tt-xla-resnet50.json
