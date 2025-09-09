#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Wheel stub testing for https://github.com/wheelnext/wheel-stub for pypi offical upload

source venv/bin/activate
pip install wheel-stub setuptools wheel build

WHEEL_TAG=tt-mlir-0.1.0.dev20250606
WHEEL_VERSION=ttmlir-0.1.0.dev20250606
WHEEL_TYPE=cp311-cp311-manylinux_2_34_x86_64.whl
wget --no-clobber https://github.com/tenstorrent/tt-forge/releases/download/$WHEEL_TAG/$WHEEL_VERSION-$WHEEL_TYPE

#INDEX_URL=https://github.com/tenstorrent/tt-forge/releases/expanded_assets/tt-torch-0.1.0.dev20250606

rm -r dist

python -m build --sdist --config-setting source_wheel=$WHEEL_VERSION-$WHEEL_TYPE

cd dist
tar -xvzf *.tar.gz
rm *.tar.gz
mkdir -p wheel_explode
cp ../*.whl wheel_explode/.
cd wheel_explode
wheel unpack $WHEEL_VERSION-cp311-cp311-linux_x86_64.whl
cp -r $WHEEL_VERSION/$WHEEL_VERSION.dist-info/. ../$WHEEL_VERSION/.
cd ..
tar -czvf $WHEEL_VERSION.tar.gz $WHEEL_VERSION
