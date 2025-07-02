#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TEST:
# NEW_VERSION_TAG="0.1.0.dev20250623"
# PIP_WHEEL_NAMES="tt-torch tt_forge_fe tt_tvm pjrt-plugin-tt"

set -eu
attempts=30
wait_time=10

echo "Installing wheels: $PIP_WHEEL_NAMES"
echo "Version: $WHL_VERSION"

for wheel_name in $PIP_WHEEL_NAMES; do
  # Check the only tenstorrent index for the wheel version
  pip install $wheel_name==$WHL_VERSION --pre --no-cache-dir --extra-index-url https://pypi.eng.aws.tenstorrent.com/
done
