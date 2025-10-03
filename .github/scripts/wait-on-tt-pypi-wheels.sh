#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TEST:
# export NEW_VERSION_TAG="0.1.0.dev20250623"
# export PIP_WHEEL_NAMES="tt-torch tt_forge_fe tt_tvm pjrt-plugin-tt"

set -eu
wait_time=60

check_wheels() {
  for wheel_name in $PIP_WHEEL_NAMES; do
    # Check the only tenstorrent index for the wheel version
    pip index versions $wheel_name --pre --index-url https://pypi.eng.aws.tenstorrent.com/ 2>/dev/null | grep -q $NEW_VERSION_TAG && echo "Found tag $NEW_VERSION_TAG for $wheel_name" || { echo "Can't find wheel $wheel_name for version $NEW_VERSION_TAG"; false; }
  done
}

while true; do
  # Wait for pypi frontend wheels to be available
  check_wheels && break
  echo "Waiting $wait_time seconds for pypi frontend wheels to be available on tt-pypi"
  sleep $wait_time
done
