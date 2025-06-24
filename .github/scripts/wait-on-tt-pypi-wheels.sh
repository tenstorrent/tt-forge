#!/usr/bin/env bash
#
# SPDX-License-Identifier: Apache-2.0

# TEST:
# NEW_VERSION_TAG="0.1.0.dev20250623"
# PIP_WHEEL_NAMES="tt-torch tt_forge_fe tt_tvm pjrt-plugin-tt"

set -eu
attempts=30
wait_time=10

check_wheels() {
  for wheel_name in $PIP_WHEEL_NAMES; do
    # Check the only tenstorrent index for the wheel version
    pip index versions $wheel_name --pre --index-url https://pypi.eng.aws.tenstorrent.com/ | grep $NEW_VERSION_TAG && echo "Found tag $NEW_VERSION_TAG for $wheel_name"
  done
}

n=0

until [ "$n" -ge $attempts ]; do
  # Wait for pypi frontend wheels to be available
  check_wheels && break
  n=$((n+1))
  echo "Waiting for pypi frontend wheels to be available on tt-pypi"
  sleep $wait_time
done
