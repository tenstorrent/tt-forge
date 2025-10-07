#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TEST:
# export NEW_VERSION_TAG="0.1.0.dev20250623"
# export PIP_WHEEL_NAMES="tt-torch tt_forge_fe tt_tvm pjrt-plugin-tt"

set -eu
# 60 seconds per attempt * 120 attempts = 2 hours
wait_time=60
attempts=120

check_wheels() {
  local output=""
  output=""
  for wheel_name in $PIP_WHEEL_NAMES; do
    # Check the only tenstorrent index for the wheel version
    output+="\n$(pip index versions $wheel_name --pre --index-url https://pypi.eng.aws.tenstorrent.com/ 2>/dev/null | grep -q $NEW_VERSION_TAG && echo "Found tag $NEW_VERSION_TAG for $wheel_name" || { echo "Can't find wheel $wheel_name for version $NEW_VERSION_TAG"; false; })"
  done
  echo "$output"
}

n=1

until [ "$n" -ge $attempts ]; do
  # Wait for pypi frontend wheels to be available
  check_wheels_output=$(check_wheels)
  echo -e "--------------------------------"
  if echo "$check_wheels_output" | grep -q "Can't find wheel"; then
    echo -e "$check_wheels_output"
    echo -e "Waiting $wait_time seconds on attempt $n for pypi frontend wheels to be available on tt-pypi"
    n=$((n+1))
    sleep $wait_time
    continue
  fi
  echo -e "$check_wheels_output"
  echo -e "All wheels are available on tt-pypi\n"
  break
done
