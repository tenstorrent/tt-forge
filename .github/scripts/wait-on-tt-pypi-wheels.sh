#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# LOCAL TEST:
# export NEW_VERSION_TAG="0.4.0.dev20250904"
# export PIP_WHEEL_NAMES="tt-torch tt_forge_fe tt_tvm pjrt_plugin_tt"
# export ALL_REPOS="tenstorrent/tt-forge-fe tenstorrent/tt-torch tenstorrent/tt-xla"

set -eu
attempts=30
wait_time=10

# Create an associative array(map) to store the wheel names and their release urls
declare -A env_map

pip_wheel_names_array=($PIP_WHEEL_NAMES)

check_wheels() {
  for repo in $ALL_REPOS; do
  # Iterate over wheels names only for tt-forge
    for wheel_name in $PIP_WHEEL_NAMES; do
      # Check each repo's release for a wheel download url
      release_urls=$(gh release view -R $repo $NEW_VERSION_TAG --json assets | jq -r '.assets[] | select(.url | contains(".whl")) | .url' | xargs)
      for release_url in $release_urls; do
        if [[ $release_url == *"$wheel_name"* ]]; then
          echo "Found $repo $wheel_name $release_url"
          env_map[$wheel_name]=$release_url
        fi
      done
    done
  done
}

n=0

until [ "$n" -ge $attempts ]; do
  # Wait for pypi frontend wheels to be available
  check_wheels
  if [[ ${#env_map[@]} -eq $(echo $PIP_WHEEL_NAMES | wc -w) ]]; then
    break
  fi
  n=$((n+1))
  echo "Waiting for all frontend wheels to be available on github release"
  sleep $wait_time
done

# Dump the env map to a file
echo "Dumping env map to file"
for env_key in "${!env_map[@]}"; do
  echo "$env_key=${env_map[$env_key]}" >> /tmp/WHEEL_ENV_MAP
done
cat /tmp/WHEEL_ENV_MAP