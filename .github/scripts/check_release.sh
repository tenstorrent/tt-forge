#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# local test:
# ./check_release.sh tenstorrent/tt-forge 0.4.0

set -euo pipefail

REPO="$1"
TAG="$2"

RELEASE=$(gh release list -R $REPO --json tagName --jq ".[] | select(.tagName == \"$TAG\")")
if [ -n "$RELEASE" ]; then
    echo "true"
else
    echo "false"
fi
