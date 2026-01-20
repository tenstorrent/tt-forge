#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This script enables workflows for a given repository or all frontends
# It reads the list of disabled workflows from <repo>-disabled-workflows.txt

enable_workflows() {
    local repo="$1"
    echo "Enabling workflows for repo: $repo"

    if [ -f "$repo-disabled-workflows.txt" ]; then
        while IFS= read -r wf; do
            echo "Enabling $wf"
            gh workflow enable "$wf" -R tenstorrent/$repo
        done < "$repo-disabled-workflows.txt"
    else
        echo "File $repo-disabled-workflows.txt does not exist."
        exit 1
    fi
}

if [ $# -ne 1 ]; then
    echo "Enabling all repos"
    enable_workflows "tt-torch"
    enable_workflows "tt-forge-onnx"
    enable_workflows "tt-xla"
else
    enable_workflows "$1"
fi
