#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This script disable all workflows for a given repository
# It generates a list of disabled workflows in <repo>-disabled-workflows.txt

if [ $# -ne 1 ]; then
    echo "Usage: $0 <repo>"
    exit 1
fi

repo="$1"
rm -f $repo-disabled-workflows.txt

gh workflow list -R tenstorrent/$repo --json name,id,state,path >$repo-workflows.json
set +e
jq -r '.[] | select(.state == "active") | .id' $repo-workflows.json | while read -r workflow; do
    echo "Disabling $workflow"
    gh workflow disable "$workflow" -R tenstorrent/$repo
    if [ $? -ne 0 ]; then
        echo "Failed to disable workflow $workflow"
    else
        echo "$workflow" >>$repo-disabled-workflows.txt
    fi
done
rm -f $repo-workflows.json
echo "Disabled workflows saved to $repo-disabled-workflows.txt"
echo "To re-enable, run: ./enable-workflows.sh $repo"
