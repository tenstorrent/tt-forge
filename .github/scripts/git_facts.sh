#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

current_release_tag=$(git tag --merged origin/$branch --sort=creatordate | grep -P '\d+.\d+\.\d+' | tail -n 1)
current_release_tag_commit=$(git show-ref -s $current_release_tag)
current_release_tag_commit_date=$(git show --no-patch --format=%ct $current_release_tag_commit)

# Failed if tag is not found in branch
git branch -a --contains $current_release_tag | grep -q "remotes/origin/$branch" || { echo "Could not find tag $current_release_tag in branch $branch"; exit 1; }

# Get the latest branch commit
latest_branch_commit=$(git rev-parse origin/$branch)
latest_branch_commit_date=$(git show --no-patch --format=%ct $latest_branch_commit)

release_tag_equals_latest_commit=false
if [[ "$current_release_tag_commit" == "$latest_branch_commit" ]]; then
    release_tag_equals_latest_commit=true
fi

echo "branch=$branch"
echo "current_release_tag_commit=$current_release_tag_commit"
echo "current_release_tag=$current_release_tag"
echo "current_release_tag_commit_date=$current_release_tag_commit_date"
echo "latest_branch_commit=$latest_branch_commit"
echo "latest_branch_commit_date=$latest_branch_commit_date"
echo "release_tag_equals_latest_commit=$release_tag_equals_latest_commit"