#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Regexes
rc_release_tag_regex="^${MAJOR}\.${MINOR}\.\d+rc\d+$"
draft_rc_release_tag_regex="^draft\.${REPO_SHORT}\.${MAJOR}\.${MINOR}.\d+rc\d+$"
stable_release_tag_regex="^${MAJOR}\.${MINOR}\.\d+$"
draft_stable_release_tag_regex="^draft\.${REPO_SHORT}\.${MAJOR}\.${MINOR}\.\d+$"

# Get the latest branch commit
latest_branch_commit=$(git rev-parse origin/$BRANCH)

# Used for get the latest release tag from the release branch. Only return latest tag match.

rc_release_tag=$(git tag --merged origin/$BRANCH --sort=creatordate --sort=taggerdate | grep -oP $rc_release_tag_regex | tail -n 1 || true)
draft_rc_release_tag=$(git tag --merged origin/$BRANCH --sort=creatordate --sort=taggerdate | grep -oP $draft_rc_release_tag_regex | tail -n 1 || true)

stable_release_tag=$(git tag --merged origin/$BRANCH --sort=creatordate --sort=taggerdate | grep -oP $stable_release_tag_regex | tail -n 1 || true)
draft_stable_release_tag=$(git tag --merged origin/$BRANCH --sort=creatordate --sort=taggerdate | grep -oP $draft_stable_release_tag_regex | tail -n 1 || true)

# Perference order for current_release_tag: Draft Stable -> Stable -> Draft RC -> RC.
if [[ -n "$draft_stable_release_tag" ]]; then
    current_release_tag=$draft_stable_release_tag
    release_type="stable"
elif [[ -n "$stable_release_tag" ]]; then
    current_release_tag=$stable_release_tag
    release_type="stable"
elif [[ -n "$draft_rc_release_tag" ]]; then
    current_release_tag=$draft_rc_release_tag
    release_type="rc"
elif [[ -n "$rc_release_tag" ]]; then
    current_release_tag=$rc_release_tag
    release_type="rc"
fi

if [[ -z "$current_release_tag" ]]; then
    echo "Could not find release tag in branch $BRANCH"
    exit 1
fi

# Get the commit hash for the latest release tag
current_release_tag_commit=$(git rev-list -n 1 $current_release_tag)

# Check if the latest release tag is the same as the latest branch commit. Used to determine if the release branch is up to date.
release_tag_equals_latest_commit=false
if [[ "$current_release_tag_commit" == "$latest_branch_commit" ]]; then
    release_tag_equals_latest_commit=true
fi

# Only echo these variables for the action output
echo "branch=$branch"
echo "current_release_tag_commit=$current_release_tag_commit"
echo "current_release_tag=$current_release_tag"
echo "latest_branch_commit=$latest_branch_commit"
echo "release_tag_equals_latest_commit=$release_tag_equals_latest_commit"
echo "release_type=$release_type"
