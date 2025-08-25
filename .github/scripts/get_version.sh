#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to get the latest released version from GitHub repo to determine the next version to use.

set -e  # Exit on any error

# Function to get the latest release version from GitHub
get_latest_github_release() {
    local repo="${1:-}"

    echo "Fetching latest release for repository: $repo" >&2

    # Get the latest release tag
    latest_version=$(gh release view --repo "$repo" --json tagName -q .tagName 2>/dev/null || echo "")

    if [[ -z "$latest_version" ]]; then
        echo "Error: No releases found for repository $repo" >&2
        exit 1
    fi

    echo "Latest release found: $latest_version" >&2
    echo "$latest_version"
}

# Function to parse version string and extract MAJOR.MINOR.PATCH
parse_version() {
    local version="$1"

    # Extract MAJOR.MINOR.PATCH using regex
    if [[ $version =~ ^([0-9]+)\.([0-9]+)\.([0-9]+) ]]; then
        API_MAJOR="${BASH_REMATCH[1]}"
        API_MINOR="${BASH_REMATCH[2]}"
        # PATCH is set to 0 since bumps for that are handled in the get-release-branch workflow
        API_PATCH="0"
        echo "Parsed version: MAJOR=$API_MAJOR, MINOR=$API_MINOR, PATCH=$API_PATCH" >&2
    else
        echo "Error: Invalid version format: $version" >&2
        exit 1
    fi
}

# Function to bump the MINOR version
bump_minor_version() {
    MINOR=$((MINOR + 1))
    echo "Bumped MINOR version: MAJOR=$MAJOR, MINOR=$MINOR, PATCH=$PATCH" >&2
}

# Function to render the .rendered_version file
render_version_file() {
    local version_file=".rendered_version"

    echo "Updating $version_file..." >&2

    cat > "$version_file" << EOF
MAJOR=$MAJOR
MINOR=$MINOR
PATCH=$PATCH
VERSION="\$MAJOR.\$MINOR.\$PATCH"
EOF

    echo "Updated $version_file with MAJOR=$MAJOR, MINOR=$MINOR, PATCH=$PATCH" >&2
}

# Main execution
main() {
    local repo="$1"

    # Get latest release version from GitHub repo
    latest_version=$(get_latest_github_release "$repo")

    # Parse the version from API
    parse_version "$latest_version"

    # Compare MAJOR versions to determine if we need to use the MAJOR from the .version file or the API version.
    # Used when needed to bump to a new major version.
    source .version
    VERSION_FILE_MAJOR=$MAJOR

    if [[ $API_MAJOR -lt $VERSION_FILE_MAJOR ]]; then
        echo "API MAJOR ($API_MAJOR) > current MAJOR ($VERSION_FILE_MAJOR), using .version file" >&2
        MAJOR=$VERSION_FILE_MAJOR
        MINOR=0
        PATCH=0
    else
        echo "Using API version and bumping MINOR" >&2
        MAJOR=$API_MAJOR
        MINOR=$API_MINOR
        PATCH=$API_PATCH
        # Bump the MINOR version
        bump_minor_version
    fi

    # Create the .rendered_version file
    render_version_file

    # Output the final version
    VERSION="$MAJOR.$MINOR.$PATCH"
    echo "Created version: $VERSION"
}

# Run main function with all arguments
main "$@"
