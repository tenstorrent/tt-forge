#!/usr/bin/env bash

# Keep only the latest $max_count releases 
max_count=30

all_releases=$(gh release list --exclude-drafts --json name,tagName,isPrerelease)
releases=$(echo "$all_releases" | jq -r '. |=map( select(.name | contains("Nightly")) | select(.tagName | contains("nightly"))| select(.isPrerelease == true))')
release_count=$(echo "$releases" | jq -r '.| length')

if [[ "$release_count" -lt "$max_count" ]]; then
    echo "No pruning needed. release_count: $release_count max_count: $max_count"
    exit 0
fi

echo "Pruning needed. release_count: $release_count max_count: $max_count"

# Iterate the array in reverse order since default order is from lastest to oldest releases
array_ptr=$(( release_count - 1 ))
stop_prune=$(( max_count - 1 ))

while [[ $array_ptr -ne $stop_prune ]]; do
    name=$(echo "$releases" | jq -r --arg array_ptr $array_ptr '.[$array_ptr|tonumber].name')
    tag=$(echo "$releases" | jq -r --arg array_ptr $array_ptr '.[$array_ptr|tonumber].tagName')
    gh release delete --yes --cleanup-tag "$tag"
    echo "Pruned nightly release name: $name tag: $tag"
    array_ptr=$((array_ptr-1))
done