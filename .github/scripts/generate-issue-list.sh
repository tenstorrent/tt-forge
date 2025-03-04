#!/usr/bin/env bash

set -e

# Return a web url that filters issues between a desired date of when the release was fixed.
# test: gh issue list -R tenstorrent/tt-forge-fe --search "sort:created-desc label:bug state:open" --web

repo="tenstorrent/tt-forge-fe"
#:2017-01-01T01:00:00+07:00..2017-03-01T15:30:15+07:00
url_parms="sort:created-desc label:bug state:open created:2025-01-01T00:00:00+00:00..2025-03-04T00:00:00+00:00"
base_url="https://github.com/$repo/issues?q="

encoded_params=$(echo -n $url_parms|jq -sRr @uri)
echo "$base_url$encoded_params"

echo "https://github.com/tenstorrent/tt-forge-fe/issues?q=sort%3Acreated-desc%20label%3Abug%20state%3Aopen%20created%3A2025-01-01T00%3A00%3A00%2B00%3A00..2025-03-04T00%3A00%3A00%2B00%3A00"