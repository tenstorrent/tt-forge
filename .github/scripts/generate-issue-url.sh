#!/usr/bin/env bash

set -e

# Return a web url that filters issues between a desired date of when the release was fixed.
# source: gh issue list -R tenstorrent/tt-forge-fe --search "sort:created-desc label:bug state:open" --web
# test:
# issue_url_params="sort:created-desc label:bug state:open created:2025-01-01T00:00:00+00:00..2025-03-04T00:00:00+00:00" \
# owner="tenstorrent" \
# repo="tt-forge-fe" ./generate-issue-url.sh



