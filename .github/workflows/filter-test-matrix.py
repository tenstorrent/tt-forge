# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sys
from pathlib import Path


def flatten_matrix(data):
    """Flatten the matrix."""
    matrix = []
    for proj in data:
        test_defaults = proj.get("test-defaults", {})
        for test in proj.get("tests", []):
            merged_test = {**test_defaults, **test, "project": proj["project"]}

            runs_on = merged_test.get("runs-on", [])
            if isinstance(runs_on, list):
                matrix.extend({**merged_test, "runs-on": runner} for runner in runs_on)
            else:
                matrix.append(merged_test)

    return matrix


def filter_matrix(matrix, project_filter, name_filter=None):
    """Filter matrix based on project and name attributes."""

    name_filters = name_filter.split(",") if name_filter else []

    def should_include(item):
        if project_filter == "tt-forge" and item.get("project") not in ["tt-xla"]:
            return False
        if project_filter != "tt-forge" and item.get("project") != project_filter:
            return False

        name_filter_match = False
        for filter in name_filters:
            if filter.lower() in item.get("name").lower():
                name_filter_match = True

        if name_filter and not name_filter_match:
            return False

        return True

    return [item for item in matrix if should_include(item)]


def update_runners(matrix, sh_runner):
    """Update runner names based on shared runner flag."""
    runner_map = {"p150": "p150b"} if sh_runner else {"n150": "n150-perf"}

    for item in matrix:
        item["runs-on-original"] = item.get("runs-on")
        if item.get("runs-on") in runner_map:
            item["runs-on"] = runner_map[item["runs-on"]]

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Filter benchmark matrix")
    parser.add_argument("matrix_file", help="Path to benchmark matrix JSON file")
    parser.add_argument("project_filter", help="Project filter")
    parser.add_argument("--test-filter", help="Test name filter")
    parser.add_argument("--sh-runner", action="store_true", help="Use shared runners")

    args = parser.parse_args()
    matrix_skip = "false"

    try:
        with open(args.matrix_file) as f:
            data = json.load(f)

        matrix = flatten_matrix(data)
        filtered = filter_matrix(matrix, args.project_filter, args.test_filter)
        update_runners(filtered, args.sh_runner)

        if not filtered:
            print("Error: No matching tests found", file=sys.stderr)
            matrix_skip = "true"

        result = {"matrix": filtered, "matrix_skip": matrix_skip}
        print(json.dumps(result))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
