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

    def should_include(item):
        if project_filter == "tt-forge" and item.get("project") not in ["tt-xla", "tt-forge-fe"]:
            return False
        if project_filter != "tt-forge" and item.get("project") != project_filter:
            return False
        if name_filter and item.get("name") != name_filter:
            return False

        return True

    return [item for item in matrix if should_include(item)]


def update_runners(matrix, sh_runner):
    """Update runner names based on shared runner flag."""
    runner_map = {"p150": "p150b"} if sh_runner else {"n150": "n150-perf"}

    for item in matrix:
        if item.get("runs-on") in runner_map:
            item["runs-on"] = runner_map[item["runs-on"]]

    return matrix


def group_by_runs_on(matrix):
    """Group matrix items by runs-on value."""
    runs_on_groups = {}

    for item in matrix:
        runs_on = item.get("runs-on")
        if runs_on not in runs_on_groups:
            runs_on_groups[runs_on] = []
        runs_on_groups[runs_on].append(item)

    return list(runs_on_groups.values())


def main():
    parser = argparse.ArgumentParser(description="Filter benchmark matrix")
    parser.add_argument("matrix_file", help="Path to benchmark matrix JSON file")
    parser.add_argument("project_filter", help="Project filter")
    parser.add_argument("--test-filter", help="Test name filter")
    parser.add_argument("--sh-runner", action="store_true", help="Use shared runners")

    args = parser.parse_args()

    try:
        with open(args.matrix_file) as f:
            data = json.load(f)

        matrix = flatten_matrix(data)
        filtered = filter_matrix(matrix, args.project_filter, args.test_filter)
        update_runners(filtered, args.sh_runner)

        matrix_skip = not filtered

        result = {"matrix": filtered, "matrix_skip": str(matrix_skip).lower()}

        print(json.dumps(result))

        if matrix_skip:
            print("Error: No matching tests found", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
