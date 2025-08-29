# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import argparse


def add_config_fields_to_perf_report(perf_report_path, config_fields):
    """
    Add config fields to the config section of the performance report JSON file.

    Parameters:
    ----------
    perf_report_path: str
        The path to the JSON benchmark report file.
    config_fields: dict
        Dictionary of config fields to add to the report.

    Returns:
    -------
    None
    """

    try:
        with open(perf_report_path, "r") as file:
            perf_report = json.load(file)
    except FileNotFoundError:
        print(f"Error: Performance report file '{perf_report_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Performance report file '{perf_report_path}' contains invalid JSON.")
        return
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        return

    # Add config fields to the config section
    if "config" not in perf_report:
        perf_report["config"] = {}

    perf_report["config"].update(config_fields)

    # Write back to the file
    try:
        with open(perf_report_path, "w") as file:
            json.dump(perf_report, file, indent=2)
        print(f"Successfully added config fields to {perf_report_path}: {list(config_fields.keys())}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Add config fields to performance report JSON")
    parser.add_argument("perf_report_path", help="Path to the performance report JSON file")
    parser.add_argument("--ttir-url", help="URL for the TTIR MLIR artifact")
    parser.add_argument("--ttnn-url", help="URL for the TTNN MLIR artifact")
    parser.add_argument("--mlir-sha", help="MLIR commit SHA")
    parser.add_argument("--job-id-url", help="Job ID URL")
    parser.add_argument(
        "--config-field",
        action="append",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Add custom config field (can be used multiple times)",
    )

    args = parser.parse_args()

    config_fields = {}

    if args.ttir_url:
        config_fields["ttir_mlir_url"] = args.ttir_url
    if args.ttnn_url:
        config_fields["ttnn_mlir_url"] = args.ttnn_url
    if args.mlir_sha:
        config_fields["mlir_sha"] = args.mlir_sha
    if args.job_id_url:
        config_fields["job_id_url"] = args.job_id_url

    if args.config_field:
        for key, value in args.config_field:
            config_fields[key] = value

    if not config_fields:
        print("Error: No config fields provided")
        return

    add_config_fields_to_perf_report(args.perf_report_path, config_fields)


if __name__ == "__main__":
    main()
