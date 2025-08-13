# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import argparse


def add_mlir_urls_to_perf_report(perf_report_path, ttir_url, ttnn_url):
    """
    Add MLIR URLs to the config section of the performance report JSON file.

    Parameters:
    ----------
    perf_report_path: str
        The path to the JSON benchmark report file.
    ttir_url: str
        The URL for the TTIR MLIR artifact.
    ttnn_url: str
        The URL for the TTNN MLIR artifact.

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

    # Add MLIR URLs to the config section
    if "config" not in perf_report:
        perf_report["config"] = {}

    perf_report["config"]["ttir_mlir_url"] = ttir_url
    perf_report["config"]["ttnn_mlir_url"] = ttnn_url

    # Write back to the file
    try:
        with open(perf_report_path, "w") as file:
            json.dump(perf_report, file, indent=2)
        print(f"Successfully added MLIR URLs to {perf_report_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Add MLIR URLs to performance report JSON")
    parser.add_argument("perf_report_path", help="Path to the performance report JSON file")
    parser.add_argument("ttir_url", help="URL for the TTIR MLIR artifact")
    parser.add_argument("ttnn_url", help="URL for the TTNN MLIR artifact")

    args = parser.parse_args()

    add_mlir_urls_to_perf_report(args.perf_report_path, args.ttir_url, args.ttnn_url)


if __name__ == "__main__":
    main()
