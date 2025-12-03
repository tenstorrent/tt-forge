# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Script to sum device performance metrics across multiple modules of a single model and write to benchmark report JSON.

Usage:
    python benchmark/device_perf.py <ttrt-artifacts-dir> <perf-report-json>

Example:
    python benchmark/device_perf.py ttrt-artifacts benchmark_report.json
"""

import os
import sys
import json
import pandas as pd

DEVICE_FW_DURATION = "DEVICE FW DURATION [ns]"
DEVICE_KERNEL_DURATION = "DEVICE KERNEL DURATION [ns]"
NANO_SEC = 1e-9
PERF_CSV_NAME = "ops_perf_results_minus_const_eval_and_input_layout_conversions.csv"


def find_perf_csv_files(artifacts_dir):
    """
    Find all performance CSV files in the ttrt-artifacts directory structure.

    Args:
        artifacts_dir: Path to ttrt-artifacts directory

    Returns:
        List of paths to performance CSV files
    """
    csv_files = []

    # Each binary creates a subdirectory in ttrt-artifacts
    # We only want the perf CSV from each binary's subdirectory
    for entry in os.listdir(artifacts_dir):
        entry_path = os.path.join(artifacts_dir, entry)
        if os.path.isdir(entry_path):
            csv_path = os.path.join(entry_path, "perf", PERF_CSV_NAME)
            csv_files.append(csv_path)

    return csv_files


def sum_device_perf(csv_files):
    """
    Sum device performance metrics across multiple CSV files.

    Args:
        csv_files: List of paths to CSV files

    Returns:
        Dictionary with summed device_fw_duration and device_kernel_duration
    """
    total_fw_duration = 0.0
    total_kernel_duration = 0.0

    print(f"Processing {len(csv_files)} device performance CSV file(s)")

    for csv_path in csv_files:
        print(f"  Processing: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            device_sum = df[[DEVICE_FW_DURATION, DEVICE_KERNEL_DURATION]].sum()
            fw_duration = device_sum[DEVICE_FW_DURATION] * NANO_SEC
            kernel_duration = device_sum[DEVICE_KERNEL_DURATION] * NANO_SEC

            total_fw_duration += fw_duration
            total_kernel_duration += kernel_duration

            print(f"    FW: {fw_duration:.6f}s, Kernel: {kernel_duration:.6f}s")
        except Exception as e:
            print(f"  Warning: Error processing {csv_path}: {e}")
            continue

    print(f"\nTotal summed performance:")
    print(f"  FW Duration: {total_fw_duration:.6f}s")
    print(f"  Kernel Duration: {total_kernel_duration:.6f}s")

    return {"device_fw_duration": total_fw_duration, "device_kernel_duration": total_kernel_duration}


def write_to_perf_report(perf_report_path, perf_data):
    """
    Write device performance data to the benchmark report JSON.

    Args:
        perf_report_path: Path to the JSON benchmark report
        perf_data: Dictionary with device_fw_duration and device_kernel_duration
    """
    try:
        with open(perf_report_path, "r") as f:
            perf_report = json.load(f)
    except FileNotFoundError:
        print(f"Error: Performance report file '{perf_report_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Performance report file '{perf_report_path}' contains invalid JSON.")
        sys.exit(1)

    # Get template from first measurement
    template = perf_report["measurements"][0]

    # Add device firmware duration measurement
    perf_report["measurements"].append(
        {
            "iteration": template["iteration"],
            "step_name": template["step_name"],
            "step_warm_up_num_iterations": template["step_warm_up_num_iterations"],
            "measurement_name": "device_fw_duration",
            "value": perf_data["device_fw_duration"],
            "target": template["target"],
            "device_power": template["device_power"],
            "device_temperature": template["device_temperature"],
        }
    )

    # Add device kernel duration measurement
    perf_report["measurements"].append(
        {
            "iteration": template["iteration"],
            "step_name": template["step_name"],
            "step_warm_up_num_iterations": template["step_warm_up_num_iterations"],
            "measurement_name": "device_kernel_duration",
            "value": perf_data["device_kernel_duration"],
            "target": template["target"],
            "device_power": template["device_power"],
            "device_temperature": template["device_temperature"],
        }
    )

    # Save updated report
    with open(perf_report_path, "w") as f:
        json.dump(perf_report, f)

    print(f"\nWritten device perf data to {perf_report_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python benchmark/device_perf.py <ttrt-artifacts-dir> <perf-report-json>")
        print("Example: python benchmark/device_perf.py ttrt-artifacts benchmark_report.json")
        sys.exit(1)

    artifacts_dir = sys.argv[1]
    perf_report_path = sys.argv[2]

    # Find all perf CSV files
    csv_files = find_perf_csv_files(artifacts_dir)

    if not csv_files:
        print(f"Error: No performance CSV files found in {artifacts_dir}")
        sys.exit(1)

    # Sum the performance metrics
    perf_data = sum_device_perf(csv_files)

    # Write to benchmark report
    write_to_perf_report(perf_report_path, perf_data)


if __name__ == "__main__":
    main()
