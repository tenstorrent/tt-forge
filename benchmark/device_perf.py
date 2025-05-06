#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Built-in modules
import os
import json
import argparse

# Third-party modules
import pandas as pd

DEVICE_FW_DURATION = "DEVICE FW DURATION [ns]"
DEVICE_KERNEL_DURATION = "DEVICE KERNEL DURATION [ns]"
NANO_SEC = 1e-9


def create_device_perf(device_perf_path, perf_report_path):
    """
    This function creates a device performance data file for testing purposes.
    It actually calls two functions that parse and write the device performance data.

    Parameters:
    ----------
    device_perf_path: str
        The path to the device performance data.

    perf_report_path: str
        The path to the JSON benchmark report.

    Returns:
    -------
    None
    """

    # Test the parse_device_perf function
    perf_data = parse_device_perf(device_perf_path)

    # Test the write_device_perf function
    write_device_perf(perf_report_path, perf_data, False)


def parse_device_perf(device_perf_path):
    """
    Parse the device performance data and prepare it for writing to the JSON benchmark report.

    Parameters:
    ----------
    device_perf_path: str
        The path to the device performance data.

    Returns:
    -------
    perf_data: dict
        A dictionary containing the device performance data.
    """

    # Read the device performance data
    df = pd.read_csv(device_perf_path)

    # Get total device fw duration and device kernel duration
    device_sum = df[[DEVICE_FW_DURATION, DEVICE_KERNEL_DURATION]].sum()
    device_fw_duration = device_sum[DEVICE_FW_DURATION] * NANO_SEC
    device_kernel_duration = device_sum[DEVICE_KERNEL_DURATION] * NANO_SEC

    perf_data = {"device_fw_duration": device_fw_duration, "device_kernel_duration": device_kernel_duration}

    return perf_data


def write_device_perf(perf_report_path, perf_data, write_new_path=False):
    """
    Write the device performance data to the JSON benchmark report.

    Parameters:
    ----------
    perf_report_path: str
        The path to the JSON benchmark report.
    perf_data: dict
        A dictionary containing the device performance data.

    Returns:
    -------
    None
    """

    # Read perf report JSON file
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
        print(f"Unexpected error: {e}")
        return

    # Upate the measurements
    # Add the device firmware duration
    perf_report["measurements"].append(
        {
            "iteration": perf_report["measurements"][0]["iteration"],
            "step_name": perf_report["measurements"][0]["step_name"],
            "step_warm_up_num_iterations": perf_report["measurements"][0]["step_warm_up_num_iterations"],
            "measurement_name": "device_fw_duration",
            "value": perf_data["device_fw_duration"],
            "target": perf_report["measurements"][0]["target"],
            "device_power": perf_report["measurements"][0]["device_power"],
            "device_temperature": perf_report["measurements"][0]["device_temperature"],
        }
    )
    # Add the device kernel duration
    perf_report["measurements"].append(
        {
            "iteration": perf_report["measurements"][0]["iteration"],
            "step_name": perf_report["measurements"][0]["step_name"],
            "step_warm_up_num_iterations": perf_report["measurements"][0]["step_warm_up_num_iterations"],
            "measurement_name": "device_kernel_duration",
            "value": perf_data["device_kernel_duration"],
            "target": perf_report["measurements"][0]["target"],
            "device_power": perf_report["measurements"][0]["device_power"],
            "device_temperature": perf_report["measurements"][0]["device_temperature"],
        }
    )

    if write_new_path:
        perf_report_path_out = perf_report_path.replace(".json", "_out.json")
    else:
        perf_report_path_out = perf_report_path

    # Save the results to the performance report file
    with open(perf_report_path_out, "w") as file:
        json.dump(perf_report, file)


def main():
    """
    The main function that creates the device performance data.

    Parameters:
    ----------
    None

    Returns:
    -------
    None

    Example:
    -------
    After running create_ttir.py, and running ttrt with the output file, run the following command:
        python ./benchmark/device_perf.py ./benchmark/test_data/device_perf/ops_perf_results.csv ./benchmark/test_data/device_perf/perf_report.json

    This command will parse the device performance data and write it to the JSON benchmark report.
    """

    if len(sys.argv) != 3:
        print("Error: No arguments provided.")
        exit(1)
    create_device_perf(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
