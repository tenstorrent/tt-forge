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


def create_ttir(ttir_path):
    """
    Create a TTIR file from the given path. TTIR is a JSON file that contains the model's information.

    Parameters:
    ----------
    ttir_path: str
        The path to the TTIR file.

    Returns:
    -------
    None

    """

    # Read the TTIR the JSON file
    try:
        with open(ttir_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: TTIR file '{ttir_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: TTIR file '{ttir_path}' contains invalid JSON.")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    # Make string from the JSON data
    # This JSON has should have the following structure:
    #   {
    #       'content': 'string'
    #       'module': 'string'
    #   }

    # Content is actually what we want to write to the TTIR file, module is the name of the module
    # Content is a string separated by newlines, we will create a list of strings from it, and modify it
    content = data["content"].split("\n")

    # The first line of the content is system descriptor, we don't need it
    # The second line is the definition of the module with attrubutes, we need to empty the attributes field
    attr_definition = "attributes {tt.system_desc = #system_desc}"
    attr_empty = "attributes {}"
    content[2] = content[2].replace(attr_definition, attr_empty)
    content.pop(1)

    ttir_path_out = ttir_path.replace(".mlir", "_out.mlir")

    for item in content:
        print(item)

    # Write the modified content to the TTIR file
    with open(ttir_path_out, "w") as file:
        file.write("\n".join(content))


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


def read_args():
    """
    Read the arguments from the command line.

    Parameters:
    ----------
    None

    Returns:
    -------
    parsed_args: dict
        The parsed arguments from the command line.
    """

    parser = argparse.ArgumentParser(description="Get device perf for benchmark end-to-end tests.")

    parser.add_argument("-ct", "--create-ttir", help="Create TTIR file from the given path.")

    parser.add_argument("-cdp", "--create-device-perf", nargs=2, help="Create device performance data.")

    args = parser.parse_args()

    if not args.create_ttir and not args.create_device_perf:
        parser.error("\n\nNo arguments provided.\n\n")
        print(parser.print_help())
        exit(1)

    if args.create_ttir and args.create_device_perf:
        parser.error("\n\nBoth arguments cannot be provided.\n\n")
        print(parser.print_help())
        exit(1)

    return args


def main():
    """
    The main function that creates the device performance data and the TTIR file.

    Parameters:
    ----------
    None

    Returns:
    -------
    None

    Example:
    -------
    From the root directory, run the following command:
        python ./forge/test/benchmark/device_perf.py -ct ./forge/test/benchmark/test_data/device_perf/ttir.mlir

    It will create a TTIR file from the given path.
    And put the output file in the same directory.
    Name of the output file will be ttir_out.mlir.

    When we run ttrt on the output file, we will get .csv file with device performance data.

    Now, run the following command:
        python ./forge/test/benchmark/device_perf.py -cdp ./forge/test/benchmark/test_data/device_perf/ops_perf_results.csv ./forge/test/benchmark/test_data/device_perf/perf_report.json

    This command will parse the device performance data and write it to the JSON benchmark report.

    Note:
        Inside the benchmark folder, you can find the test_data/device_per folder that contains the data we are using in the example.
    """

    # Read the arguments
    args = read_args()

    if args.create_ttir:
        ttir_path = args.create_ttir
        create_ttir(ttir_path)
    elif args.create_device_perf:
        device_perf_path = args.create_device_perf[0]
        perf_report_path = args.create_device_perf[1]
        create_device_perf(device_perf_path, perf_report_path)


if __name__ == "__main__":
    main()
