#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib.util
import json
import os
import sys
from typing import Dict, Any


def load_module(module_path: str, module_name: str):
    """
    Dynamically load a Python module from a file path.

    Parameters:
    ----------
        module_path: The path to the module file
        module_name: The name to assign to the loaded module

    Returns:
    -------
        module: The loaded module

    Raises:
    ------
        ImportError: If the module cannot be loaded
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_common_module(project_dir: str):
    """
    Attempt to load the common.py module from the project directory.

    Parameters:
    ----------
        project_dir: The directory of the project

    Returns:
    -------
        common_module: The loaded common module or None if it doesn't exist
    """
    common_file = os.path.join(project_dir, "common.py")
    if not os.path.exists(common_file):
        return None

    module_name = f"{os.path.basename(project_dir)}.common"
    return load_module(common_file, module_name)


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a benchmark function from the specified project and test.

    Parameters:
    ----------
        config: Configuration dictionary to pass to the benchmark function

    Returns:
    -------
        The result dictionary from the benchmark function
    """

    # Extract project and test from the config
    project = config["project"]
    test = config["model"]

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Add the project directory to sys.path so imports within the module work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, project))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Construct the path to the test file
    test_file = os.path.join(project_dir, f"{test}.py")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Try to load the common module if it exists
    common_module = load_common_module(project_dir)

    # Apply pre_test function if it exists in common module
    if common_module and hasattr(common_module, "pre_test"):
        print(f"Running pre_test function from {project}/common.py")
        config = common_module.pre_test(config, test)

    # Load the test module
    module_name = f"{project}.{test}"
    module = load_module(test_file, module_name)

    # Check if the module has a benchmark function
    if not hasattr(module, "benchmark"):
        raise AttributeError(f"No benchmark function found in {test_file}")

    # Run the benchmark function with the provided config
    results = module.benchmark(config)

    # Apply post_test function if it exists in common module
    if common_module and hasattr(common_module, "post_test"):
        print(f"Running post_test function from {project}/common.py")
        results = common_module.post_test(results, config, test)

    return results


def save_results(config: Dict[str, Any], results: Dict[str, Any], project: str, model: str, ttnn_perf_metrics_output_file: str):
    """
    Save the benchmark results to a JSON file.

    Parameters:
    ----------
    config: dict
        The configuration used for the benchmark.
    results: dict
        The results of the benchmark.
    project: str
        The name of the project.
    model: str
        The name of the model.

    Returns:
    -------
    None
    """
    if config["output"]:
        output_file = config["output"]
    else:
        output_file = os.path.join(f"{project}_{model}.json")
    
    # If the perf_metrics report file exists, load existing results and append to config, the only appendable field for superset
    print(f"Looking for perf report file at: {ttnn_perf_metrics_output_file}")
    if os.path.exists(ttnn_perf_metrics_output_file):
        with open(ttnn_perf_metrics_output_file, "r") as f:
            perf_metrics_data = json.load(f)
        if "summary" in perf_metrics_data and isinstance(perf_metrics_data["summary"], dict):
            results["config"]["ttnn_total_ops"] = perf_metrics_data["summary"]["total_ops"]
            results["config"]["ttnn_total_shardable_ops"] = perf_metrics_data["summary"]["total_shardable_ops"]
            results["config"]["ttnn_effectively_sharded_ops"] = perf_metrics_data["summary"]["effectively_sharded_ops"]
            results["config"]["ttnn_effectively_sharded_percentage"] = perf_metrics_data["summary"]["effectively_sharded_percentage"]
            results["config"]["ttnn_system_memory_ops"] = perf_metrics_data["summary"]["system_memory_ops"]

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark completed successfully. Results saved to {output_file}")


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

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run benchmark functions from projects")

    parser.add_argument("-p", "--project", help="The project directory containing the model file")
    parser.add_argument(
        "-m", "--model", help="Model to benchmark (i.e. bert, mnist_linear). The test file name (without .py extension)"
    )
    parser.add_argument(
        "-c", "--config", default=None, help="Model configuration to benchmark (i.e. tiny, base, large)."
    )
    parser.add_argument("-t", "--training", action="store_true", default=False, help="Benchmark training.")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=1, help="Batch size, number of samples to process at once."
    )
    parser.add_argument("-lp", "--loop_count", type=int, default=1, help="Number of times to run the benchmark.")
    parser.add_argument(
        "-isz",
        "--input_size",
        type=int,
        default=None,
        help="Input size, size of the input sample. If the model gives opportunity to change input size.",
    )
    parser.add_argument(
        "-hs",
        "--hidden_size",
        type=int,
        default=None,
        help="Hidden size, size of the hidden layer. `If the model gives opportunity to change hidden size.",
    )
    parser.add_argument(
        "-isl",
        "--input_sequence_length",
        type=int,
        default=0,
        help="Input sequence length for text/sequence models. Default is 0.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output json file to write results to, optionally. If file already exists, results will be appended.",
    )
    parser.add_argument(
        "-ts", "--task", default="na", help="Task to benchmark, i.e. classification, segmentation, etc. Default is 'na'"
    )
    parser.add_argument("-df", "--data_format", default="float32", help="Data format i.e. float32, bfloat16, etc.")
    parser.add_argument("-r", "--run_origin", default="tt-forge", help="Repo where the benchmark is run from.")
    parser.add_argument("-mc", "--measure_cpu", action="store_true", default=False, help="Measure CPU FPS.")
    args = parser.parse_args()

    config = {}

    if not args.project:
        print("\nProject directory must be specified.\n\n")
        print(parser.print_help())
        exit(1)

    if not args.model:
        print("\nModel must be specified.\n\n")
        print(parser.print_help())
        exit(1)

    config["project"] = args.project
    config["model"] = args.model
    config["config"] = args.config
    config["training"] = args.training
    config["loop_count"] = args.loop_count
    config["batch_size"] = args.batch_size
    config["input_size"] = args.input_size
    config["hidden_size"] = args.hidden_size
    config["input_sequence_length"] = args.input_sequence_length
    config["output"] = args.output
    config["task"] = args.task
    config["data_format"] = args.data_format
    config["run_origin"] = args.run_origin
    config["measure_cpu"] = args.measure_cpu

    return config


def main():
    """
    Main function for running the benchmark tests.

    Parameters:
    ----------
    None

    Returns:
    -------
    None
    """

    # Read the arguments from the command line.
    config = read_args()

    config["ttnn_perf_metrics_output_file"] = config["model"] + "_perf_metrics.json"

    # Run the benchmark
    results = run_benchmark(config)
    results["project"] = config["run_origin"] + "/" + config["project"]
    results["model_rawname"] = config["model"]

    # Save the results
    save_results(config, results, config["project"], config["model"], config["ttnn_perf_metrics_output_file"])

    print("Done.")


if __name__ == "__main__":
    main()
