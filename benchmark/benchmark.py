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
    """Dynamically load a Python module from a file path."""
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
    Returns None if the module doesn't exist.
    """
    common_file = os.path.join(project_dir, "common.py")
    if not os.path.exists(common_file):
        return None

    module_name = f"{os.path.basename(project_dir)}.common"
    return load_module(common_file, module_name)


def run_benchmark(project: str, test: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a benchmark function from the specified project and test.

    Args:
        project: The name of the project directory
        test: The name of the test file (without .py extension)
        config: Configuration dictionary to pass to the benchmark function

    Returns:
        The result dictionary from the benchmark function
    """
    # Add the project directory to sys.path so imports within the module work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, project))
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

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


def save_results(config: Dict[str, Any], results: Dict[str, Any], project: str, model: str):
    if config["output"]:
        output_file = config["output"]
    else:
        output_file = os.path.join(f"{project}_{model}.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark completed successfully. Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark functions from projects")
    parser.add_argument("project", help="The project directory containing the model file")
    parser.add_argument(
        "model", help="Model to benchmark (i.e. bert, mnist_linear). The test file name (without .py extension)"
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
        "-o",
        "--output",
        default=None,
        help="Output json file to write results to, optionally. If file already exists, results will be appended.",
    )

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

    config["config"] = args.config
    config["training"] = args.training
    config["loop_count"] = args.loop_count
    config["batch_size"] = args.batch_size
    config["input_size"] = args.input_size
    config["hidden_size"] = args.hidden_size
    config["output"] = args.output

    try:
        # Run the benchmark
        results = run_benchmark(args.project, args.model, config)
        results["project"] = "tt-" + args.project
        results["model_rawname"] = args.model

        # Save the results
        save_results(config, results, args.project, args.model)

    except Exception as e:
        print(f"Error running benchmark: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
