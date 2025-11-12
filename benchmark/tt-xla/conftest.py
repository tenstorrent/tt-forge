# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import os
import pytest

# Add tt-forge directory to Python path so imports like 'from benchmark.utils' work
# when running pytest from any directory (e.g., from tt-xla repo)
# Path structure: tt-forge/benchmark/tt-xla/conftest.py
# We need: tt-forge/ in sys.path so 'benchmark' module can be imported
tt_forge_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if tt_forge_dir not in sys.path:
    sys.path.insert(0, tt_forge_dir)


# Valid values for parameters
VALID_DATA_FORMATS = {"bfloat16", "float32"}
VALID_TASKS = {"text-generation"}
VALID_BOOLEAN_VALUES = {"true", "false"}


def make_validator_boolean(option_name):
    """Create a boolean validator with the option name in error messages."""

    def validate(value):
        if value.lower() not in VALID_BOOLEAN_VALUES:
            raise pytest.UsageError(
                f"Invalid value for {option_name}: '{value}'. Must be one of: {', '.join(sorted(VALID_BOOLEAN_VALUES))}"
            )
        return value.lower() == "true"

    return validate


def make_validator_data_format(option_name):
    """Create a data format validator with the option name in error messages."""

    def validate(value):
        if value not in VALID_DATA_FORMATS:
            raise pytest.UsageError(
                f"Invalid value for {option_name}: '{value}'. Must be one of: {', '.join(sorted(VALID_DATA_FORMATS))}"
            )
        return value

    return validate


def make_validator_task(option_name):
    """Create a task validator with the option name in error messages."""

    def validate(value):
        if value not in VALID_TASKS:
            raise pytest.UsageError(
                f"Invalid value for {option_name}: '{value}'. Must be one of: {', '.join(sorted(VALID_TASKS))}"
            )
        return value

    return validate


def make_validator_positive_int(option_name):
    """Create a positive integer validator with the option name in error messages."""

    def validate(value):
        try:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError
            return int_value
        except (ValueError, TypeError):
            raise pytest.UsageError(f"Invalid value for {option_name}: '{value}'. Must be a positive integer (> 0).")

    return validate


def pytest_addoption(parser):
    """Adds a custom command-line option to pytest."""
    parser.addoption("--output", action="store", default=None, help="Path to save benchmark results as JSON.")
    parser.addoption(
        "--variant",
        action="store",
        default=None,
        help="Specify the model variant to test. If not set, tests will be skipped.",
    )
    # Optional configuration arguments
    parser.addoption(
        "--optimizer-enabled",
        action="store",
        default=None,
        type=make_validator_boolean("--optimizer-enabled"),
        help="Enable optimizer (true/false). Overrides config value.",
    )
    parser.addoption(
        "--memory-layout-analysis",
        action="store",
        default=None,
        type=make_validator_boolean("--memory-layout-analysis"),
        help="Enable memory layout analysis (true/false). Overrides config value.",
    )
    parser.addoption(
        "--trace-enabled",
        action="store",
        default=None,
        type=make_validator_boolean("--trace-enabled"),
        help="Enable trace (true/false). Overrides config value.",
    )
    parser.addoption(
        "--batch-size",
        action="store",
        default=None,
        type=make_validator_positive_int("--batch-size"),
        help="Batch size (positive integer). Overrides config value.",
    )
    parser.addoption(
        "--loop-count",
        action="store",
        default=None,
        type=make_validator_positive_int("--loop-count"),
        help="Number of benchmark iterations (positive integer). Overrides config value.",
    )
    parser.addoption(
        "--input-sequence-length",
        action="store",
        default=None,
        type=make_validator_positive_int("--input-sequence-length"),
        help="Input sequence length (positive integer). Overrides config value.",
    )
    parser.addoption(
        "--data-format",
        action="store",
        default=None,
        type=make_validator_data_format("--data-format"),
        help=f"Data format. Valid values: {', '.join(sorted(VALID_DATA_FORMATS))}. Overrides config value.",
    )
    parser.addoption(
        "--measure-cpu",
        action="store",
        default=None,
        type=make_validator_boolean("--measure-cpu"),
        help="Measure CPU FPS (true/false). Overrides config value.",
    )
    parser.addoption(
        "--task",
        action="store",
        default=None,
        type=make_validator_task("--task"),
        help=f"Task type. Valid values: {', '.join(sorted(VALID_TASKS))}. Overrides config value.",
    )
    parser.addoption(
        "--experimental-compile",
        action="store",
        default=None,
        type=make_validator_boolean("--experimental-compile"),
        help="Enable experimental compile flag (true/false). Overrides config value.",
    )


@pytest.fixture
def output(request):
    return request.config.getoption("--output")


@pytest.fixture
def variant(request):
    return request.config.getoption("--variant")


@pytest.fixture
def optimizer_enabled(request):
    return request.config.getoption("--optimizer-enabled")


@pytest.fixture
def memory_layout_analysis(request):
    return request.config.getoption("--memory-layout-analysis")


@pytest.fixture
def trace_enabled(request):
    return request.config.getoption("--trace-enabled")


@pytest.fixture
def batch_size(request):
    return request.config.getoption("--batch-size")


@pytest.fixture
def loop_count(request):
    return request.config.getoption("--loop-count")


@pytest.fixture
def input_sequence_length(request):
    return request.config.getoption("--input-sequence-length")


@pytest.fixture
def data_format(request):
    return request.config.getoption("--data-format")


@pytest.fixture
def measure_cpu(request):
    return request.config.getoption("--measure-cpu")


@pytest.fixture
def task(request):
    return request.config.getoption("--task")


@pytest.fixture
def experimental_compile(request):
    return request.config.getoption("--experimental-compile")
