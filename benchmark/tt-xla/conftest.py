# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest


def pytest_addoption(parser):
    """Adds a custom command-line option to pytest."""
    parser.addoption("--output", action="store", default=None, help="Path to save benchmark results as JSON.")
    parser.addoption(
        "--variant",
        action="store",
        default=None,
        help="Specify the LLaMA variant to test. If not set, LLaMA tests will be skipped.",
    )


@pytest.fixture
def output(request):
    return request.config.getoption("--output")


@pytest.fixture
def variant(request):
    return request.config.getoption("--variant")
