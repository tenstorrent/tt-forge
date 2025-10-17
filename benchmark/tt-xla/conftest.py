# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest


def pytest_addoption(parser):
    """Adds a custom command-line option to pytest."""
    parser.addoption("--output", action="store", default=None, help="Path to save benchmark results as JSON.")


@pytest.fixture
def output(request):
    return request.config.getoption("--output")
