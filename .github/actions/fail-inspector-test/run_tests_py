# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys

if __name__ == "__main__":
    with open(sys.argv[1], "r") as fd:
        test_list = [line.strip() for line in fd.readlines()]
    print(f"Running tests: {test_list}")
    sys.exit(pytest.main(test_list))
