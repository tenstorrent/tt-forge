#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import sys


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

    with open(ttir_path, "r") as file:
        content = file.read()

    # Content is a string separated by newlines
    content = content.split("\n")

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


def main():
    """
    The main function that creates the TTIR file.

    Parameters:
    ----------
    None

    Returns:
    -------
    None

    Example:
    -------
    From the root directory, run the following command:
        python ./benchmark/create_ttir.py ./benchmark/test_data/device_perf/ttir.mlir

    It will create a TTIR file from the given path.
    And put the output file in the same directory.
    Name of the output file will be ttir_out.mlir.

    When we run ttrt on the output file, we will get .csv file with device performance data.
    """

    if len(sys.argv) != 2:
        print("Error: No arguments provided.")
        exit(1)
    create_ttir(sys.argv[1])


if __name__ == "__main__":
    main()
