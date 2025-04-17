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

    # Read the TTIR the JSON file
    try:
        with open(ttir_path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: TTIR file '{ttir_path}' not found.")
        sys.exit(2)
    except json.JSONDecodeError:
        print(f"Error: TTIR file '{ttir_path}' contains invalid JSON.")
        sys.exit(3)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(4)

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
    content[1] = content[1].replace(attr_definition, attr_empty)
    content = content[1:]  # Remove the first line

    # At the beginning of the content, we need to add two commands
    # The first command is to ttmlir-optimize the module
    # The second command is to ttmlir-translate the module
    ttmlir_optimize = (
        '// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o out.mlir %s'
    )
    ttmlir_translate = "// RUN: ttmlir-translate --ttnn-to-flatbuffer out.mlir > %t.ttnn"
    content.insert(0, ttmlir_translate)
    content.insert(0, ttmlir_optimize)

    ttir_path_out = ttir_path.replace(".mlir", "_out.mlir")

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
