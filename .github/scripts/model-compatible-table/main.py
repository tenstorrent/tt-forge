# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import lxml.etree
import pathlib
import ast
from typing import List, Dict, Union, Set
from tabulate import tabulate
import os
import copy
import re
import json


"""
Main script for hardware-compatible
"""

inf_regex: re.Pattern = re.compile(r"inf(?=\,)")

url_path_prefixes: Dict[str, str] = {
    "tt-xla": "https://github.com/tenstorrent/tt-xla/tree",
    "tt-forge-fe": "https://github.com/tenstorrent/tt-forge-fe/tree",
    "tt-torch": "https://github.com/tenstorrent/tt-torch/tree",
}

wormhole_cards: List[str] = ["n150", "n300", "wormhole"]
blackhole_cards: List[str] = ["p150", "p300", "blackhole"]


def get_property(case: lxml.etree.ElementTree, property_name: str) -> Dict[str, str]:
    """Get the property from the test case

    Args:
        case (lxml.etree.ElementTree): Test case
        property_name (str): Property name

    Returns:
        Dict[str, str]: Property value
    """
    a = case.xpath(f'properties/property[@name="{property_name}"]')
    try:
        value = a[0].get("value")
        if inf_regex.search(value):
            value = inf_regex.sub("0.0", value)
        if "{" in value:
            # string is a python dict not a json value due to single qoutes.
            return ast.literal_eval(value)
        return value
    except Exception as e:
        print(e)


def get_card_arch(file_path: pathlib.PosixPath) -> str:
    """Get the card arch from the file path

    Args:
        file_path (pathlib.PosixPath): File path

    Returns:
        str: Card arch
    """
    str_path = str(file_path).lower()
    if any(True for x in wormhole_cards if x in str_path):
        return "Wormhole"
    if any(True for x in blackhole_cards if x in str_path):
        return "Blackhole"
    return "N/A"


def get_test_file_name(case: lxml.etree.ElementTree) -> str:
    """Get the test file name from the test case

    Args:
        case (lxml.etree.ElementTree): Test case

    Returns:
        str: Test file name"""
    return case.get("classname").replace(".", "/") + ".py"


def parse_xml() -> Union[Dict[str, List[Dict[str, str]]], Set[str]]:
    """Parse the xml files and return a dictionary of model tests and a set of card archs

    Returns:
        Dict[str,List[Dict[str, str]]]: Dictionary of model tests
        Set[str]: Set of card archs
    """

    xml_root: str = os.environ.get("XML_ROOT")

    if not xml_root:
        print("XML_ROOT ENV is not definded")
        exit(1)

    card_archs: Set[str] = set()
    model_tests: Dict[str, List[Dict[str, str]]] = {}
    skip_process: Dict[str, bool] = {}

    xml_collection = pathlib.Path(xml_root).glob("**/**/*.xml")

    xml_parse_list: Dict[str, lxml.etree.ElementTree] = {x: lxml.etree.parse(x) for x in xml_collection}

    for k, v in xml_parse_list.items():
        print(f"processing file {k}")
        test_cases = v.xpath("/testsuites/testsuite/testcase")
        for case in test_cases:
            if not case.xpath("skipped"):
                tag_attrs: Dict = get_property(case, "tags")
                if not tag_attrs:
                    continue
                model_name = tag_attrs.get("model_name")
                if model_name is None:
                    continue
                frontend = get_property(case, "owner")
                card = get_card_arch(file_path=k)
                ## Don't process the same model twice
                if skip_process.get(f"{model_name}-{frontend}-{card}"):
                    continue

                ## execution_phase only needed for tt-forge-fe other repos use bringup_status
                status = (
                    tag_attrs.get("bringup_status")
                    if tag_attrs.get("bringup_status")
                    else tag_attrs.get("execution_phase")
                )
                if status != "PASSED":
                    continue

                temp_dict: Dict[str, str] = {
                    "model_name": model_name,
                    "card": card,
                    "frontend": frontend,
                    "status": status,
                    "file_path": get_test_file_name(case),
                }

                card_archs.add(card)
                skip_process[f"{model_name}-{frontend}-{card}"] = True

                if model_tests.get(model_name):
                    model_tests[model_name].append(temp_dict)
                    continue
                model_tests[tag_attrs.get("model_name")] = [temp_dict]

    return model_tests, card_archs


def create_table(model_tests: Dict[str, List[Dict[str, str]]], card_archs: Set[str]) -> None:
    """Create a table from the model tests and card archs

    Args:
        model_tests (Dict[str,List[Dict[str, str]]]): Dictionary of model tests
        card_archs (Set[str]): Set of card archs

    Returns:
        None
    """

    url_shas: str = os.environ.get("URL_SHAS")

    if not url_shas:
        print("URL_SHAS ENV is not definded")
        exit(1)

    url_shas = json.loads(url_shas)

    table_data: List[List[str]] = []

    header = ["frontend", "model_name"] + list(card_archs)
    table_data.append(header)

    for model_name, list_attrs in model_tests.items():
        for attrs in list_attrs:
            frontend = attrs.pop("frontend")
            url = f'{url_path_prefixes[frontend]}/{url_shas.get(frontend)}/{attrs.get("file_path")}'
            temp_row = [frontend, f"[{model_name}]({url})"]
            for card_arch in card_archs:
                if not attrs.get("card") == card_arch:
                    temp_row.append("N/A")
                    continue
                temp_row.append("✅" if attrs.get("status") == "PASSED" else "❌")
            table_data.append(temp_row)

    # Define the table format
    tablefmt = "github"

    # Create the table
    table = tabulate(table_data, headers="firstrow", tablefmt=tablefmt)

    print_builder = "\n\n###  Model Compatibility \n"
    print_builder += table

    file_output: str = os.environ.get("FILE_OUTPUT")

    if file_output:
        file1 = open(file_output, "w")
        file1.write(print_builder)
        file1.close()
    # Print the table
    print(print_builder)


if __name__ == "__main__":
    model_tests, card_archs = parse_xml()
    create_table(model_tests, card_archs)
