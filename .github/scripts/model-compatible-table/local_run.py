# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import main
import os
from pathlib import Path

os.environ["XML_ROOT"] = str(Path(__file__).resolve().parent.joinpath("tests"))
os.environ[
    "URL_SHAS"
] = '{"tt-forge-fe": "0872c4955e50be6b38a15635e1ea2e00189d45fe", "tt-torch": "15c37e0fdf54e791555c0a1bda2cc8ba922d7a0f"}'


model_tests, card_names = main.parse_xml()
main.create_table(model_tests, card_names)
