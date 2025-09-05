# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup

setup(
    name="tt-forge",
    version="${NEW_VERSION_TAG}",
    homepage="https://github.com/tenstorrent/tt-forge",
    install_requires=[
        "tt_forge_fe @${tt_forge_fe}",
        "tt_tvm @${tt_tvm}",
        "tt_torch @${tt_torch}",
        "pjrt_plugin_tt @${pjrt_plugin_tt}",
    ],
    python_requires=">=3.11",
    py_modules=[],
)
