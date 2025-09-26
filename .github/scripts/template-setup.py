# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup

setup(
    name="tt-forge",
    version="${NEW_VERSION_TAG}",
    homepage="https://github.com/tenstorrent/tt-forge",
    install_requires=[
        "tt-forge-fe @https://pypi.eng.aws.tenstorrent.com/tt-forge-fe/tt_forge_fe-${TT_FORGE_FE_TAG}-cp311-cp311-linux_x86_64.whl",
        "tt-tvm @https://pypi.eng.aws.tenstorrent.com/tt-tvm/tt_tvm-${TT_FORGE_FE_TAG}-cp311-cp311-linux_x86_64.whl",
        "pjrt-plugin-tt @https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-${PJRT_PLUGIN_TT_TAG}-cp311-cp311-linux_x86_64.whl",
    ],
    python_requires=">=3.11",
    py_modules=[],
    has_ext_modules=lambda: True,  # Force platform-specific wheel
)
