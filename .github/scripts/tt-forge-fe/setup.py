# SPDX-License-Identifier: Apache-2.0

from setuptools import setup


setup(
    name="forge-pypi",
    version="0.1",
    author="Aleks Knezevic",
    author_email="aknezevic@tenstorrent.com",
    license="Apache-2.0",
    homepage="https://github.com/tenstorrent/tt-forge-fe",
    description="TT FrontEnd",
    install_requires=[
        "forge @https://github.com/tenstorrent/tt-forge/releases/download/nightly-0.1.0.dev20250505060214/forge-0.1.0.dev20250505060214-cp310-cp310-linux_x86_64.whl",
        "tvm @https://github.com/tenstorrent/tt-forge/releases/download/nightly-0.1.0.dev20250505060214/tvm-0.1.0.dev20250505060214-cp310-cp310-linux_x86_64.whl",
    ],
)