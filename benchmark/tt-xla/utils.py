# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os

# Third-party modules
from torch_xla.experimental import plugins
from loguru import logger
import requests
import time
import shutil
import urllib


def download_model(download_func, *args, num_retries=3, timeout=180, **kwargs):
    for _ in range(num_retries):
        try:
            return download_func(*args, **kwargs)
        except (
            requests.exceptions.HTTPError,
            urllib.error.HTTPError,
            requests.exceptions.ReadTimeout,
            urllib.error.URLError,
        ):
            logger.trace("HTTP error occurred. Retrying...")
            shutil.rmtree(os.path.expanduser("~") + "/.cache", ignore_errors=True)
            shutil.rmtree(os.path.expanduser("~") + "/.torch/models", ignore_errors=True)
            shutil.rmtree(
                os.path.expanduser("~") + "/.torchxrayvision/models_data",
                ignore_errors=True,
            )
            os.mkdir(os.path.expanduser("~") + "/.cache")
        time.sleep(timeout)

    logger.error("Failed to download the model after multiple retries.")
    assert False, "Failed to download the model after multiple retries."


# --------------------------------
# Plugin registration
# --------------------------------
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"


def find_plugin_path():
    """Find plugin path from wheel installation or local build."""
    # Try wheel installation first
    plugin_spec = importlib.util.find_spec("jax_plugins.pjrt_plugin_tt")
    if plugin_spec is not None and plugin_spec.origin is not None:
        wheel_plugin_path = os.path.join(os.path.dirname(plugin_spec.origin), "pjrt_plugin_tt.so")
        if os.path.exists(wheel_plugin_path):
            # Export path to metal for wheel installations
            tt_metal_path = os.path.join(os.path.dirname(wheel_plugin_path), "tt-mlir/install/tt-metal")
            os.environ["TT_METAL_HOME"] = str(tt_metal_path)
            return wheel_plugin_path

    # Fallback to local build
    build_plugin_path = os.path.join(os.getcwd(), "build/src/tt/pjrt_plugin_tt.so")
    if os.path.exists(build_plugin_path):
        return build_plugin_path

    raise FileNotFoundError(
        "Could not find TT PJRT plugin either from wheel installation or from " f"local build at {build_plugin_path}"
    )


class TTPjrtPlugin(plugins.DevicePlugin):
    def __init__(self, plugin_path: str):
        self._plugin_path = plugin_path
        super().__init__()

    def library_path(self):
        return self._plugin_path


plugin_path = find_plugin_path()
plugins.register_plugin("TT", TTPjrtPlugin(plugin_path))
