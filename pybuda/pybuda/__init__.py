# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import os

# Set home directory paths for pybuda and buda
def set_home_paths():
    import sys
    import pathlib
    from loguru import logger
    pybuda_path = pathlib.Path(__file__).parent.parent.resolve()

        # deployment path
    base_path = str(pybuda_path)
    out_path = "."

    if "PYBUDA_HOME" not in os.environ:
        os.environ["PYBUDA_HOME"] = str(pybuda_path)
    if "TVM_HOME" not in os.environ:
        os.environ["TVM_HOME"] = str(base_path) + "/tvm"
    if "BUDA_OUT" not in os.environ:
        os.environ["BUDA_OUT"] = str(out_path)
    if "LOGGER_FILE" in os.environ:
        sys.stdout = open(os.environ["LOGGER_FILE"], "w")
        logger.remove()
        logger.add(sys.stdout)

set_home_paths()

# eliminate tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from .module import Module, PyTorchModule, PyBudaModule, TFGraphDefModule, OnnxModule, JaxModule, TFLiteModule
from .torch_compile import compile_torch
from .compiled_graph_state import CompiledGraphState 
from .config import CompilerConfig, CompileDepth, set_configuration_options, set_epoch_break, set_chip_break, override_op_size, PerfTraceLevel, insert_buffering_nop, insert_nop, _internal_insert_fj_buffering_nop, override_dram_queue_placement, configure_mixed_precision
from .verify import VerifyConfig
from .pybudaglobal import pybuda_reset, set_device_pipeline, is_silicon, get_tenstorrent_device
from .parameter import Parameter
from .tensor import Tensor, SomeTensor, TensorShape
from .optimizers import SGD, Adam, AdamW, LAMB, LARS
from ._C import DataFormat, MathFidelity
from ._C import k_dim

import pybuda.op as op
import pybuda.transformers

import pybuda.typing
from .compile import pybuda_compile_torch, compile_main as compile

# Torch backend registration
# TODO: move this in a separate file / module.
from torch._dynamo.backends.registry import _BACKENDS
from torch._dynamo import register_backend

# register backend with torch:
# - enables backend to be shown when calling torch._dynamo.list_backends()
# - enables torch.compile(model, backend="<name_from_list_backends>"), where <name_from_list_backends> is "tt" in this case
if "tt" in _BACKENDS:
    del _BACKENDS["tt"]
register_backend(compile_torch, "tt")

