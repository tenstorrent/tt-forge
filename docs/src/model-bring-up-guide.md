# Bringing Up Models on Tenstorrent Hardware with TT-Forge

**Audience:** Developers running their first model on Tenstorrent silicon, and LLM coding agents porting HuggingFace models.

**Goal:** Walk you from zero to a working model on device, explain the concepts that are unique to this stack, and give you a playbook for debugging the common issues.

---

## 1\. How the Stack Fits Together

Before touching code, it helps to know what is compiling what:

```
Your Model (PyTorch / JAX / ONNX)
        │
        ▼
┌──────────────────────┐
│   Frontend Layer     │
│  ┌────────────────┐  │
│  │   TT-XLA       │  │  ← PyTorch (via torch_xla) and JAX models
│  │   (PJRT)       │  │     Produces StableHLO graphs
│  └────────────────┘  │
│  ┌────────────────┐  │
│  │ TT-Forge-ONNX  │  │  ← ONNX, TensorFlow, PaddlePaddle
│  │   (TVM-based)  │  │     Produces TTIR directly
│  └────────────────┘  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   TT-MLIR Compiler   │
│                      │
│  StableHLO → TTIR   │  ← Common intermediate representation
│  TTIR → Graph Passes │  ← Fusing, layout transforms, sharding
│  TTIR → TTNN-IR      │  ← Maps to TTNN library ops
│  TTIR → TTKernel-IR  │  ← Custom kernels (advanced)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   TT-Metalium        │
│   (TTNN + TTMetal)   │  ← Runtime: dispatches ops to hardware
└──────────┬───────────┘
           │
           ▼
   Wormhole / Blackhole
       (your card)
```

**Key repos:**

| Repo | What it does |
| :---- | :---- |
| [tt-forge](https://github.com/tenstorrent/tt-forge) | Central hub — demos, benchmarks, releases |
| [tt-xla](https://github.com/tenstorrent/tt-xla) | PJRT frontend for PyTorch and JAX |
| [tt-mlir](https://github.com/tenstorrent/tt-mlir) | MLIR compiler (TTIR, TTNN, TTKernel dialects) |
| [tt-metal](https://github.com/tenstorrent/tt-metal) | Low-level runtime and kernel library |
| [tt-forge-models](https://github.com/tenstorrent/tt-forge-models) | Community model test suite |

**Which frontend should I use?**

- **PyTorch or JAX** → TT-XLA (supports single-chip and multi-chip)  
- **ONNX, TensorFlow, PaddlePaddle** → TT-Forge-ONNX (single-chip only)  
- TT-Torch is deprecated; use TT-XLA for all new PyTorch work.

---

## 2\. Quick Start: Your First Model on Device

### 2.1 Install

```shell
# Install the PJRT plugin (includes tt-mlir + tt-metal)
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/

# Verify the device is visible
python -c "import jax; print(jax.devices('tt'))"
# → [TTDevice(id=0, arch=Wormhole_b0)]
```

### 2.2 Run a PyTorch Model (torch.compile path)

This is the simplest way to run an arbitrary PyTorch model:

```py
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer

xr.set_device_type("TT")
device = xm.xla_device()

# Load any HuggingFace model
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

model.eval()

# Compile for Tenstorrent
compiled_model = torch.compile(model, backend="tt")

# Run inference
inputs = tokenizer("The capital of France is", return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    outputs = compiled_model(input_ids)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)
    print(tokenizer.decode(next_token[0]))
```

### 2.3 Run a JAX Model

```py
import jax
import jax.numpy as jnp

# JAX auto-discovers the TT device via the PJRT plugin
@jax.jit
def forward(params, x):
    return jnp.dot(x, params["w"]) + params["b"]

params = {
    "w": jnp.ones((768, 768)),
    "b": jnp.zeros((768,))
}
x = jnp.ones((1, 768))
result = forward(params, x)
print(result.devices())  # → {TTDevice(id=0)}
```

---

## 3\. Concepts You Need to Know

### 3.1 Compilation is Lazy (and Cached)

When you call `torch.compile(model, backend="tt")` or `jax.jit`, the graph isn't compiled immediately. Compilation happens on the first forward pass and is cached. The first two iterations are always slow:

1. **Iteration 1:** Full compilation \+ weight transfer \+ kernel compilation  
2. **Iteration 2:** Runtime trace capture (see §5.3)  
3. **Iteration 3+:** Fast steady-state

**Always warm up with at least 3 dummy iterations before measuring performance.**

### 3.2 Tiling: Everything Is 32×32

Tenstorrent hardware operates on 32×32 tiles natively. The compiler handles padding automatically, but you'll get better performance when tensor dimensions are multiples of 32\. This matters most for:

- Hidden dimensions (e.g., `hidden_size`, `intermediate_size`)  
- Sequence lengths  
- Batch sizes (to a lesser extent)

If you see unexpected padding overhead, check your tensor shapes.

### 3.3 Memory Hierarchy: SRAM vs DRAM

Each Tensix core has **1.5 MB of local SRAM** — there is no shared cache. The compiler controls data placement:

- **Interleaved (DRAM):** Default. Tensors distributed across all DRAM banks. Safe but slower.  
- **Sharded (L1/SRAM):** Tensors distributed across Tensix cores' local SRAM. Fast but constrained by 1.5 MB per core.

The `optimization_level` compile option (§5.1) controls how aggressively the compiler moves data to SRAM.

### 3.4 Data Types

Tenstorrent hardware supports several precisions:

| Type | Size | When to use |
| :---- | :---- | :---- |
| `float32` | 32-bit | Default, widest support, slowest |
| `bfloat16` | 16-bit | **Recommended default.** 2× memory savings, minimal accuracy loss |
| `bfloat8_b` | 8-bit | Further speedup; verify accuracy on your workload |

**Always cast your model to bfloat16 before compilation:**

```py
model = model.to(dtype=torch.bfloat16)
```

For bfloat8\_b, enable via compile options (the model must already be bfloat16):

```py
torch_xla.set_custom_compile_options({
    "enable_bfp8_conversion": "true",
})
```

---

## 4\. How Ops Get Lowered: Fusing and Composite Ops

Understanding this pipeline is critical when a model hits unsupported ops.

### 4.1 The Compilation Pipeline

```
PyTorch nn.Module
    │
    ▼
torch.compile traces → FX Graph
    │
    ▼
Torch FX Fusion Passes              ← Pattern-matches multi-op sequences
    │                                   (e.g., LlamaRMSNorm → torch.rms_norm)
    ▼
Composite Op Wrapping                ← Wraps known ops with StableHLO markers
    │                                   (e.g., torch.rms_norm → tenstorrent.rms_norm)
    ▼
Export + Torch Decompositions        ← Composites survive decomposition
    │
    ▼
StableHLO                           ← Standard MLIR representation
    │
    ▼
TTIR Legalization                   ← TT-MLIR recognizes composites
    │
    ▼
Graph Passes (optimizer)            ← Layout transforms, op fusing, sharding
    │
    ▼
TTNN-IR → Hardware
```

### 4.2 Currently Supported Composite Ops

These ops are recognized and mapped to optimized TTNN implementations:

- `tenstorrent.gelu` / `tenstorrent.gelu_tanh`  
- `tenstorrent.rms_norm`  
- `tenstorrent.layer_norm`

If your model uses a custom implementation of these (e.g., HuggingFace's `LlamaRMSNorm`), the **fusion pass** will detect it and rewrite it to the standard `torch.nn.functional` version, which then gets wrapped as a composite. This means most HuggingFace models work without modification.

### 4.3 Scaled Dot-Product Attention (SDPA)

SDPA is handled through the composite/fusion system. When `torch.nn.functional.scaled_dot_product_attention` appears in the graph, it is preserved as a composite and lowered to an optimized TTNN implementation that takes advantage of the Tensix architecture's local SRAM.

**Best practices for attention:**

- Use `torch.nn.functional.scaled_dot_product_attention` rather than manual Q·K^T/√d\_k softmax V implementations  
- The compiler will handle KV cache management for autoregressive generation  
- For multi-head attention, standard HuggingFace implementations (GQA, MQA, MHA) are supported

### 4.4 What Happens When an Op Isn't Supported

If an op doesn't have a TTNN lowering, you'll see a compilation error. Common strategies:

1. **Check if a decomposition exists.** Many complex ops decompose into supported primitives automatically.  
2. **Rewrite to use a supported equivalent.** For example, replace a custom activation with `torch.nn.functional.gelu`.  
3. **File an issue** on [tt-forge](https://github.com/tenstorrent/tt-forge/issues) or [tt-mlir](https://github.com/tenstorrent/tt-mlir/issues) with the op name and a minimal repro.

---

## 5\. Performance Optimization

### 5.1 Optimization Levels

```py
torch_xla.set_custom_compile_options({
    "optimization_level": "2",  # 0, 1, or 2
})
```

| Level | What it does | Compile time | Runtime |
| :---- | :---- | :---- | :---- |
| 0 | No optimizer passes, all tensors in DRAM | Fastest | Slowest |
| 1 | Const-eval, conv weight preprocessing, fusion | Moderate | Good |
| 2 | Level 1 \+ maximize SRAM usage | Slowest | **Best** |

**Recommendation:** Start with level 0 to verify correctness, then move to level 2 for performance.

### 5.2 Data Format Optimization

Cast to bfloat16 **before** compilation, then optionally enable bfloat8\_b:

```py
model = model.to(dtype=torch.bfloat16)

torch_xla.set_custom_compile_options({
    "optimization_level": "2",
    "enable_bfp8_conversion": "true",  # Optional: 8-bit weights
})
```

### 5.3 Runtime Trace

Runtime trace eliminates host-device dispatch overhead by recording the command sequence and replaying it as a single command:

```py
import os
os.environ["TT_RUNTIME_TRACE_REGION_SIZE"] = "10000000"  # Set BEFORE importing torch_xla

torch_xla.set_custom_compile_options({
    "enable_trace": "true",
})
```

The trace is captured on the 2nd iteration and replayed on the 3rd+. This is why warmup requires 3 iterations.

### 5.4 Batch Size Tuning

Larger batch sizes generally improve throughput but increase latency. Start with powers of 2 (1, 2, 4, 8, 16, 32, 64\) and measure `samples/second`. Stop when throughput plateaus or you get an OOM error.

Note: smaller batches can sometimes outperform larger ones if they fit entirely in SRAM (with optimization level 2).

---

## 6\. Multi-Chip: Tensor Parallelism with SPMD

TT-XLA supports multi-chip execution through PyTorch/XLA's SPMD (Single Program Multiple Data) system. This lets you shard tensors across devices without writing explicit collective communication code — the compiler inserts the necessary all-gathers and reduce-scatters automatically.

### 6.1 Setting Up the Mesh

```py
import torch_xla.distributed.spmd as xs
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

xr.set_device_type("TT")

# Enable SPMD mode
xr.use_spmd()

device = xm.xla_device()

# Create a device mesh — shape depends on your hardware
# N300 has 2 chips, so mesh is (1, 2) for tensor parallelism
num_devices = xr.global_runtime_device_count()
mesh = xs.Mesh(
    list(range(num_devices)),
    (1, num_devices),
    ("batch", "model")
)
```

### 6.2 Sharding Model Weights (Tensor Parallelism)

The key idea: shard weight matrices along one dimension so each device holds a slice, then the compiler inserts collectives to produce the correct result.

```py
import torch_xla.distributed.spmd as xs

# Shard a weight tensor along the "model" mesh axis
# For a linear layer weight [out_features, in_features]:
#   - Column parallelism: shard dim 0 (output features)
#   - Row parallelism: shard dim 1 (input features)

# Column-parallel: each device gets out_features/N rows
xs.mark_sharding(linear.weight, mesh, ("model", None))

# Row-parallel: each device gets in_features/N columns
xs.mark_sharding(linear.weight, mesh, (None, "model"))

# Shard input activations along the batch dimension
xs.mark_sharding(input_tensor, mesh, ("batch", None))
```

### 6.3 A Complete Multi-Chip LLM Example

```py
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

xr.set_device_type("TT")
xr.use_spmd()

device = xm.xla_device()
num_devices = xr.global_runtime_device_count()
mesh = xs.Mesh(list(range(num_devices)), (1, num_devices), ("batch", "model"))

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.eval()

# Shard attention and MLP weights across devices
for layer in model.model.layers:
    # Attention: shard QKV projections column-parallel
    xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
    # Attention: shard output projection row-parallel
    xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    # MLP: shard gate/up projections column-parallel
    xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
    # MLP: shard down projection row-parallel
    xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

compiled_model = torch.compile(model, backend="tt")

# Move to device and run
inputs = tokenizer("Hello, world!", return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    outputs = compiled_model(input_ids)
```

### 6.4 Supported Hardware

For the current list of supported cards, systems, and chip configurations, see [tenstorrent.com/cards](https://tenstorrent.com/cards/) and the [hardware docs](https://docs.tenstorrent.com).

---

## 7\. Playbook: Porting a HuggingFace Model

This is the step-by-step process for bringing up a new HuggingFace model. It's designed to work whether you're a human or an LLM agent.

### Step 1: Check if it already runs

```shell
# Clone tt-forge and look for existing demos/tests
git clone https://github.com/tenstorrent/tt-forge.git
grep -r "YourModelName" tt-forge/demos/ tt-forge/benchmark/

# Clone tt-forge-models and search the community test suite
git clone https://github.com/tenstorrent/tt-forge-models.git
# Model directories use snake_case (e.g., llama/, gpt2/, qwen_2_5/)
ls tt-forge-models/ | grep -i "yourmodelname"
# Also search inside loader files for the HuggingFace model ID
grep -r "your-org/your-model" tt-forge-models/
```

### Step 2: Try the naive path first

```py
from transformers import AutoModel
import torch
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm

xr.set_device_type("TT")
device = xm.xla_device()

model = AutoModel.from_pretrained("your-model-id", torch_dtype=torch.bfloat16)
model.eval()
compiled = torch.compile(model, backend="tt")

dummy_input = torch.randn(1, 128, model.config.hidden_size, dtype=torch.bfloat16).to(device)

with torch.no_grad():
    output = compiled(dummy_input)
```

If this runs without error, you're mostly done — proceed to performance tuning (§5).

### Step 3: Handle compilation errors

Common errors and fixes:

| Error | Likely Cause | Fix |
| :---- | :---- | :---- |
| `Unsupported op: aten.xxx` | Op not lowered to TTNN | Check if a decomposition exists, or file an issue |
| `Shape mismatch` / `Tile alignment` | Tensor dim not tile-aligned | Pad inputs to multiples of 32 |
| `OOM` / `Insufficient L1` | Model too large for SRAM | Lower `optimization_level`, reduce batch size, or use multi-chip |
| `Timeout waiting for Ethernet` | Device hung | Run `tt-smi --reset 0` and retry |

### Step 4: Validate correctness

Compare device output against CPU reference:

```py
# Run on CPU
model_cpu = AutoModel.from_pretrained("your-model-id", torch_dtype=torch.bfloat16)
model_cpu.eval()
with torch.no_grad():
    cpu_output = model_cpu(cpu_input)

# Run on TT device
with torch.no_grad():
    tt_output = compiled(device_input).cpu()

# Check PCC (Pearson Correlation Coefficient)
from scipy.stats import pearsonr
pcc, _ = pearsonr(cpu_output.flatten().float().numpy(),
                  tt_output.flatten().float().numpy())
print(f"PCC: {pcc}")  # Should be > 0.99 for bfloat16
```

### Step 5: Write the test

Follow the conventions in [tt-forge-models](https://github.com/tenstorrent/tt-forge-models):

```py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.runtime as xr
from transformers import AutoModel, AutoTokenizer

from utils import ModelGroup, run_model_test  # repo-specific harness

xr.set_device_type("TT")

@pytest.mark.parametrize("model_id", [
    "your-org/your-model",
])
def test_your_model(model_id, request):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model.eval()

    inputs = tokenizer("Test input", return_tensors="pt")

    run_model_test(
        model=model,
        inputs=[inputs["input_ids"]],
        model_group=ModelGroup.VULCAN,
        request=request,
    )
```

### Step 6: Optimize performance

Apply the techniques from §5 in order:

1. Cast to bfloat16  
2. Set `optimization_level: 2`  
3. Enable runtime trace  
4. Tune batch size  
5. (Optional) Enable bfloat8\_b and verify accuracy

---

## 8\. Debugging Toolkit

### Environment Variables

| Variable | Value | Effect |
| :---- | :---- | :---- |
| `TTXLA_LOGGER_LEVEL` | `DEBUG` or `VERBOSE` | Detailed compilation logs |
| `TT_RUNTIME_TRACE_REGION_SIZE` | `10000000` | Enable runtime tracing (\~10 MB) |
| `TT_RUNTIME_ENABLE_PROGRAM_CACHE` | `1` | Cache compiled programs (default: on) |

### Device Management

```shell
# Check device status
tt-smi

# Reset a hung device
tt-smi --reset 0

# Reset all devices
tt-smi --reset
```

### Visualizing the Compiled Graph

Use `tt-explorer` to inspect the MLIR graph after compilation:

```shell
# See: https://docs.tenstorrent.com/tt-mlir/tt-explorer/tt-explorer.html
```

### Getting Help

- **File issues:** [tt-forge issues](https://github.com/tenstorrent/tt-forge/issues)  
- **Discord:** [Tenstorrent Discord](https://discord.gg/tenstorrent)  
- **Docs:** [docs.tenstorrent.com](https://docs.tenstorrent.com)

---

## 9\. Reference: Compiler Options

All options are set via `torch_xla.set_custom_compile_options({...})` or `torch.compile(model, backend="tt", options={...})`.

| Option | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| `optimization_level` | `"0"`, `"1"`, `"2"` | `"0"` | Compiler optimization aggressiveness |
| `enable_trace` | `"true"` / `"false"` | `"false"` | Enable runtime trace for dispatch optimization |
| `enable_bfp8_conversion` | `"true"` / `"false"` | `"false"` | Cast all ops to bfloat8\_b |
| `experimental_enable_weight_bfp8_conversion` | `"true"` / `"false"` | `"false"` | Cast only weights to bfloat8\_b |
| `tt_enable_torch_fx_fusion_pass` | `True` / `False` | `True` | Enable FX-level op fusion |
| `tt_enable_composite_ops` | `True` / `False` | `True` | Enable composite op wrapping |

---

## 10\. Notes for LLM Agents Porting Models

If you are an LLM (e.g., Claude Code) working on model bring-up, here are the key things to know:

1. **Start with `torch.compile(model, backend="tt")`** — this is the path of least resistance. Don't manually lower ops.  
     
2. **Always use bfloat16.** HuggingFace models default to float32; cast them before compilation.  
     
3. **The fusion system handles most HuggingFace patterns.** Custom RMSNorm, GeLU, and LayerNorm implementations are automatically fused to their `torch.nn.functional` equivalents.  
     
4. **If you hit an unsupported op**, check whether disabling specific model features (e.g., flash attention variants, rotary embedding implementations) resolves it. Often a simpler code path exists.  
     
5. **Test with `ModelGroup.VULCAN`** in tt-forge-models — this is the enum for community/bring-up models.  
     
6. **Pre-commit is mandatory.** Always run `pre-commit run --all-files` before committing.  
     
7. **SPDX headers are required** on all source files:

```py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
```

8. **Common model families known to work:** Llama, Phi, Qwen, Falcon, ResNet, ViT, MobileNet, EfficientNet, GPT-2, OPT. Check existing demos in `tt-forge/demos/tt-xla/` and benchmarks in `tt-forge/benchmark/tt-xla/` for reference implementations.  
     
9. **For multi-chip models**, use the SPMD sharding approach (§6). The pattern is always: column-parallel for QKV/gate/up projections, row-parallel for output/down projections.  
     
10. **Atomic commits.** If iterating on fixes, make each fix a separate commit with a descriptive message.

