---
name: ttnn
description: TTNN operations library reference for Tenstorrent hardware. Covers tensor APIs, ops catalog, model conversion from PyTorch, and memory/layout configuration.
---

## External Resources

- [TTNN Documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html)
- [TT-Metal Repository](https://github.com/tenstorrent/tt-metal)
- [TTNN API Reference](api.rst) -- full operation catalog (~400+ ops organized by category)
- [Tensor Reference](tensor.rst) -- shapes, layouts, memory configs, data types
- [Converting PyTorch Models to TTNN](converting_torch_model_to_ttnn.rst) -- step-by-step conversion guide
- [Multi-Device Programming](multi_device.md) -- MeshDevice, tensor parallelism, data parallelism, CCL ops
- [Tensor Sharding](tensor_sharding.md) -- height, width, and block sharding strategies

## Multi-Device

TTNN natively supports multi-chip execution via the MeshDevice abstraction. See [multi_device.md](multi_device.md) for full details.

```python
# Single device
device = ttnn.open_device(device_id=0, trace_region_size=100000000)

# Multi-device mesh (e.g., 4 chips in a row)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                     trace_region_size=100000000)

# Replicate a tensor to all devices
x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=mesh_device,
                     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))

# Shard a tensor across devices along a dimension (tensor parallelism)
w = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=mesh_device,
                     mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1))

# Read back sharded results by concatenating
result = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))

# Tensor parallel matmul pattern: column parallel + row parallel + all_reduce
col_out = ttnn.matmul(x_replicated, w_col_sharded)  # shard W along dim=1
row_out = ttnn.matmul(col_out, w_row_sharded)        # shard W along dim=0
reduced = ttnn.all_reduce(row_out)                    # sync across chips
```

## Sharding

Tensor sharding distributes data across cores for locality and reduced communication. See [tensor_sharding.md](tensor_sharding.md) for height, width, and block sharding strategies.

## Custom Program Sizes

Large fused kernels can exceed the default kernel config buffer limit (~69KB). The fix is to reduce `worker_l1_size`, which trades user L1 (for CBs/buffers) for more kernel config space.

```python
# Get the default worker L1 size
default_size = ttnn.device.get_max_worker_l1_unreserved_size()

# Subtract enough for your kernel's config buffer needs
# e.g., fused kernel is ~85KB, so give 88KB (90112 bytes) more config space
device = ttnn.open_device(device_id=0, worker_l1_size=default_size - 90112)
```

The tradeoff: slightly less L1 available for tile buffers. Start with a small reduction (e.g., 8192) and increase if you still hit the config buffer limit.

## Tracing

TTNN supports captured traces for eliminating host overhead in hot loops. See the [tt-enable-tracing](../tt-enable-tracing/) skill for setup and usage.

## Looking Up Op Documentation

Find the op name in [api.rst](api.rst), then fetch its full documentation:
```
curl https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/api/ttnn.<OP>.html
```
For example: `api/ttnn.conv2d.html`, `api/ttnn.matmul.html`, `api/ttnn.softmax.html`.

## Output Tensors and Scratch Memory

Most TTNN ops accept an `output_tensor` or `optional_output_tensor` parameter that lets you write the result into a pre-allocated tensor instead of allocating a new one. This is useful for:
- **Performance**: avoids repeated allocation/deallocation overhead
- **Tracing**: required for pre-allocating all tensors before trace capture
- **Scratch buffers**: reuse the same tensor across ops or loop iterations

```python
# Pre-allocate a scratch tensor
scratch = ttnn.zeros_like(x, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Reuse it across ops
ttnn.relu(x, output_tensor=scratch)
ttnn.add(scratch, bias, output_tensor=scratch)
```

Look up individual ops in the [API reference](api.rst) to check whether they support `output_tensor`.

## Overview

TTNN is the high-level operations library for Tenstorrent hardware. It provides a PyTorch-like API for tensor creation, manipulation, and computation on TT devices. TTNN ops run individually (one kernel launch per op call). For fusing multiple ops into a single kernel, use [TT-Lang](../tt-lang/SKILL.md).

## Key Concepts

- **Tensors** must be moved to device before computation: `ttnn.to_device(tensor, device)`
- **Layouts**: `ttnn.ROW_MAJOR_LAYOUT` or `ttnn.TILE_LAYOUT` (32x32 tiles, required for most compute ops)
- **Memory configs**: `ttnn.DRAM_MEMORY_CONFIG` (default, large) or `ttnn.L1_MEMORY_CONFIG` (fast, limited ~1.5MB/core)
- **Data types**: `ttnn.bfloat16` (standard), `ttnn.float32`, `ttnn.bfloat8_b`, `ttnn.uint32`

## Common Patterns

```python
import torch
import ttnn

device = ttnn.open_device(device_id=0)

# Torch -> TTNN
x_torch = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                     device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

# Compute
y = ttnn.relu(x)
y = ttnn.matmul(a, b)
y = ttnn.softmax(x, dim=-1)

# TTNN -> Torch
result = ttnn.to_torch(y)

ttnn.close_device(device)
```

