---
name: tt-enable-tracing
description: TTNN trace capture and replay for eliminating dispatch overhead. Essential for real-time inference and multi-chip performance.
---

## External Resources

- [Advanced Performance Optimizations](advanced_perf_optimizations.md) -- trace APIs, multiple command queues, combining trace + multi-CQ, programming examples

## Overview

Trace capture records a sequence of TTNN operations once, then replays them without host dispatch overhead.

## Prerequisites

When opening the device, reserve space for the trace with `trace_region_size`:

```python
# Single device
device = ttnn.open_device(device_id=0, trace_region_size=100000000)

# Multi-device mesh
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, N_CHIPS),
                                     trace_region_size=100000000)
```

## Rules

The trace replays the exact recorded command sequence. Everything inside the trace MUST be pure device work:

1. **You MUST remove all host-to-device and device-to-host transfers** from the traced region. All `ttnn.from_torch`, `ttnn.to_torch`, `ttnn.copy_host_to_device_tensor` calls must happen outside the trace.
2. **You MUST remove all host (CPU) logic** from the traced region, even if it's small. No Python conditionals, no tensor creation, no shape computation. The trace is a static sequence of device ops.
3. **You MUST pre-allocate all tensors** before capture. Every tensor used inside the trace must already exist on device with a fixed address.
4. **Use scratch tensors** shared between ops and iterations. Pre-allocate reusable intermediate buffers and pass them as `output_tensor` arguments. This avoids dynamic allocation inside the trace.

## Basic Pattern

```python
# 1. Pre-allocate all tensors that will be used in the trace
trace_input = ttnn.from_torch(dummy_input, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=device,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

# 2. Capture the trace (runs the ops once to record them)
trace_id = ttnn.begin_trace_capture(device, cq_id=0)
result = ttnn.matmul(trace_input, weights)
result = ttnn.relu(result)
ttnn.end_trace_capture(device, trace_id, cq_id=0)
ttnn.synchronize_device(device)

# 3. Replay with new inputs (no dispatch overhead)
for batch in batches:
    ttnn.copy_host_to_device_tensor(batch_host_tensor, trace_input)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
```

`synchronize_device` is only needed if you use non-blocking execution. If you pass `blocking=True` to `execute_trace`, you don't need it (but you lose the ability to overlap host work).

## Multi-Chip Traces

Traces work with mesh devices and collective operations:

```python
trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
partial = ttnn.matmul(x_sharded, w_sharded)
reduced = ttnn.all_reduce(partial)
ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
```
