---
name: tt-lang-profile-optimize
description: Profile and optimize TT-Lang kernels for performance. Covers auto-profiling, perf summary, signposts, and optimization workflow.
argument-hint: <kernel-file>
---

## External Resources

- [Performance Tools Reference](performance-tools.md) -- environment variables, perf summary, auto-profiling, signposts, Perfetto trace server
- [TT-Lang Specification, Section 10](../tt-lang/TTLangSpecification.md#10-performance-and-debugging) -- signpost and dprint language spec


## Before You Start: Understand the Real Workload

Before optimizing, ask the user:
- **What data sizes will this kernel run on in production?** The test data may be much smaller than the real workload. Do not over-optimize for test data.
- **Where will this run?** Single chip? Multi-chip? This affects core budget.

Keep real-world constraints in mind throughout. For example, do not move everything to L1 just because the test data fits -- if the real data is larger, you need streaming. Optimizations must hold for the production workload, not just the test case.

## Optimization Targets

Three goals, in priority order:

### 1. Maximize Core Utilization (target: 100%)

The kernel MUST use all available cores. If the kernel runs on `grid=(1, 1)`, it is leaving performance on the table. Partition work across cores using the multicore patterns from the [tt-lang skill](../tt-lang/SKILL.md). Check `PERF SUMMARY` for grid size.

### 2. Reduce DRAM Traffic

Minimize unnecessary DRAM reads and writes. Key strategies:
- Fuse multiple kernels into one (eliminate intermediate DRAM round-trips)
- Stream large tensors through small CBs instead of loading everything at once
- Reuse data already in L1 instead of re-reading from DRAM

Note: if tensors are small enough, moving them to L1 memory space (`ttnn.L1_MEMORY_CONFIG`) avoids DRAM reads entirely, but this only helps when the data actually fits.

Check `PERF SUMMARY` for DRAM read/write volumes and effective bandwidth.

### 3. Increase DFB Block Size

Larger DFB shapes (block sizes) mean fewer DMA transfers and better throughput. Keep increasing `shape=(R, C)` on dataflow buffers until you run out of L1 (~1.5MB per core). This is often a big win.

## Iteration Flow

### Step 0: Establish Validation Criteria

Ask the user how to verify correctness after each optimization. Examples: numerical comparison against a reference output, assertion in the test script, visual inspection, or a tolerance threshold. Use this criteria throughout the optimization loop to ensure changes don't break the kernel.

### Step 1: Establish Baseline

Run with perf profiling on hardware:
```bash
# Via run-test.sh:
run-test.sh --perf --hw /path/to/kernel.py

# Or directly:
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TTLANG_PERF_DUMP=1 python /path/to/kernel.py
```

Record:
- **Wall time** (duration in us from `PERF SUMMARY`) -- this is your ground truth
- Grid size and core count
- DRAM read/write volumes
- Effective bandwidth
- Compute vs memory bound ratio

### Step 2: Identify Bottleneck

From the baseline, determine the primary bottleneck:
- **Underutilized cores**: grid is smaller than the data allows
- **Excess DRAM traffic**: multiple kernel calls, redundant reads, no streaming
- **Large transfer count with small transfer size**: DFB block size too small
- **Compute bound**: heavy math ops, possible to restructure
- **Memory bound**: data movement dominates, possible to overlap or reduce

### Step 3: Propose Plan

Present your optimization plan to the user **before making changes**. Include:
- What the bottleneck is and why
- What you plan to change
- Expected impact on wall time

Wait for user approval.

### Step 4: Implement and Measure

Make ONE change at a time. After each change:

1. Verify correctness using the validation criteria from Step 0
2. Run with perf profiling to measure performance (see Step 1 for commands)
3. Compare wall time against baseline: did it improve? By how much?
4. If it regressed or broke correctness, revert and try a different approach

**Wall time is the metric that matters.** Other metrics (cycles, BW) are diagnostic tools to understand why wall time changed.

Iterate as many times as needed. There is no limit on profiling runs. Keep going until you've exhausted the optimization targets or hit diminishing returns.

### Step 5: Report

Summarize:
- **Wall time: baseline vs final** (the most important number)
- Baseline vs final metrics (cycles, BW, core utilization)
- What changes were made and their individual wall time impact
- Any remaining bottlenecks

## Output

1. Baseline wall time and metrics
2. Final wall time and metrics
3. Wall time delta (speedup)
4. Summary of each change and its impact

## Constraint: One Kernel Invocation

The auto-profiler can only profile **one kernel invocation at a time**. Before profiling, read the file and check if there are multiple `@ttl.kernel` calls or if a kernel is called in a loop. If so, comment out extra invocations so only the target kernel runs once. If it's ambiguous which kernel to profile, ask the user.

## Auto profiling

The auto profiler can produce line-by-line cycle counts. The user may request this or you may find it helpful for optimizing. To auto profile, use the below flow.

### Step 1: Prepare and Run

Read the input file. Ensure only one kernel invocation will execute (see constraint above).

Run with auto-profiling on hardware:
```bash
# Via run-test.sh:
run-test.sh --auto-profile --hw /path/to/kernel.py

# Or directly:
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TTLANG_AUTO_PROFILE=1 python /path/to/kernel.py
```

Then read the output log.

### Step 2: Analyze Results

The log contains several sections. Grep for these markers:

- `THREAD SUMMARY` -- Per-thread cycle counts, op counts, and a compute-vs-memory bound analysis with a visual bar
- `PERF SUMMARY` -- Grid size, duration, DRAM read/write volumes, effective bandwidth, transfer sizes, barrier counts
- `DFB FLOW GRAPH` -- JSON describing dataflow buffer producer/consumer relationships and DMA ops
- `PIPE GRAPH` -- Inter-core pipe communication graph (may be empty if no pipes)

### Step 3: Report to User

Summarize:
- What was profiled (kernel name, grid size, tensor shapes)
- Whether the kernel is compute-bound or memory-bound, and by how much
- Per-thread cycle breakdown and hotspots
- DRAM bandwidth utilization and transfer patterns
- Any anomalies (imbalanced cores, unexpected stalls, low BW)

Be specific with numbers.

### Step 4: Provide Full Profiler Command

Give the user a command they can run directly on the server for the full auto-profiler result:
```
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TTLANG_AUTO_PROFILE=1 python <path_to_kernel.py>
```

Do NOT include `TTLANG_PERF_DUMP=1` in this command.

## Output

1. Profile summary with cycle counts and bound analysis
2. Interesting findings called out
3. The server command for full profiler output
