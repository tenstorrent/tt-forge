---
name: model-bringup-tt-hardware
description: Install tt-forge, run the model loader from the cpu bringup branch on Tenstorrent hardware, iterate on failures, and open a PR to tenstorrent/tt-forge-models on success.
argument-hint: <model_id> <branch_name> <device> [--report <path>]
---

# TT Hardware Bringup: Validate on Tenstorrent Hardware and Open PR

You are running inside a GitHub Actions job on a Tenstorrent machine (Ubuntu 24.04, `/dev/tenstorrent` present). The `gh` CLI is already authenticated. Git identity is pre-configured.

## Arguments

Parse from the invocation line before proceeding:

| Argument | Example | Required |
|----------|---------|----------|
| `<model_id>` | `meta-llama/Llama-3.2-1B` | yes |
| `<branch_name>` | `claude/bringup-llama-3-2-1b` | yes |
| `<device>` | `n150` | yes |
| `--report <path>` | `--report /github/workspace/report.md` | no |

**`REPORT_PATH`** — if `--report` is present use that value; otherwise default to `./bringup-report-tt-hardware.md` (relative to current working directory, useful when running locally).

**`FORGE_MODELS_DIR`** — set by the workflow to `$GITHUB_WORKSPACE/tt-xla/third_party/tt_forge_models` (already populated as a submodule; Step 2 switches it to the CPU bringup branch).

**`GITHUB_WORKSPACE`** — set by the workflow; if absent (local run), fall back to the directory of `REPORT_PATH` for the status file.

## Step 1 — Install model-specific dependencies

Read `loader.py` and check the HuggingFace model card for any imports not yet available. Install only what is actually needed:

```bash
pip install <packages identified from loader imports and model card>
```

## Step 2 — Run on Tenstorrent hardware (up to 4 attempts)

Replace `FAMILY` and `TASK` with the actual paths from the CPU bringup loader.

**Attempt 1** (bfloat16):
```python
import os, sys, torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
sys.path.insert(0, os.environ["FORGE_MODELS_DIR"])
xr.set_device_type("TT")
device = xm.xla_device()
from FAMILY.TASK.pytorch import ModelLoader
loader = ModelLoader()
model = loader.load_model(dtype_override=torch.bfloat16).eval().to(device)
inputs = {k: v.to(device) for k, v in loader.load_inputs().items()}
with torch.no_grad():
    outputs = model(**inputs)
xm.mark_step()
print("HARDWARE SUCCESS")
```

**Attempt 2** — dtype error: switch to float32 in `load_model` and `load_inputs`.

**Attempt 3** — compilation error: try reducing sequence length (`input_ids = input_ids[:, :32]`) or check error for unsupported ops.

**Attempt 4** — loader.py needs changes: edit the file, re-run CPU test to confirm it still passes, then leave the changes uncommitted in `$FORGE_MODELS_DIR` — the workflow step that follows will commit, push, and open the PR.

## Step 3 — Write status file

Write `$GITHUB_WORKSPACE/bringup-hw-status.txt` (the workflow reads this to decide whether to push fixes and open the PR):

On hardware **success**:
```bash
echo "SUCCESS" > "$GITHUB_WORKSPACE/bringup-hw-status.txt"
```

On hardware **failure** (after all attempts):
```bash
echo "FAILED" > "$GITHUB_WORKSPACE/bringup-hw-status.txt"
```

## Step 4 — Write report

Write the report to `$REPORT_PATH` (default `./bringup-report-tt-hardware.md`).

```bash
mkdir -p "$(dirname "$REPORT_PATH")"
```

Include:
- Model ID, branch, device, status (SUCCESS/FAILED)
- Hardware test output or error snippet
- Each fix attempt and outcome
- PR URL (on success) or root-cause analysis (on failure): is it an unsupported op, dtype issue, or architecture limitation? What would a human need to do to continue?
