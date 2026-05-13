---
name: model-bringup-cpu
description: Write a ForgeModel-compatible loader for a HuggingFace model, validate it on CPU, and push the result to a branch on tenstorrent/tt-forge-models.
argument-hint: <model_id> <branch_name> [--report <path>]
---

# CPU Bringup: Write Loader and Validate on CPU

You are running inside a GitHub Actions job on a Tenstorrent machine (Ubuntu 24.04 container). The `gh` CLI is already authenticated with a token that has write access to `tenstorrent/tt-forge-models`. Git identity is pre-configured.

## Arguments

Parse from the invocation line before proceeding:

| Argument | Example | Required |
|----------|---------|----------|
| `<model_id>` | `meta-llama/Llama-3.2-1B` | yes |
| `<branch_name>` | `claude/bringup-llama-3-2-1b` | yes |
| `--report <path>` | `--report /github/workspace/report.md` | no |

**`REPORT_PATH`** — if `--report` is present use that value; otherwise default to `./bringup-report-cpu.md` (relative to the current working directory, useful when running locally).

**`STATUS_FILE`** — always `$GITHUB_WORKSPACE/bringup-cpu-status.txt` (the workflow reads exactly this path to decide whether to commit and push).

**`FORGE_MODELS_DIR`** — read from the `$FORGE_MODELS_DIR` environment variable (set by the workflow to `$GITHUB_WORKSPACE/tt-xla/third_party/tt_forge_models`). tt-forge-models is already checked out here as a submodule by the workflow.

**`TT_XLA_DIR`** — read from the `$TT_XLA_DIR` environment variable (set by the workflow to `$GITHUB_WORKSPACE/tt-xla`).

**Activate the tt-xla venv before every Python or pip command:**
```bash
source "$TT_XLA_DIR/venv/activate"
```
The venv is pre-created by the workflow setup step and has torch, jax, and the latest tt-forge wheel installed.

## Step 1 — Study the interface

Read these files to understand what you must implement:
- `$FORGE_MODELS_DIR/base.py` — the `ForgeModel` abstract base class (`load_model`, `load_inputs`, `_get_model_info`)
- `$FORGE_MODELS_DIR/config.py` — `ModelVariant`/`StrEnum`, `ModelGroup`, `ModelTask`, `ModelSource`, `Framework`, `LLMModelConfig`
- `$FORGE_MODELS_DIR/README.md` — directory structure and interface contract

## Step 2 — Find 2-3 similar existing loaders as templates

All paths under `$FORGE_MODELS_DIR/`:
- **Causal LM**: `llama/causal_lm/pytorch/loader.py`, `gpt2/causal_lm/pytorch/loader.py`
- **Masked LM / encoder**: `bert/masked_lm/pytorch/loader.py`
- **Vision / image classification**: `efficientnet/image_classification/pytorch/loader.py`, `resnet/image_classification/pytorch/loader.py`
- **Sequence classification**: any `sequence_classification/pytorch/loader.py`

Read at least 2 loaders that are closest in architecture to the target model.

## Step 3 — Inspect the HuggingFace model

```bash
python3 - <<'PYEOF'
from huggingface_hub import model_info
info = model_info("MODEL_ID_HERE")
print("Pipeline tag:", info.pipeline_tag)
print("Library:", info.library_name)
print("Tags:", info.tags[:20])
PYEOF
```

Also check transformers auto-classes:
```bash
python3 - <<'PYEOF'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("MODEL_ID_HERE", trust_remote_code=False)
print("Model type:", cfg.model_type)
print("Architectures:", getattr(cfg, 'architectures', None))
PYEOF
```

Use this to determine:
- **model_family** — short lowercase name (e.g. `mistral`, `phi`, `gemma`) used as the top-level directory name
- **task** — one of: `causal_lm`, `masked_lm`, `sequence_classification`, `image_classification`, `token_classification`, `question_answering`
- **ModelTask enum value** — matching entry from `config.py` (e.g. `ModelTask.NLP_CAUSAL_LM`)
- **transformers class** — e.g. `AutoModelForCausalLM`, `AutoModelForSequenceClassification`
- **variant name** — short identifier like `1b`, `7b`, `base`, `large`

## Step 4 — Write the loader files

Create this directory structure under `$FORGE_MODELS_DIR/` (replace family, task, variant placeholders):

```
third_party/tt_forge_models/
  <family>/
    __init__.py
    <task>/
      __init__.py
      pytorch/
        __init__.py
        loader.py
```

### loader.py requirements

Follow EXACTLY the pattern from the reference loaders you read. Key requirements:
- SPDX header (first 3 lines, Apache-2.0)
- `ModelVariant(StrEnum)` with at least one variant
- `_VARIANTS` dict mapping variant to `LLMModelConfig(pretrained_model_name=...)`
- `DEFAULT_VARIANT = ModelVariant.<VARIANT>`
- `ModelLoader(ForgeModel)` class
- `__init__` calls `super().__init__(variant)`, initialises `self._tokenizer = None` (for NLP) or similar
- `_get_model_info` returns `ModelInfo(model=..., variant=variant, group=ModelGroup.GENERALITY, task=..., source=ModelSource.HUGGING_FACE, framework=Framework.TORCH)`
- `load_model(dtype_override=None)` loads via appropriate Auto class with `from_pretrained`, applies dtype if provided
- `load_inputs(dtype_override=None)` creates a small batch of sample inputs (tokenized text or dummy tensors), returns a dict

### __init__.py for each directory

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
```

## Step 5 — Install any model-specific requirements

```bash
source "$TT_XLA_DIR/venv/activate"
uv pip install <any-extra-packages>  # e.g. tiktoken, einops, timm
```

Check the HuggingFace model card or loader imports for extra requirements not already in the venv.

## Step 6 — CPU test (up to 3 attempts)

```bash
source "$TT_XLA_DIR/venv/activate"
cd "$FORGE_MODELS_DIR"
python3 - <<'PYEOF'
import sys, torch
sys.path.insert(0, ".")
from FAMILY.TASK.pytorch import ModelLoader
loader = ModelLoader()
model = loader.load_model()
model.eval()
inputs = loader.load_inputs()
with torch.no_grad():
    outputs = model(**inputs)
print("CPU SUCCESS")
print("Output type:", type(outputs))
PYEOF
```

On failure:
1. Read the full traceback. Fix import errors, missing dependencies, wrong tensor shapes.
2. If the model is too large for available RAM, try loading with `torch_dtype=torch.float32` and check the model size first.
3. If the `load_inputs()` dict keys do not match what the model expects, check `model.forward` signature and adjust.
4. Maximum 3 fix attempts. After 3 failures, skip to the failure report.

## Step 7 — Write status file and report

```
REPORT_PATH = <--report value>  OR  ./bringup-report-cpu.md
STATUS_FILE = $GITHUB_WORKSPACE/bringup-cpu-status.txt
             (if GITHUB_WORKSPACE is not set, fall back to ./bringup-cpu-status.txt)
```

On CPU test **success**:
```bash
mkdir -p "$(dirname "$REPORT_PATH")"
echo "SUCCESS" > "${GITHUB_WORKSPACE:-$(dirname "$REPORT_PATH")}/bringup-cpu-status.txt"
```

On CPU test **failure** (after all attempts):
```bash
mkdir -p "$(dirname "$REPORT_PATH")"
echo "FAILED" > "${GITHUB_WORKSPACE:-$(dirname "$REPORT_PATH")}/bringup-cpu-status.txt"
```

The workflow reads `$STATUS_FILE` to decide whether to commit and push — you do not need to run any git commands.

Write a Markdown report to `$REPORT_PATH`:

```markdown
# CPU Bringup Report

**Model:** MODEL_ID
**Branch:** BRANCH_NAME
**Status:** SUCCESS or FAILED

## Files created
- List of files created with brief description

## CPU test result
Pass or Fail — include relevant output or error snippet

## Issues encountered and fixes applied
- Bullet list of any problems hit and how they were resolved

## Notes for TT hardware bringup
- Any known limitations, dtype requirements, or size warnings
```
