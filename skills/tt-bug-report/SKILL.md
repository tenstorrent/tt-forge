---
name: tt-bug-report
description: File a bug report with a reproducer against Tenstorrent repos (tt-lang, tt-metal, tt-xla)
argument-hint: <description of the problem>
---

## Task

File a bug report for a Tenstorrent issue. Gather context, minimize the reproducer, and create a GitHub issue in the right repo.

## Where to File

| Component | Repo |
|-----------|------|
| TT-Lang (DSL, compiler, kernels) | `tenstorrent/tt-lang` |
| TTNN (ops, tensor APIs, device) | `tenstorrent/tt-metal` |
| TT-Metal (low-level C++ kernels) | `tenstorrent/tt-metal` |
| TT-XLA (PyTorch/JAX frontend) | `tenstorrent/tt-xla` |

If it's unclear which repo the bug belongs to, ask the user.

## Input

$ARGUMENTS

## Process

### Step 1: Describe the Problem

Determine the type of issue and gather relevant info:

**For crashes or incorrect results:**
- A reproducer script (even a non-minimal one is valuable)
- Actual vs expected output

**For compiler errors (tt-lang):**
- The full error message
- Compiler flags used
- MLIR before and after passes (`/tmp/ttlang_initial.mlir`, `/tmp/ttlang_final.mlir`)
- Any other context that seems relevant

### Step 2: Minimize the Reproducer

Ask the user if it's OK to upload the reproducer to the issue and spend some iterations trying to minimize it. If they agree, try to reduce it to the smallest script that still triggers the bug. Any reproducer is better than none -- if you can't minimize further, use what you have.

### Step 3: File the Issue

Use `gh` to create an issue in the appropriate repo:

```bash
gh issue create --repo <owner/repo> \
  --title "[external bug report][from claude] <short title>" \
  --body "$(cat <<'EOF'
<body>
EOF
)"
```

The body should include:
- Description of the problem
- Reproducer (if available)
- Error output / MLIR snippets (if applicable)
- Environment context (simulator vs HW, etc.)

## Output

Return the issue URL to the user.
