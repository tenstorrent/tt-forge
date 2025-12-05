# Project Context

## Purpose
TT-Forge creates a release that includes `tt-xla` and `vllm-tt` wheels. This repo is also used for releasing and shared GitHub actions across our MLIR-based compiler repos (tt-xla, tt-mlir, tt-forge-fe).

## Repository Structure
- `.github/actions/`: **Shared Actions** - Reusable CI/CD logic (sources of truth for release steps).
- `.github/workflows/`: **Workflows** - Entry points for CI/CD (nightly, release, PR checks).
- `benchmark/`: **Performance Suite** - Python scripts measuring model performance.
- `demos/`: **Usage Examples** - End-to-end scripts demonstrating model usage.
- `openspec/`: **Planning** - Specifications and change proposals.
- `docs/`: **Documentation** - Source files for project documentation.

## Project Conventions

### Best Practices (All Languages)
- **Never Nester** - Keep nesting ≤2–3 levels using these techniques:
  - **Inversion** - Check for invalid/error conditions first and return/exit early (guard clauses).
  - **Extraction** - Pull nested logic into separate, well-named functions.
  - **Early Returns** - Exit functions as soon as the result is known.
  - **Continue in Loops** - Skip iterations early with `continue` instead of wrapping in `if`.
- **Composition over Inheritance** - Prefer composing behavior from small, reusable components over deep class hierarchies.
  - Inheritance is rigid; once a class is placed in a hierarchy, it's frozen there.
  - If you create classes like `LifedMovingEntity` or `ValidatedAuthenticatedUser`, your design has problems.
  - Compose objects from focused components that each do one thing.
- **Strategy Pattern (Policy Pattern)** - Replace conditionals with interchangeable strategy objects.
  - Lets algorithms vary independently from clients that use them.
  - Encapsulate behaviors in interfaces, not inheritance hierarchies.
  - Swap strategies at runtime or design-time without changing client code.
  - Follows Open/Closed Principle: add new strategies without modifying existing code.
- **Dependency Injection** - Pass dependencies into objects rather than creating them internally.
  - Makes code testable by allowing mock/fake dependencies.
  - Decouples components; changes to dependencies don't require changes to consumers.
  - Prefer constructor injection over service locators or global state.
- **Small, Focused Functions** - Each function should do one thing well (~40 lines max).
- **Fail Fast** - Check for errors/invalid conditions early and return/exit immediately.
- **DRY (Don't Repeat Yourself)** - Extract repeated logic into functions or shared modules.
- **Meaningful Names** - Use descriptive names that reveal intent; avoid abbreviations.
- **Comments Explain "Why"** - Code should be self-documenting; comments explain reasoning, not mechanics.
- **Consistent Style** - Follow the established conventions for the language; use formatters/linters.
- **Handle Errors** - Always handle errors explicitly; never silently swallow exceptions.
- **Right Tool for the Job** - Choose the appropriate language for the task:
  - Shell: Small utilities, simple wrappers, calling other programs. Rewrite in Python if >100 lines.
  - Python: Data manipulation, complex logic, anything needing maintainability.
- **Prefer Async & Non-Blocking Solutions** - Use async/await and non-blocking patterns for I/O-bound operations:
  - Enables concurrent execution without threads.
  - Better resource utilization for network calls, file I/O, and waiting on external processes.
  - In Python: use `asyncio`, `aiohttp`, `aiofiles`.
  - In GitHub Actions: run independent jobs in parallel; use matrix strategies.
- **TODO Comments:** Format: `# TODO(username): Description of task`.

### Code Style
- **SPDX Headers Required:** Every source file must have SPDX license headers.
  - Checked by `check-copyright` pre-commit hook (config: `.github/check-spdx.yml`).
  - Python: `# SPDX-FileCopyrightText: © {current_year} Tenstorrent AI ULC` followed by `# SPDX-License-Identifier: Apache-2.0`
  - Bash: `# SPDX-FileCopyrightText: © {current_year} Tenstorrent AI ULC` followed by `# SPDX-License-Identifier: Apache-2.0`
- **Python Style:** (Based on [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html))
  - **Formatter:** `black` (line-length=120)
  - **Linter:** `flake8` (via pre-commit/CI); consider `pylint` for deeper analysis.
  - **Shebang:** Use `#!/usr/bin/env python3` for executable scripts.
  - **Formatting:**
    - Indent with 4 spaces (no tabs).
    - 2 blank lines between top-level definitions; 1 blank line between methods.
    - One statement per line; no semicolons.
  - **Imports:**
    - Order: standard library → third-party → local (separated by blank lines).
    - Use full package paths; avoid relative imports.
    - Use `import x` for packages/modules; `from x import y` for specific items.
  - **Naming:**
    - `module_name`, `package_name`, `file_name.py` (lowercase with underscores).
    - `ClassName` (CapWords).
    - `function_name`, `method_name`, `variable_name` (lowercase with underscores).
    - `CONSTANT_NAME` (uppercase with underscores).
    - `_private` (single underscore prefix for internal use).
  - **Strings:**
    - Use f-strings for formatting: `f"Value: {value}"`.
    - Use consistent quote style (`"` or `'`); prefer `"` for strings with apostrophes.
  - **Functions:**
    - Use `@property` instead of `get_`/`set_` accessor methods.
    - Use `if __name__ == '__main__':` pattern for executable scripts.
  - **Exceptions:**
    - Use specific exceptions; avoid bare `except:` clauses.
    - Prefer `raise ValueError("msg")` over `raise ValueError, "msg"`.
  - **Type Hints:**
    - Required for public APIs.
    - Use `typing` module for complex types; prefer built-in generics (`list[int]` over `List[int]`).
  - **Docstrings:** Google-style with `Args:`, `Returns:`, `Raises:` sections.
  - **Resources:** Use `with` statement for files, sockets, and database connections.
  - **Best Practices:**
    - Use implicit boolean: `if items:` not `if len(items) > 0:`.
    - Don't use mutable default arguments: use `None` then assign inside function.
    - Use list/dict comprehensions for simple cases; avoid nested comprehensions.
    - Use lambda only for one-liners; otherwise define a function.
  - **Avoid:** Metaclasses, `__del__`, bytecode manipulation, dynamic attribute access via strings.
- **Bash Style:** (Based on [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html))
  - **When to Use:** Shell for small utilities only; scripts >100 lines should be rewritten in Python.
  - **Interpreter:** Use `#!/bin/bash` (bash only, no `sh`).
  - **Linter:** Run `shellcheck` on scripts.
  - **Formatting:**
    - Indent with 2 spaces (no tabs).
    - Line length: ~80 characters.
    - Put `; then` and `; do` on same line as `if`/`while`/`for`.
  - **Naming:**
    - `snake_case` for local variables, functions, and filenames.
    - `UPPERCASE_SNAKE_CASE` for constants and exported variables.
  - **Structure:**
    - Define a `main` function for scripts with multiple functions.
    - Place functions at top, constants below shebang, `main "$@"` at bottom.
  - **Best Practices:**
    - Use `local` for function-scoped variables (declare and assign separately: `local var; var="$(cmd)"`).
    - Use `readonly` for constants.
    - Quote all variables: `"${var}"`.
    - Prefer `$(...)` over backticks for command substitution.
    - Prefer `[[ ... ]]` over `[ ... ]` for tests.
    - Use `(( ... ))` for arithmetic.
    - Use arrays for lists: `"${array[@]}"`.
    - Prefer shell builtins over external commands.
    - Check return values with `if` or `$?`; use `PIPESTATUS` for pipelines.
  - **Error Handling:**
    - Use `set -e` to fail fast.
    - Print errors to `STDERR` using `>&2`.
  - **Avoid:** `eval` (security risk), aliases (use functions instead).
- **GitHub Actions Style:**
  - **Workflow Tiers:**
    - **Primary workflows:** High-level orchestrators (e.g., `pr-main`, `push-main`, `scheduled-nightly`, `manual-test`).
    - **Reusable workflows:** Prefixed with `call-` (e.g., `call-build`, `call-test`, `call-docker-build`).
  - **Artifact Naming:**
    - Build artifacts: `build-{config}-{sha}` (e.g., `build-release-a1b2c3d`).
  - **Prefer Scripts Over Inline YAML:**
    - If a `run:` block exceeds ~10 lines, extract it to a script in `.github/scripts/`.
    - Scripts are testable, lintable, and reusable; inline YAML is not.
  - **Script Organization:**
    - **Bash scripts:** Place in `.github/scripts/` (e.g., `.github/scripts/build-wheel.sh`).
    - **Python scripts:** Place in `.github/scripts/{script_name}/` with:
      - `main.py` - the program
      - `test_main.py` - tests
      - `requirements.txt` - dependencies

### Error Messages
Error messages must be:
- **Specific:** Include actual values and conditions that caused the error
- **Explanatory:** Provide context on why the error occurred
- **Actionable:** Give clear guidance on how to resolve the issue
- Avoid vague messages, ALL CAPS, exclamation points, and the word "bad"

### Architecture Patterns
- **Github Actions Pipeline Pattern:**
  - **Primary Workflows** (orchestrators):
    - `pr-main`: Validates PRs → pre-commit → inspect-changes → conditional build → test.
    - `push-main`: Creates canonical artifacts → build (force) → test → update-docs → tag-docker.
    - `scheduled-nightly`: Reuses canonical build → nightly + perf tests → release to PyPI.
    - `scheduled-optional-tests`: Weekly long-running tests reusing canonical build.
    - `manual-test`: Ad-hoc testing with artifact reuse; supports cross-repo triggering.
  - **Reusable Workflows** (building blocks, prefixed `call-`):
    - `call-pre-commit`: Style/format checks (black, clang-format).
    - `call-docker-build`: Builds layered Docker images; skips if hash unchanged.
    - `call-build`: Compiles project; reuses existing artifacts if available.
    - `call-test`: Runs test suites; downloads artifacts; uploads reports.
    - `call-tag-docker`: Tags Docker image with `latest` on main or tagged with git sha in PR.
  - **Key Patterns:**
    - **Inspect-changes:** Detects what changed → sets flags (skip-build, run-pr-tests, run-perf-tests).
    - **Canonical build:** `push-main` is single source of truth; downstream reuses via `run_id`.
    - **Docker hash caching:** Skip rebuild if Dockerfile/requirements hash unchanged.
    - **Cross-repo triggering:** `manual-test` accepts `mlir-override-sha` for compatibility checks.

### Testing Strategy
- **Python Testing:**
  - **Framework:** Use `pytest` for all Python tests.
  - **Unit Tests:** Test individual functions/classes in isolation; aim for fast, deterministic tests.
  - **Mocking:** Use `unittest.mock` or `pytest-mock` to isolate units from dependencies.
    - Mock external services, hardware calls, and slow operations.
    - Prefer dependency injection to make code testable.
  - **Naming:** `test_<module>.py` for files, `test_<function>_<scenario>` for test functions.
  - **Assertions:** Use specific assertions (`assert x == y`) over generic (`assert x`).
- **Bash Testing:**
  - **Framework:** Use `bats-core` (Bash Automated Testing System) for Bash tests.
  - **Helpers:** Use `bats-support`, `bats-assert`, and `bats-file` for common assertions.
  - **Naming:** `test_<script>.bats` for test files.
  - **Structure:** Each `@test` should test one behavior; use `setup` and `teardown` for fixtures.
  - **Assertions:** Use `run` to capture output/status; check `$status` and `$output`.
- **GitHub Actions Testing:**
  - **Linter:** Use `actionlint` to catch syntax errors, type mismatches, and invalid references.
  - **Path-based Triggers:** Test workflows should trigger only when relevant workflow/action files change.
  - **Isolation:** Use a `draft` or `test` prefix for test artifacts (tags, releases, branches) to isolate from production.
  - **Mock Actions:** Create mock/stub actions to simulate operations without real side effects.
  - **Validation:** Define expected artifacts upfront; validate they exist after the workflow completes.
  - **Cleanup:** Delete test artifacts after validation to keep the repository clean.

### Performance Guidelines
- **Profile Before Optimizing** - Measure first; don't guess where bottlenecks are.
- **Algorithm First** - Choose the right algorithm/data structure before micro-optimizing.
- **Avoid Premature Optimization** - Write clear code first; optimize only when data proves it's needed.
- **Branchless When It Matters** - In hot paths, prefer arithmetic/bitwise operations over branches to avoid pipeline stalls.
- **Memory Matters** - Be aware of memory allocation in hot paths; prefer pre-allocation.

## Domain Context
- **Related Repos:**
  - [TT-MLIR](https://github.com/tenstorrent/tt-mlir) - Core compiler
  - [TT-XLA](https://github.com/tenstorrent/tt-xla) - JAX/PyTorch frontend and include tt-vllm
  - [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe) - ONNX/TF frontend
  - [TT-Metal](https://github.com/tenstorrent/tt-metal) - Low-level SDK

## Important Constraints
- **TT-XLA recommended for PyTorch** (TT-Forge-FE supports it but TT-XLA is preferred)
- **TT-Forge-FE is single-chip only** (TT-XLA supports multi-chip)
- **TT-Torch is deprecated** - use TT-XLA instead
- **Apache 2.0 License** - all contributions must be compatible

## Release Process
- See [RELEASE.md](../RELEASE.md).

## Installation & Usage

### Installation
The `tt-forge` package acts as a central hub. Installation typically depends on the frontend you are using (TT-XLA or TT-Forge-FE).

1.  **Docker (Recommended)**:
    -   Pull the latest Docker image: `docker pull ghcr.io/tenstorrent/tt-forge-slim:latest`
    -   See [TT-XLA Docker Docs](https://docs.tenstorrent.com/tt-xla/getting_started_docker.html) or [TT-Forge-FE Docker Docs](https://docs.tenstorrent.com/tt-forge-fe/getting_started_docker.html) for specific run commands.

2.  **Pip Wheel**:
    -   Wheels are hosted on Tenstorrent's public PyPI.
    -   Install via pip pointing to the index (check documentation for specific URL):
        ```bash
        pip install tt-forge --extra-index-url https://pypi.tenstorrent.com/simple
        ```
