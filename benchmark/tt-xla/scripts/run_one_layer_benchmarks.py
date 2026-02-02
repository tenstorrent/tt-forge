#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import inspect
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import pytest

# Critical error messages that should stop execution
CRITICAL_ERRORS = [
    "Read unexpected run_mailbox value from core",
    "Timeout waiting for Ethernet core service remote IO request",
]


def _handle_critical_error() -> None:
    """Print device error message and exit."""
    print(
        f"\n{'='*60}\n"
        f"DEVICE ERROR: TT device needs reset\n"
        f"{'='*60}\n"
        f"\n"
        f"  >>> Run: tt-smi -r\n"
        f"  >>> Then re-run with --continue to resume.\n"
        f"\n"
        f"{'='*60}\n"
    )
    sys.exit(1)


def _resolve_paths() -> Tuple[Path, Path]:
    scripts_dir = Path(__file__).resolve().parent
    tt_xla_dir = scripts_dir.parent
    repo_root = tt_xla_dir.parents[1]
    return tt_xla_dir, repo_root


def _ensure_import_paths(repo_root: Path, tt_xla_dir: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    tt_xla_str = str(tt_xla_dir)
    if tt_xla_str not in sys.path:
        sys.path.insert(0, tt_xla_str)


def _find_ttirs_by_model(export_path: Path, model_name: str) -> list[Path]:
    """Find TTIR files for a specific model by searching for files matching the model name pattern.
    
    This searches for files after the test completes, similar to generate_transformer_layers.py.
    Files are typically in modules/irs/ with names like: ttir_falcon3_1b_lyr1_bs32_isl128_run2048_g0_*.mlir
    """
    if not export_path.exists():
        return []

    # Normalize model name for matching (remove underscores/hyphens, lowercase)
    model_name_norm = model_name.lower().replace("_", "").replace("-", "")
    
    paths = []
    # Search for .mlir files (the actual format, not .ttir)
    for path in export_path.rglob("*.mlir"):
        # Parse the filename stem (without extension)
        # Files are named like: ttir_falcon3_1b_lyr1_bs32_isl128_run2048_g0_*.mlir
        filename_stem = path.stem  # e.g., "ttir_falcon3_1b_lyr1_bs32_isl128_run2048_g0_1770047972511"
        
        # Check if filename contains model name and lyr pattern (lyr1, lyr2, etc.)
        stem_norm = filename_stem.lower().replace("_", "").replace("-", "")
        has_model_name = model_name_norm in stem_norm
        has_lyr = re.search(r"lyr1", stem_norm) is not None
        
        if has_model_name and has_lyr:
            paths.append(path)

    # Return sorted by modification time (most recent first)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def _parse_export_name(export_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse export name from directory or filename.
    
    Handles both formats:
    - Directory: model_name_1lyr_bs32_isl128
    - Filename: ttir_exactname_lyr1_bs32_isl128_run2048_g0_*
    """
    bs_match = re.search(r"(?:^|_)bs(\d+)(?:_|$)", export_name)
    isl_match = re.search(r"(?:^|_)isl(\d+)(?:_|$)", export_name)
    bs = bs_match.group(1) if bs_match else None
    isl = isl_match.group(1) if isl_match else None

    # Extract model name and lyr pattern directly
    # Match: ttir_<name>_lyr1 or <name>_1lyr
    lyr_match = re.search(r"(?:^ttir_)?(.+?)_(?:lyr(\d+)|(\d+)lyr)(?:_|$)", export_name, re.IGNORECASE)
    if lyr_match:
        base_name = lyr_match.group(1)
        lyr_num = lyr_match.group(2) or lyr_match.group(3) or "1"
        # Normalize to _1lyr format for consistency
        export_name = f"{base_name}_{lyr_num}lyr"
        return export_name, bs, isl
    else:
        # If no lyr pattern found, this is not a valid one-layer export
        return None, bs, isl


def _graph_label_from_stem(stem: str) -> str:
    match = re.search(r"(?:^|_)(g[01])(?:_|$)", stem)
    if match and match.group(1) == "g0":
        return "prefill"
    if match and match.group(1) == "g1":
        return "decode"
    return "graph"


def _normalize_ttir_name(ttir_path: Path, group: str) -> str:
    # Parse from filename stem (files are in modules/irs/ with names like ttir_model_lyr1_bs32_isl128_*.mlir)
    export_name, bs, isl = _parse_export_name(ttir_path.stem)
    # export_name should never be None here since files are filtered during collection
    assert export_name is not None, f"Expected lyr pattern in {ttir_path.stem}"
    
    graph_label = _graph_label_from_stem(ttir_path.stem)
    if group == "llm":
        parts = [export_name]  # export_name already includes the lyr pattern
        parts.append(graph_label)
        include_isl = graph_label == "prefill"
    elif group == "encoder":
        parts = [f"{export_name}_encoder"]  # export_name already includes the lyr pattern
        include_isl = True
    else:
        parts = [export_name]  # export_name already includes the lyr pattern
        include_isl = True

    if bs:
        parts.append(f"bs{bs}")
    if include_isl and isl:
        parts.append(f"isl{isl}")
    return "_".join(parts) + ".ttir"


def _copy_ttirs(ttir_paths: list[Path], output_dir: Path, group: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for ttir_path in ttir_paths:
        target_name = _normalize_ttir_name(ttir_path, group)
        target_path = output_dir / target_name

        if target_path.exists():
            stem = target_path.stem
            suffix = target_path.suffix
            counter = 2
            while True:
                candidate = output_dir / f"{stem}_v{counter}{suffix}"
                if not candidate.exists():
                    target_path = candidate
                    break
                counter += 1

        shutil.copy2(ttir_path, target_path)
        copied.append(target_path)

    return copied


def _run_pytest(test_file: str, test_name: str, num_layers: int = 1) -> tuple[bool, Optional[str]]:
    """Run a pytest test with num_layers parameter.
    
    Returns:
        (success, error_message) tuple
    """
    test_path = f"{test_file}::{test_name}"
    cmd = [
        sys.executable, "-m", "pytest",
        "-x",  # Exit on first failure
        "-v",  # Verbose
        test_path,
        "--num-layers", str(num_layers),
        "--output-file", "",  # Empty output file
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        
        if result.returncode == 0:
            return True, None
        elif result.returncode == pytest.ExitCode.SKIPPED:
            return False, "Test was skipped"
        else:
            # Extract error message from stderr or stdout
            error_output = result.stderr or result.stdout
            # Check for critical errors
            if any(critical in error_output for critical in CRITICAL_ERRORS):
                _handle_critical_error()
            return False, error_output.split("\n")[-1] if error_output else "Test failed"
    except subprocess.TimeoutExpired:
        return False, "Test timed out"
    except Exception as e:
        return False, str(e)


def _is_test_failed(test_file: Path, test_name: str) -> bool:
    """Check if a test has # FAILED comment above it."""
    if not test_file.exists():
        return False
    
    with open(test_file, "r") as f:
        source = f.read()
        lines = source.splitlines()
    
    # Parse the source to find the test function
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == test_name:
                # Check the line before the function definition
                func_line = node.lineno - 1  # Convert to 0-based index
                if func_line > 0:
                    prev_line = lines[func_line - 1].strip()
                    if prev_line.startswith("# FAILED"):
                        return True
    except SyntaxError:
        # If we can't parse, fall back to simple line search
        pass
    
    return False


def _is_test_completed(test_name: str, output_dir: Path, group: str) -> bool:
    """Check if a test has already been completed by looking for output files."""
    if not output_dir.exists():
        return False
    
    # Extract model name: everything after "test_" is the model name
    # e.g., "test_falcon3_1b" -> "falcon3_1b", "test_llama_3_2_1b_tp" -> "llama_3_2_1b_tp"
    if test_name.startswith("test_"):
        model_name = test_name[5:]  # Remove "test_" prefix
    else:
        model_name = test_name
    
    # Remove "_tp" suffix if present for matching (TP tests may share same base files)
    if model_name.endswith("_tp"):
        model_name_base = model_name[:-3]
    else:
        model_name_base = model_name
    
    # Normalize for comparison: remove underscores and hyphens, convert to lowercase
    model_name_norm = model_name_base.lower().replace("_", "").replace("-", "")
    
    # Look for any .ttir files that might correspond to this test
    # File must contain both the model name and "_1lyr" (or "1lyr") pattern
    for ttir_file in output_dir.glob("*.ttir"):
        stem = ttir_file.stem.lower()
        stem_norm = stem.replace("_", "").replace("-", "")
        # Check if filename contains both model name and 1lyr pattern
        has_model_name = model_name_norm in stem_norm
        has_1lyr = "1lyr" in stem_norm or "_1lyr" in stem.lower()
        if has_model_name and has_1lyr:
            return True
    
    return False


def _run_llm_tests(include_tp: bool, export_path: Path, output_dir: Path, skip_tests: set[str], resume: bool) -> list[dict]:
    import test_llms

    results = []
    tt_xla_dir, _ = _resolve_paths()
    test_file_path = tt_xla_dir / "test_llms.py"
    test_file = str(test_file_path)

    for name, func in inspect.getmembers(test_llms, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if name in {"test_llm", "test_llm_tp"}:
            continue
        if not include_tp and name.endswith("_tp"):
            continue
        
        # Skip if test has # FAILED comment above it
        if _is_test_failed(test_file_path, name):
            print(f"Skipping {name} (has # FAILED comment)")
            results.append(
                {
                    "group": "llm",
                    "test": name,
                    "status": "skipped",
                    "error": "Test marked as FAILED",
                    "ttir": [],
                }
            )
            continue
        
        # Skip if test name matches any skip pattern
        if any(skip_pattern.lower() in name.lower() for skip_pattern in skip_tests):
            print(f"Skipping {name} (matches skip pattern)")
            results.append(
                {
                    "group": "llm",
                    "test": name,
                    "status": "skipped",
                    "error": "Skipped via --skip option",
                    "ttir": [],
                }
            )
            continue
        
        # Skip if already completed (resume mode)
        if resume and _is_test_completed(name, output_dir, "llm"):
            print(f"Skipping {name} (already completed - output file exists)")
            results.append(
                {
                    "group": "llm",
                    "test": name,
                    "status": "skipped",
                    "error": "Already completed (resume mode)",
                    "ttir": [],
                }
            )
            continue

        # Extract model name (everything after "test_")
        model_name = name[5:] if name.startswith("test_") else name
        
        status = "ok"
        error = None
        try:
            success, error_msg = _run_pytest(test_file, name, num_layers=1)
            if not success:
                if error_msg and "num_layers override requested but ModelLoader does not support it" in error_msg:
                    status = "unsupported"
                    error = error_msg
                elif error_msg and "Test was skipped" in error_msg:
                    status = "skipped"
                    error = error_msg
                else:
                    status = "failed"
                    error = error_msg
        except Exception as exc:
            # Check for critical errors - stop execution
            error_msg = str(exc)
            if any(critical in error_msg for critical in CRITICAL_ERRORS):
                _handle_critical_error()
            # Otherwise continue with next test
            status = "failed"
            error = f"{type(exc).__name__}: {error_msg}"
        
        # After test completes, search for files by model name pattern
        ttir_paths = _find_ttirs_by_model(export_path, model_name)
        if len(ttir_paths) > 2:
            ttir_paths = ttir_paths[:2]  # Take most recent 2
        if ttir_paths:
            print(f"Found {len(ttir_paths)} TTIR file(s) for {model_name}: {[p.name for p in ttir_paths]}")
        else:
            print(f"WARNING: No TTIR files found for {model_name} in {export_path}")
        copied_paths = _copy_ttirs(ttir_paths, output_dir, "llm")
        if copied_paths:
            print(f"Copied {len(copied_paths)} file(s) to {output_dir}: {[p.name for p in copied_paths]}")
        results.append(
            {
                "group": "llm",
                "test": name,
                "status": status,
                "error": error,
                "ttir": [str(path) for path in copied_paths],
            }
        )

    return results


def _run_encoder_tests(export_path: Path, output_dir: Path, skip_tests: set[str], resume: bool) -> list[dict]:
    import test_encoders

    results = []
    tt_xla_dir, _ = _resolve_paths()
    test_file_path = tt_xla_dir / "test_encoders.py"
    test_file = str(test_file_path)
    
    for name, func in inspect.getmembers(test_encoders, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if name == "test_encoder":
            continue

        sig = inspect.signature(func)
        if "num_layers" not in sig.parameters:
            continue

        # Skip if test has # FAILED comment above it
        if _is_test_failed(test_file_path, name):
            print(f"Skipping {name} (has # FAILED comment)")
            results.append(
                {
                    "group": "encoder",
                    "test": name,
                    "status": "skipped",
                    "error": "Test marked as FAILED",
                    "ttir": [],
                }
            )
            continue

        # Skip if test name matches any skip pattern
        if any(skip_pattern.lower() in name.lower() for skip_pattern in skip_tests):
            print(f"Skipping {name} (matches skip pattern)")
            results.append(
                {
                    "group": "encoder",
                    "test": name,
                    "status": "skipped",
                    "error": "Skipped via --skip option",
                    "ttir": [],
                }
            )
            continue

        # Skip if already completed (resume mode)
        if resume and _is_test_completed(name, output_dir, "encoder"):
            print(f"Skipping {name} (already completed - output file exists)")
            results.append(
                {
                    "group": "encoder",
                    "test": name,
                    "status": "skipped",
                    "error": "Already completed (resume mode)",
                    "ttir": [],
                }
            )
            continue

        # Extract model name (everything after "test_")
        model_name = name[5:] if name.startswith("test_") else name
        
        status = "ok"
        error = None
        try:
            # Run test via pytest - this automatically handles fixtures like request
            # subprocess.run is blocking, so it waits for the test to complete
            success, error_msg = _run_pytest(test_file, name, num_layers=1)
            if not success:
                if error_msg and "num_layers override requested but ModelLoader does not support it" in error_msg:
                    status = "unsupported"
                    error = error_msg
                elif error_msg and "Test was skipped" in error_msg:
                    status = "skipped"
                    error = error_msg
                else:
                    status = "failed"
                    error = error_msg
        except Exception as exc:
            # Check for critical errors - stop execution
            error_msg = str(exc)
            if any(critical in error_msg for critical in CRITICAL_ERRORS):
                _handle_critical_error()
            # Otherwise continue with next test
            status = "failed"
            error = f"{type(exc).__name__}: {error_msg}"
        
        # After test completes, search for files by model name pattern
        ttir_paths = _find_ttirs_by_model(export_path, model_name)
        if len(ttir_paths) > 1:
            ttir_paths = ttir_paths[:1]  # Take most recent 1
        if ttir_paths:
            print(f"Found {len(ttir_paths)} TTIR file(s) for {model_name}: {[p.name for p in ttir_paths]}")
        else:
            print(f"WARNING: No TTIR files found for {model_name} in {export_path}")
        copied_paths = _copy_ttirs(ttir_paths, output_dir, "encoder")
        if copied_paths:
            print(f"Copied {len(copied_paths)} file(s) to {output_dir}: {[p.name for p in copied_paths]}")
        results.append(
            {
                "group": "encoder",
                "test": name,
                "status": status,
                "error": error,
                "ttir": [str(path) for path in copied_paths],
            }
        )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-layer benchmarks for models that support num_layers.")
    parser.add_argument(
        "--export-path",
        default=None,
        help="Export path for XLA modules (defaults to benchmark/tt-xla/modules).",
    )
    parser.add_argument(
        "--include-tp",
        action="store_true",
        help="Include TP LLM tests (multi-chip) in the run.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save renamed single-layer TTIRs (defaults to benchmark/tt-xla/single_layer_tests).",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume from where left off - skip tests that already have output files in the output directory.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip tests whose names contain the given pattern (case-insensitive). Can be specified multiple times. Example: --skip bert --skip llama",
    )
    args = parser.parse_args()

    tt_xla_dir, repo_root = _resolve_paths()
    _ensure_import_paths(repo_root, tt_xla_dir)
    os.chdir(tt_xla_dir)

    export_path = Path(args.export_path) if args.export_path else tt_xla_dir / "modules"
    export_path = export_path.resolve()

    output_dir = Path(args.output_dir) if args.output_dir else tt_xla_dir / "single_layer_tests"
    output_dir = output_dir.resolve()

    # Parse skip patterns (handle comma-separated values)
    skip_tests = set()
    for skip_arg in args.skip:
        skip_tests.update(s.strip() for s in skip_arg.split(",") if s.strip())

    print(f"\n{'='*60}")
    print(f"One-Layer Benchmark Runner")
    print(f"{'='*60}")
    print(f"Export path:    {export_path}")
    print(f"Output directory: {output_dir}")
    print(f"Test types:     LLM{' + TP' if args.include_tp else ''}, Encoder")
    if skip_tests:
        print(f"Skipping tests matching: {', '.join(skip_tests)}")
    if args.resume:
        print(f"Resume mode:    Enabled (checking {output_dir} for existing files)")
    print(f"{'='*60}\n")

    results = []
    results.extend(_run_llm_tests(args.include_tp, export_path, output_dir, skip_tests, args.resume))
    results.extend(_run_encoder_tests(export_path, output_dir, skip_tests, args.resume))

    print("\nOne-layer benchmark results:")
    for result in results:
        ttir_summary = result["ttir"] if result["ttir"] else ["none"]
        print(f"- {result['group']}::{result['test']} status={result['status']} ttir={ttir_summary}")
        if result["error"]:
            print(f"  error={result['error']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
