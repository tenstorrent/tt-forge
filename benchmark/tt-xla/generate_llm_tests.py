#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate block and layer TTIR tests for all benchmarked LLMs.

This script:
1. Runs --generate-block-test for each LLM (decoder block only)
2. Runs --generate-layer-test for each LLM (prefill g0 + decode g1)
3. Copies TTIRs to llm_blocks_and_layers/ with clean names:
   - {model}_block.mlir
   - {model}_prefill_layer.mlir
   - {model}_decode_layer.mlir

Usage:
    python generate_llm_tests.py [--models MODEL1,MODEL2,...] [--dry-run]

Example:
    python generate_llm_tests.py --models phi1,phi2
    python generate_llm_tests.py --models gemma  # Run all gemma models
    python generate_llm_tests.py --models qwen   # Run all qwen models
    python generate_llm_tests.py  # Run all models
"""

import argparse
import ast
import glob
import os
import re
import shutil
import subprocess
import sys


def discover_test_models():
    """Discover LLM test models from llms.py that support single_block/single_layer.
    
    Filters out tests that have '# FAILED' comments before them.
    """
    llms_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llms.py")
    with open(llms_path, "r") as f:
        source = f.read()
    
    # Parse AST and keep raw lines for comment checking
    tree = ast.parse(source)
    lines = source.splitlines()
    
    models = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # Skip the main test_llm function (it's a helper, not a model test)
            if node.name == "test_llm":
                continue
            # Check if function has single_block parameter
            param_names = [arg.arg for arg in node.args.args]
            if "single_block" in param_names:
                # Check if preceding line contains "# FAILED"
                func_line = node.lineno - 1  # 0-indexed
                is_failed = False
                if func_line > 0:
                    prev_line = lines[func_line - 1].strip()
                    if prev_line.startswith("# FAILED"):
                        is_failed = True
                
                if not is_failed:
                    # Extract model name by removing "test_" prefix
                    model_name = node.name[5:]  # Remove "test_"
                    models.append(model_name)
    return models


# Auto-discover models from llms.py that support single_block/single_layer
ALL_MODELS = discover_test_models()

BLOCK_MODELS = ALL_MODELS
LAYER_MODELS = ALL_MODELS

# Get the directory where this script lives (tt-forge/benchmark/tt-xla)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for output (in script directory)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "llm_blocks_and_layers")
MODULES_DIR = os.path.join(SCRIPT_DIR, "modules/irs")


def run_pytest(test_name: str, flag: str, dry_run: bool = False) -> tuple[bool, str]:
    """Run pytest for a specific test with the given flag.
    
    Runs pytest from the script's directory to ensure conftest.py is found.
    
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    cmd = [
        sys.executable, "-m", "pytest", "-svv",
        f"llms.py::test_{test_name}",
        flag,
    ]
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Running: {' '.join(cmd)}")
    print(f"  (from directory: {SCRIPT_DIR})")
    
    if dry_run:
        return True, ""
    
    try:
        # Run pytest from the script's directory so conftest.py is found
        # Capture output so we can extract error details on failure
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=SCRIPT_DIR
        )
        # Print output on success too (so user sees progress)
        if result.stdout:
            print(result.stdout)
        return True, ""
    except subprocess.CalledProcessError as e:
        # Extract the most relevant error info from output
        error_msg = extract_error_from_output(e.stdout or "", e.stderr or "")
        print(f"ERROR: Test {test_name} with {flag} failed!")
        if error_msg:
            print(f"  Reason: {error_msg}")
        return False, error_msg


def extract_error_from_output(stdout: str, stderr: str) -> str:
    """Extract the most relevant error message from pytest output."""
    combined = stdout + "\n" + stderr
    
    # Look for common error patterns (most specific first)
    patterns = [
        # Python exceptions with message
        r'((?:AssertionError|RuntimeError|ValueError|TypeError|AttributeError|KeyError|ModuleNotFoundError|ImportError|FileNotFoundError|NameError|IndexError|NotImplementedError)[:\s]+[^\n]+)',
        # FAILED line from pytest
        r'(FAILED[^\n]+)',
        # Error with traceback context
        r'(E\s+[A-Z][a-zA-Z]+Error[:\s]+[^\n]+)',
        # Any line starting with "Error:"
        r'(Error:[^\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, combined, re.IGNORECASE)
        if match:
            error = match.group(1).strip()
            # Truncate if too long
            if len(error) > 200:
                error = error[:200] + "..."
            return error
    
    # If no pattern matched, get last non-empty lines from stderr
    lines = [l.strip() for l in combined.split('\n') if l.strip()]
    if lines:
        # Return last few lines that might contain the error
        return " | ".join(lines[-3:])[:200]
    
    return "Unknown error (check full output above)"


def find_and_copy_ttir(model_name: str, mode: str, output_dir: str, dry_run: bool = False) -> list:
    """Find generated TTIR files and copy them with clean names.
    
    Args:
        model_name: e.g., "phi1", "falcon3_1b" (test function name without "test_")
        mode: "blk" for block, "lyr" for layer
        output_dir: destination directory
        dry_run: if True, only print what would be done
    
    Returns:
        List of copied file paths
    """
    copied_files = []
    
    # Pattern: ttir_{mode}_{model}*_bs{batch}_{runid}_g{N}_{timestamp}.mlir
    # Use * after model_name to catch variants like llama_3_2_1b_instruct
    pattern = f"{MODULES_DIR}/ttir_{mode}_{model_name}*_bs*_g*.mlir"
    all_files = glob.glob(pattern)
    
    # Fallback: model variant name may differ from test name (e.g., falcon3_1b vs falcon_1b)
    # Try without version numbers in the name
    if not all_files:
        # Try stripping digits between name parts: falcon3_1b -> falcon*1b
        alt_pattern = re.sub(r'(\d+)_', r'*', model_name, count=1)
        if alt_pattern != model_name:
            pattern = f"{MODULES_DIR}/ttir_{mode}_{alt_pattern}*_bs*_g*.mlir"
            all_files = glob.glob(pattern)
            if all_files:
                print(f"  (matched with alternate pattern: {alt_pattern})")
    
    if not all_files:
        print(f"WARNING: No TTIR files found matching {pattern}")
        return copied_files
    
    # Group files by graph number, keep only the most recent (highest timestamp)
    graph_files = {}  # graph_num -> (timestamp, filepath)
    for f in all_files:
        match = re.search(r'_g(\d+)_(\d+)\.mlir$', f)
        if not match:
            continue
        graph_num = int(match.group(1))
        timestamp = int(match.group(2))
        
        if graph_num not in graph_files or timestamp > graph_files[graph_num][0]:
            graph_files[graph_num] = (timestamp, f)
    
    # For block mode, only need g0; for layer mode, only need g0 and g1
    if mode == "blk":
        wanted_graphs = [0]
    else:
        wanted_graphs = [0, 1]
    
    for graph_num in wanted_graphs:
        if graph_num not in graph_files:
            print(f"WARNING: Missing g{graph_num} for {model_name} {mode}")
            continue
        
        src_file = graph_files[graph_num][1]
        
        # Determine output name
        if mode == "blk":
            dst_name = f"{model_name}_decode_block.mlir"
        else:
            if graph_num == 0:
                dst_name = f"{model_name}_prefill_layer.mlir"
            else:
                dst_name = f"{model_name}_decode_layer.mlir"
        
        dst_file = os.path.join(output_dir, dst_name)
        
        if dry_run:
            print(f"[DRY-RUN] Would copy: {src_file} -> {dst_file}")
        else:
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")
        
        copied_files.append(dst_file)
    
    return copied_files


def main():
    parser = argparse.ArgumentParser(description="Generate block and layer TTIR tests for LLMs")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated list of models or prefixes (e.g., 'gemma' matches all gemma models). Available: {','.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually running tests"
    )
    parser.add_argument(
        "--skip-block",
        action="store_true",
        help="Skip block tests (--generate-block-test)"
    )
    parser.add_argument(
        "--skip-layer",
        action="store_true",
        help="Skip layer tests (--generate-layer-test)"
    )
    parser.add_argument(
        "--clean-dir",
        action="store_true",
        help="Clear output directory (llm_blocks_and_layers) before running"
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Skip pytest, only find and copy existing TTIR files from modules/irs/"
    )
    args = parser.parse_args()
    
    # Parse models - supports exact names or prefixes (e.g., "gemma" matches all gemma models)
    if args.models:
        patterns = [m.strip() for m in args.models.split(",")]
        models = []
        for pattern in patterns:
            # Check for exact match first
            if pattern in ALL_MODELS:
                models.append(pattern)
            else:
                # Try prefix match
                matches = [m for m in ALL_MODELS if m.startswith(pattern)]
                if matches:
                    models.extend(matches)
                else:
                    print(f"ERROR: No models match '{pattern}'. Available: {', '.join(ALL_MODELS)}")
                    sys.exit(1)
        # Remove duplicates while preserving order
        models = list(dict.fromkeys(models))
    else:
        models = ALL_MODELS
    
    # Clear output directory if requested
    if args.clean_dir and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleared output directory: {OUTPUT_DIR}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Track results
    results = {"success": [], "failed": [], "skipped": []}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Processing: {model}")
        print(f"{'='*60}")
        
        # Run block test
        if not args.skip_block:
            if args.copy_only:
                # Skip pytest, just copy existing files
                success = True
            else:
                success, error_msg = run_pytest(model, "--generate-block-test", args.dry_run)
            if success:
                copied = find_and_copy_ttir(model, "blk", OUTPUT_DIR, args.dry_run)
                if copied:
                    results["success"].extend(copied)
                else:
                    results["skipped"].append(f"{model}_block")
            else:
                results["failed"].append((f"{model}_block", error_msg))
        
        # Run layer test
        if not args.skip_layer:
            if args.copy_only:
                # Skip pytest, just copy existing files
                success = True
            else:
                success, error_msg = run_pytest(model, "--generate-layer-test", args.dry_run)
            if success:
                copied = find_and_copy_ttir(model, "lyr", OUTPUT_DIR, args.dry_run)
                if copied:
                    results["success"].extend(copied)
                else:
                    results["skipped"].append(f"{model}_layer")
            else:
                results["failed"].append((f"{model}_layer", error_msg))
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Success: {len(results['success'])} files")
    for f in results["success"]:
        print(f"  ✓ {f}")
    
    if results["skipped"]:
        print(f"\n⚠ Files not found: {len(results['skipped'])} tests (test passed but TTIR files not matched)")
        print(f"  This is likely a naming mismatch - check modules/irs/ for actual filenames")
        for f in results["skipped"]:
            print(f"  ⊘ {f}")
    
    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])} tests")
        for test_name, error_msg in results["failed"]:
            print(f"  ✗ {test_name}")
            if error_msg:
                print(f"      → {error_msg}")
        sys.exit(1)
    
    print(f"\nAll files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

