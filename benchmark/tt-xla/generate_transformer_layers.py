#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate single-layer TTIR tests for transformer models (LLMs, encoders, vision transformers).

Usage:
    python generate_transformer_layers.py llm [--models MODEL1,MODEL2,...] [--dry-run]
    python generate_transformer_layers.py encoder [--models MODEL1,MODEL2,...] [--dry-run]
    python generate_transformer_layers.py vision [--models MODEL1,MODEL2,...] [--dry-run]
    python generate_transformer_layers.py all [--dry-run]

Examples:
    python generate_transformer_layers.py llm --models phi,gemma
    python generate_transformer_layers.py encoder --models bert
    python generate_transformer_layers.py vision --models vit,swin
    python generate_transformer_layers.py all
    python generate_transformer_layers.py llm --continue    # Resume incomplete
    python generate_transformer_layers.py --status-only     # Show progress
"""

import argparse
import ast
import glob
import os
import re
import shutil
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "transformer_layers")
MODULES_DIR = os.path.join(SCRIPT_DIR, "modules/irs")

# ============================================================================
# Model type configurations
# ============================================================================

LLM_CONFIG = {
    "source_file": "llms.py",
    "required_param": "single_layer",
    "helper_func_name": "test_llm",
    "skip_patterns": ["# FAILED"],
    "expected_files": ["_decode_block.mlir", "_prefill_layer.mlir", "_decode_layer.mlir"],
    "tests": [
        {"flag": "--generate-block-test", "mode": "blk", "label": "block", "skip_arg": "skip_block"},
        {"flag": "--generate-layer-test", "mode": "lyr", "label": "layer", "skip_arg": "skip_layer"},
    ],
}

ENCODER_CONFIG = {
    "source_file": "encoders.py",
    "required_param": "single_layer",
    "helper_func_name": "test_encoder",
    "skip_patterns": ["# FAILED", "# [pytest.skip]"],
    "expected_files": ["_encoder_layer.mlir"],
    "tests": [
        {"flag": "--generate-layer-test", "mode": "lyr", "label": "layer", "skip_arg": None},
    ],
}

VISION_CONFIG = {
    "source_file": "vision_models.py",
    "required_param": "single_layer",
    "helper_func_name": "test_vision",
    "skip_patterns": ["# FAILED", "# [pytest.skip]"],
    "expected_files": ["_vision_block.mlir", "_vision_layer.mlir"],
    "tests": [
        {"flag": "--generate-block-test", "mode": "blk", "label": "block", "skip_arg": "skip_block"},
        {"flag": "--generate-layer-test", "mode": "lyr", "label": "layer", "skip_arg": "skip_layer"},
    ],
}

MODEL_CONFIGS = {
    "llm": LLM_CONFIG,
    "encoder": ENCODER_CONFIG,
    "vision": VISION_CONFIG,
}

# ============================================================================
# Utilities
# ============================================================================


def discover_models(config: dict) -> list[str]:
    """Discover test models from source file based on config."""
    source_path = os.path.join(SCRIPT_DIR, config["source_file"])
    with open(source_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    lines = source.splitlines()
    skip_patterns = config.get("skip_patterns", ["# FAILED"])

    models = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        if node.name == config["helper_func_name"]:
            continue

        param_names = [arg.arg for arg in node.args.args]
        if config["required_param"] not in param_names:
            continue

        # Check if preceding line has skip pattern
        func_line = node.lineno - 1
        if func_line > 0:
            prev_line = lines[func_line - 1].strip()
            if any(prev_line.startswith(p) for p in skip_patterns):
                continue

        models.append(node.name[5:])  # Remove "test_" prefix

    return models


def parse_models(models_arg: str | None, all_models: list[str]) -> list[str]:
    """Parse --models argument. Supports exact names or prefixes."""
    if not models_arg:
        return all_models

    models = []
    for pattern in models_arg.split(","):
        pattern = pattern.strip()
        if pattern in all_models:
            models.append(pattern)
        else:
            matches = [m for m in all_models if m.startswith(pattern)]
            if matches:
                models.extend(matches)
            else:
                print(f"ERROR: No models match '{pattern}'. Available: {', '.join(all_models)}")
                sys.exit(1)

    return list(dict.fromkeys(models))  # Remove duplicates, preserve order


def check_status_and_filter(models: list, expected_files: list[str], status_only: bool) -> list | None:
    """Check model status and optionally filter to incomplete only."""
    complete, incomplete, missing = [], [], {}

    for model in models:
        model_missing = [s for s in expected_files if not os.path.exists(os.path.join(OUTPUT_DIR, f"{model}{s}"))]
        if model_missing:
            incomplete.append(model)
            missing[model] = model_missing
        else:
            complete.append(model)

    # Print status report
    total = len(models)
    pct = 100 * len(complete) // total if total else 0
    print(f"\n{'='*60}\nPROGRESS REPORT\n{'='*60}")
    print(f"Complete: {len(complete)}/{total} ({pct}%) | Incomplete: {len(incomplete)}/{total}")

    if complete:
        print(f"\n✓ Complete ({len(complete)}): {', '.join(complete[:10])}{'...' if len(complete) > 10 else ''}")
    if incomplete:
        print(f"\n✗ Incomplete ({len(incomplete)}):")
        for m in incomplete[:20]:
            short = [s.replace(".mlir", "").replace("_", " ").strip() for s in missing[m]]
            print(f"    {m}: {', '.join(short)}")
        if len(incomplete) > 20:
            print(f"    ... and {len(incomplete) - 20} more")
    print(f"{'='*60}\n")

    if status_only:
        return None
    if not incomplete:
        print("All models complete! Nothing to do.")
        return None

    print(f"Continuing with {len(incomplete)} incomplete model(s)...\n")
    return incomplete


def execute_pytest(test_file: str, model: str, flag: str, dry_run: bool, dump_source: bool = False) -> tuple[bool, str]:
    """Execute pytest for a specific test."""
    cmd = [sys.executable, "-m", "pytest", "-svv", f"{test_file}::test_{model}", flag]
    if dump_source:
        cmd.append("--dump-source")
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Running: {' '.join(cmd)}")

    if dry_run:
        return True, ""

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=SCRIPT_DIR)
        if result.stdout:
            print(result.stdout)
        return True, ""
    except subprocess.CalledProcessError as e:
        output = (e.stdout or "") + "\n" + (e.stderr or "")
        # Try to find specific error patterns with more context
        for pattern in [
            r"(TT_FATAL[^\n]+(?:\n[^\n]*){0,5})",
            r"((?:AssertionError|RuntimeError|ValueError|TypeError|KeyError|AttributeError|ModuleNotFoundError)[:\s]+[^\n]+(?:\n[^\n]*){0,5})",
            r"(FAILED[^\n]+)",
            r"(E\s+[A-Z]\w+Error[:\s]+[^\n]+(?:\n[^\n]*){0,3})",
        ]:
            if match := re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
                error = match.group(1).strip()[:500]
                print(f"ERROR: {error}")
                return False, error
        # If no pattern matched, show last 20 lines
        lines = output.strip().split('\n')
        last_lines = '\n'.join(lines[-20:]) if len(lines) > 20 else output
        print(f"ERROR (last 20 lines):\n{last_lines}")
        return False, "Unknown error - see output above"


def copy_ttir_files(model: str, mode: str, model_type: str, dry_run: bool) -> list[str]:
    """Find and copy generated TTIR files with clean names."""
    # Try multiple patterns: exact match, without underscores, with underscores removed before digits
    patterns = [
        f"{MODULES_DIR}/ttir_{mode}_{model}*_bs*_g*.mlir",
        f"{MODULES_DIR}/ttir_{mode}_{model.replace('_', '')}*_bs*_g*.mlir",  # phi_1 -> phi1
    ]

    all_files = []
    for pattern in patterns:
        all_files = glob.glob(pattern)
        if all_files:
            break

    if not all_files:
        print(f"WARNING: No TTIR files found matching patterns for {model}")
        return []

    # Group by graph number, keep most recent
    graph_files = {}
    for f in all_files:
        if match := re.search(r"_g(\d+)_(\d+)\.mlir$", f):
            g, ts = int(match.group(1)), int(match.group(2))
            if g not in graph_files or ts > graph_files[g][0]:
                graph_files[g] = (ts, f)

    # Determine output names based on model type
    if mode == "blk":
        if model_type == "vision":
            graph_names = {0: f"{model}_vision_block.mlir"}
        else:
            graph_names = {0: f"{model}_decode_block.mlir"}
    elif model_type == "encoder":
        graph_names = {0: f"{model}_encoder_layer.mlir"}
    elif model_type == "vision":
        graph_names = {0: f"{model}_vision_layer.mlir"}
    else:  # llm layer
        graph_names = {0: f"{model}_prefill_layer.mlir", 1: f"{model}_decode_layer.mlir"}

    copied = []
    for g, dst_name in graph_names.items():
        if g not in graph_files:
            print(f"WARNING: Missing g{g} for {model} {mode}")
            continue

        src, dst = graph_files[g][1], os.path.join(OUTPUT_DIR, dst_name)
        if dry_run:
            print(f"[DRY-RUN] Would copy: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
        copied.append(dst)

    return copied


# ============================================================================
# Main logic
# ============================================================================


def run_model_tests(model_type: str, models: list[str], args) -> dict:
    """Run tests for models of a given type."""
    config = MODEL_CONFIGS[model_type]
    results = {"success": [], "failed": [], "skipped": []}

    for model in models:
        print(f"\n{'='*60}\nProcessing: {model} ({model_type})\n{'='*60}")

        for test in config["tests"]:
            if test["skip_arg"] and getattr(args, test["skip_arg"], False):
                continue

            if args.copy_only:
                success, error = True, ""
            else:
                success, error = execute_pytest(
                    config["source_file"], model, test["flag"], args.dry_run,
                    dump_source=getattr(args, "dump_source", False)
                )

            if success:
                copied = copy_ttir_files(model, test["mode"], model_type, args.dry_run)
                if copied:
                    results["success"].extend(copied)
                else:
                    results["skipped"].append(f"{model}_{test['label']}")
            else:
                results["failed"].append((f"{model}_{test['label']}", error))

    return results


def print_summary(results: dict):
    """Print test results summary."""
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Success: {len(results['success'])} files")
    for f in results["success"]:
        print(f"  ✓ {f}")

    if results["skipped"]:
        print(f"\n⚠ Not found: {len(results['skipped'])} (check modules/irs/ for naming)")
        for f in results["skipped"]:
            print(f"  ⊘ {f}")

    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for name, err in results["failed"]:
            print(f"  ✗ {name}: {err}")
    else:
        print(f"\nAll files saved to: {OUTPUT_DIR}/")


def run(model_types: list[str], args) -> int:
    """Main entry point."""
    if args.clean_dir and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleared: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")

    all_results = {"success": [], "failed": [], "skipped": []}
    show_headers = len(model_types) > 1

    for model_type in model_types:
        config = MODEL_CONFIGS[model_type]

        if show_headers:
            print(f"\n{'#'*60}\n# {model_type.upper()}s\n{'#'*60}")

        models = parse_models(args.models, discover_models(config))

        if args.continue_mode or args.status_only:
            models = check_status_and_filter(models, config["expected_files"], args.status_only)
            if models is None:
                continue

        results = run_model_tests(model_type, models, args)
        for key in all_results:
            all_results[key].extend(results[key])

    if not args.status_only:
        print_summary(all_results)

    return 1 if all_results["failed"] else 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-layer TTIR tests for transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_transformer_layers.py llm                     # All LLMs
    python generate_transformer_layers.py llm --models phi,gemma  # Phi and Gemma
    python generate_transformer_layers.py encoder                 # All encoders
    python generate_transformer_layers.py vision                  # Vision transformers
    python generate_transformer_layers.py all                     # All transformers
    python generate_transformer_layers.py llm --continue          # Resume incomplete
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Model type to process")

    for cmd, help_text, has_block in [
        ("llm", "Generate tests for LLM models", True),
        ("encoder", "Generate tests for encoder models", False),
        ("vision", "Generate tests for vision transformers", True),
        ("all", "Generate tests for all transformer models", True),
    ]:
        p = subparsers.add_parser(cmd, help=help_text)
        p.add_argument("--models", help="Comma-separated models or prefixes")
        p.add_argument("--dry-run", action="store_true", help="Print without running")
        p.add_argument("--clean-dir", action="store_true", help="Clear output directory first")
        p.add_argument("--copy-only", action="store_true", help="Only copy existing TTIR files")
        p.add_argument("--continue", dest="continue_mode", action="store_true", help="Resume incomplete")
        p.add_argument("--status-only", action="store_true", help="Only show progress")
        p.add_argument("--dump-source", action="store_true", help="Export source model before TT compilation")
        if has_block:
            p.add_argument("--skip-block", action="store_true", help="Skip block tests")
            p.add_argument("--skip-layer", action="store_true", help="Skip layer tests")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Set defaults for models without block tests
    if args.command == "encoder":
        args.skip_block = True
        args.skip_layer = False

    if args.command == "all":
        model_types = ["llm", "encoder", "vision"]
    else:
        model_types = [args.command]

    return run(model_types, args)


if __name__ == "__main__":
    sys.exit(main())
