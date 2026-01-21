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
OUTPUT_DIR_TTIR = os.path.join(SCRIPT_DIR, "minimal_test_ttir")
OUTPUT_DIR_TTNN = os.path.join(SCRIPT_DIR, "minimal_test_ttnn")
OUTPUT_DIR_SOURCES = os.path.join(SCRIPT_DIR, "minimal_test_sources")
MODULES_DIR = os.path.join(SCRIPT_DIR, "modules/irs")
DUMPED_MODELS_DIR = os.path.join(SCRIPT_DIR, "dumped_models")

# ============================================================================
# Model type configurations
# ============================================================================

LLM_CONFIG = {
    "source_file": "llms.py",
    "required_param": "single_layer",
    "helper_func_name": "test_llm",
    "skip_patterns": ["# FAILED", "# [pytest.skip]"],
    "expected_files": ["_decode_block.mlir", "_prefill_layer.mlir", "_decode_layer.mlir"],
    "tests": [
        {"flag": "--generate-block-test", "mode": "blk", "label": "block", "skip_arg": "skip_block"},
        {"flag": "--generate-layer-test", "mode": "lyr", "label": "layer", "skip_arg": "skip_layer"},
    ],
    # Output naming: {mode: {graph_idx: suffix}}
    "output_names": {
        "blk": {0: "_decode_block.mlir"},
        "lyr": {0: "_prefill_layer.mlir", 1: "_decode_layer.mlir"},
    },
}

ENCODER_CONFIG = {
    "source_file": "encoders.py",
    "required_param": "single_layer",
    "helper_func_name": "test_encoder",
    "skip_patterns": ["# FAILED", "# [pytest.skip]"],
    "expected_files": ["_encoder_block.mlir", "_encoder_layer.mlir"],
    "tests": [
        {"flag": "--generate-block-test", "mode": "blk", "label": "block", "skip_arg": "skip_block"},
        {"flag": "--generate-layer-test", "mode": "lyr", "label": "layer", "skip_arg": "skip_layer"},
    ],
    "output_names": {
        "blk": {0: "_encoder_block.mlir"},
        "lyr": {"largest": "_encoder_layer.mlir"},  # Use largest graph (g0 for Qwen, g1 for BERT)
    },
    # Models that don't support block extraction (only expect layer files)
    "no_block_models": ["qwen_3_embedding_4b"],
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
    "output_names": {
        "blk": {0: "_vision_block.mlir"},
        "lyr": {0: "_vision_layer.mlir"},
    },
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


def check_status_and_filter(models: list, config: dict, status_only: bool) -> list | None:
    """Check model status and optionally filter to incomplete only."""
    complete, incomplete, missing = [], [], {}
    expected_files = config["expected_files"]
    no_block_models = config.get("no_block_models", [])

    for model in models:
        # Adjust expected files for models that don't support block extraction
        if model in no_block_models:
            model_expected = [f for f in expected_files if "block" not in f]
        else:
            model_expected = expected_files

        model_missing = [s for s in model_expected if not os.path.exists(os.path.join(OUTPUT_DIR_TTIR, f"{model}{s}"))]
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
        lines = output.strip().split("\n")
        last_lines = "\n".join(lines[-20:]) if len(lines) > 20 else output
        print(f"ERROR (last 20 lines):\n{last_lines}")
        return False, "Unknown error - see output above"


def find_files(model: str, mode: str, directory: str, prefix: str = "", ext: str = "mlir") -> list[str]:
    """Find files matching pattern, trying with and without underscores in model name."""
    prefix_part = f"{prefix}_" if prefix else ""
    patterns = [
        f"{directory}/{prefix_part}{mode}_{model}*.{ext}",
        f"{directory}/{prefix_part}{mode}_{model.replace('_', '')}*.{ext}",
    ]

    for pattern in patterns:
        if files := glob.glob(pattern):
            return files
    return []


def group_by_graph(files: list[str]) -> dict[int, tuple[int, str]]:
    """Group IR files by graph number, keeping most recent timestamp for each."""
    graph_files = {}
    for f in files:
        if match := re.search(r"_g(\d+)_(\d+)\.mlir$", f):
            g, ts = int(match.group(1)), int(match.group(2))
            if g not in graph_files or ts > graph_files[g][0]:
                graph_files[g] = (ts, f)
    return graph_files


def resolve_graph_key(graph_key, graph_files: dict) -> int | None:
    """Resolve graph key to actual graph number."""
    if graph_key == "largest":
        if not graph_files:
            return None
        return max(graph_files.keys(), key=lambda x: os.path.getsize(graph_files[x][1]))
    else:
        return graph_key if graph_key in graph_files else None


def copy_ir_files(
    model: str,
    mode: str,
    config: dict,
    dry_run: bool,
    prefix: str,
    output_dir: str,
    suffix_transform=None,
    label: str = "",
) -> list[str]:
    """Generic function to copy IR files (TTIR or TTNN) with clean names."""
    files = find_files(model, mode, MODULES_DIR, prefix=prefix, ext="mlir")
    graph_files = group_by_graph(files)

    if not graph_files and prefix == "ttir":
        print(f"WARNING: No {prefix.upper()} files found for {model}")
        return []

    output_names = config.get("output_names", {}).get(mode, {})
    copied = []

    for graph_key, suffix in output_names.items():
        g = resolve_graph_key(graph_key, graph_files)
        if g is None:
            if prefix == "ttir":
                print(f"WARNING: Missing g{graph_key} for {model} {mode}")
            continue

        dst_suffix = suffix_transform(suffix) if suffix_transform else suffix
        dst_name = f"{model}{dst_suffix}"
        src, dst = graph_files[g][1], os.path.join(output_dir, dst_name)

        if dry_run:
            print(f"[DRY-RUN] Would copy {label}: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"Copied {label}: {os.path.basename(src)} -> {dst_name}")
        copied.append(dst)

    return copied


def copy_ttir_files(model: str, mode: str, config: dict, dry_run: bool) -> list[str]:
    """Find and copy generated TTIR files with clean names."""
    return copy_ir_files(model, mode, config, dry_run, "ttir", OUTPUT_DIR_TTIR, label="TTIR")


def copy_ttnn_files(model: str, mode: str, config: dict, dry_run: bool) -> list[str]:
    """Find and copy generated TTNN IR files with clean names."""
    return copy_ir_files(
        model,
        mode,
        config,
        dry_run,
        "ttnn",
        OUTPUT_DIR_TTNN,
        suffix_transform=lambda s: s.replace(".mlir", "_ttnn.mlir"),
        label="TTNN",
    )


def copy_source_files(model: str, mode: str, config: dict, dry_run: bool) -> list[str]:
    """Find and copy dumped source model files with clean names."""
    files = find_files(model, mode, DUMPED_MODELS_DIR, prefix="pt", ext="py")
    if not files:
        return []

    src = sorted(files)[-1]  # Most recent by hash
    output_names = config.get("output_names", {}).get(mode, {})
    copied = []

    for _, suffix in output_names.items():
        dst_name = f"{model}{suffix.replace('.mlir', '.py')}"
        dst = os.path.join(OUTPUT_DIR_SOURCES, dst_name)

        if dry_run:
            print(f"[DRY-RUN] Would copy PT: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"Copied PT: {os.path.basename(src)} -> {dst_name}")
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
                    config["source_file"],
                    model,
                    test["flag"],
                    args.dry_run,
                    dump_source=True,
                )

            if success:
                copied = copy_ttir_files(model, test["mode"], config, args.dry_run)
                if copied:
                    results["success"].extend(copied)
                else:
                    results["skipped"].append(f"{model}_{test['label']}")

                # Also copy TTNN and source files (sources only if dump_source enabled)
                copy_ttnn_files(model, test["mode"], config, args.dry_run)
                copy_source_files(model, test["mode"], config, args.dry_run)
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
        print(f"\nAll files saved to:")
        print(f"  TTIR:    {OUTPUT_DIR_TTIR}/")
        print(f"  TTNN:    {OUTPUT_DIR_TTNN}/")
        print(f"  Sources: {OUTPUT_DIR_SOURCES}/")


def run(model_types: list[str], args) -> int:
    """Main entry point."""
    output_dirs = [OUTPUT_DIR_TTIR, OUTPUT_DIR_TTNN, OUTPUT_DIR_SOURCES]
    if args.clean_dir:
        for d in output_dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"Cleared: {d}")

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Output directories: {', '.join(os.path.basename(d) for d in output_dirs)}")

    all_results = {"success": [], "failed": [], "skipped": []}
    show_headers = len(model_types) > 1

    for model_type in model_types:
        config = MODEL_CONFIGS[model_type]

        if show_headers:
            print(f"\n{'#'*60}\n# {model_type.upper()}s\n{'#'*60}")

        models = parse_models(args.models, discover_models(config))

        if args.continue_mode or args.status_only:
            models = check_status_and_filter(models, config, args.status_only)
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

    for cmd, help_text in [
        ("llm", "Generate tests for LLM models"),
        ("encoder", "Generate tests for encoder models"),
        ("vision", "Generate tests for vision transformers"),
        ("all", "Generate tests for all transformer models"),
    ]:
        p = subparsers.add_parser(cmd, help=help_text)
        p.add_argument("--models", help="Comma-separated models or prefixes")
        p.add_argument("--dry-run", action="store_true", help="Print without running")
        p.add_argument("--clean-dir", action="store_true", help="Clear output directory first")
        p.add_argument("--copy-only", action="store_true", help="Only copy existing TTIR files")
        p.add_argument("--continue", dest="continue_mode", action="store_true", help="Resume incomplete")
        p.add_argument("--status-only", action="store_true", help="Only show progress")
        p.add_argument("--skip-block", action="store_true", help="Skip block tests")
        p.add_argument("--skip-layer", action="store_true", help="Skip layer tests")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "all":
        model_types = ["llm", "encoder", "vision"]
    else:
        model_types = [args.command]

    return run(model_types, args)


if __name__ == "__main__":
    sys.exit(main())
