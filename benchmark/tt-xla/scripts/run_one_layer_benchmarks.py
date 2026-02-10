#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
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

CRITICAL_ERRORS = [
    "Read unexpected run_mailbox value from core",
    "Timeout waiting for Ethernet core service remote IO request",
]


def _handle_critical_error() -> None:
    print(
        f"\n{'='*60}\n"
        f"DEVICE ERROR: TT device needs reset\n"
        f"{'='*60}\n"
        f"\n"
        f"  >>> Run: tt-smi -r\n"
        f"  >>> Re-run with --continue to resume.\n"
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


def _find_mlirs_by_model(
    export_path: Path, model_name: str, prefix: str, min_mtime: Optional[float] = None
) -> list[Path]:
    if not export_path.exists():
        return []

    name_pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(model_name)}_1lyr_bs\d+_isl\d+")
    matches = []
    for path in export_path.rglob("*.mlir"):
        if not name_pattern.search(path.stem):
            continue
        if min_mtime is not None and path.stat().st_mtime < min_mtime:
            continue
        matches.append(path)
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)


def _parse_mlir_stem(stem: str, model_name: str, prefix: str) -> tuple[Optional[str], Optional[str]]:
    match = re.search(rf"^{re.escape(prefix)}_{re.escape(model_name)}_1lyr_bs(\d+)_isl(\d+)", stem)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _graph_label_from_stem(stem: str) -> str:
    match = re.search(r"(?:^|_)(g[01])(?:_|$)", stem)
    if match and match.group(1) == "g0":
        return "prefill"
    if match and match.group(1) == "g1":
        return "decode"
    return "graph"


def _scan_output_dir(output_dir: Path, model_key: str) -> tuple[bool, bool, bool]:
    found_prefill = False
    found_decode = False
    found_encoder = False
    if not output_dir.exists():
        return found_prefill, found_decode, found_encoder

    for mlir_file in output_dir.glob("*.mlir"):
        stem = mlir_file.stem.lower()
        has_model_name = stem.startswith(f"{model_key}_")
        has_1lyr = "1lyr" in stem
        if not (has_model_name and has_1lyr):
            continue
        if "encoder" in stem:
            found_encoder = True
        if "prefill" in stem:
            found_prefill = True
        if "decode" in stem:
            found_decode = True

    return found_prefill, found_decode, found_encoder


def _normalize_mlir_name(
    mlir_path: Path,
    group: str,
    model_name: str,
    prefix: str,
    graph_label_override: Optional[str] = None,
) -> str:
    bs, isl = _parse_mlir_stem(mlir_path.stem, model_name, prefix)
    assert bs is not None and isl is not None, f"Expected bs/isl in {mlir_path.stem}"
    base = f"{model_name}_1lyr_bs{bs}"
    graph_label = graph_label_override or _graph_label_from_stem(mlir_path.stem)
    if group == "encoder":
        suffix = f"encoder_isl{isl}"
    elif graph_label == "decode":
        suffix = "decode"
    else:
        suffix = f"prefill_isl{isl}"
    ttnn_suffix = "_ttnn" if prefix == "ttnn" else ""
    return f"{base}_{suffix}{ttnn_suffix}.mlir"


def _copy_mlirs(
    mlir_paths: list[Path],
    output_dir: Path,
    group: str,
    model_name: str,
    prefix: str,
    graph_labels: Optional[dict[Path, str]] = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for mlir_path in mlir_paths:
        graph_label_override = graph_labels.get(mlir_path) if graph_labels else None
        target_name = _normalize_mlir_name(mlir_path, group, model_name, prefix, graph_label_override)
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

        shutil.copy2(mlir_path, target_path)
        copied.append(target_path)

    return copied


def _extract_run_id(ttir_stem: str) -> Optional[str]:
    match = re.search(r"_run([0-9a-zA-Z]+)", ttir_stem)
    return match.group(1) if match else None


def _select_tp_ttirs(ttir_paths: list[Path]) -> tuple[list[Path], dict[Path, str]]:
    if not ttir_paths:
        return [], {}

    run_groups: dict[Optional[str], list[Path]] = {}
    for path in ttir_paths:
        run_id = _extract_run_id(path.stem)
        run_groups.setdefault(run_id, []).append(path)

    def _group_mtime(paths: list[Path]) -> float:
        return max(p.stat().st_mtime for p in paths)

    selected_run_id = max(run_groups.items(), key=lambda item: _group_mtime(item[1]))[0]
    candidates = run_groups[selected_run_id]
    candidates_sorted = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
    selected = candidates_sorted[:2]

    labels: dict[Path, str] = {}
    if len(selected) == 2:
        selected_by_time = sorted(selected, key=lambda p: p.stat().st_mtime)
        labels[selected_by_time[0]] = "prefill"
        labels[selected_by_time[1]] = "decode"

    return selected, labels


def _run_pytest(test_file: str, test_name: str, num_layers: int = 1) -> tuple[bool, Optional[str], Optional[str]]:
    test_path = f"{test_file}::{test_name}"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-v",
        test_path,
        "--num-layers",
        str(num_layers),
        "--output-file",
        "",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "Test timed out", None
    except Exception as exc:
        error_msg = str(exc)
        if any(critical in error_msg for critical in CRITICAL_ERRORS):
            _handle_critical_error()
        return False, f"{type(exc).__name__}: {error_msg}", None

    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""
    output = "\n".join(text for text in (stdout_text, stderr_text) if text)
    if result.returncode == 0:
        return True, None, None
    skipped_code = getattr(pytest.ExitCode, "SKIPPED", 5)
    if result.returncode == skipped_code:
        return False, "Test was skipped", None
    if any(critical in output for critical in CRITICAL_ERRORS):
        _handle_critical_error()
    output_lines = [line for line in output.split("\n") if line.strip()]
    tail = "\n".join(output_lines[-20:]) if output_lines else None
    last_line = output_lines[-1] if output_lines else "Test failed"
    return False, last_line, tail


def _is_test_failed(test_file: Path, test_name: str) -> bool:
    if not test_file.exists():
        return False

    lines = test_file.read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith(f"def {test_name}("):
            prev_line = lines[idx - 1].strip() if idx > 0 else ""
            return prev_line.startswith("# FAILED")
    return False


def _is_test_completed(test_name: str, output_dir: Path, group: str, ttnn_dir: Optional[Path] = None) -> bool:
    completed, _missing = _get_output_status(test_name, output_dir, group, ttnn_dir)
    return completed


def _get_output_status(
    test_name: str, output_dir: Path, group: str, ttnn_dir: Optional[Path] = None
) -> tuple[bool, list[str]]:
    expected = ["encoder"] if group == "encoder" else ["prefill", "decode"]
    if not output_dir.exists():
        missing = expected.copy()
        if ttnn_dir is not None and not ttnn_dir.exists():
            missing.extend(f"ttnn_{item}" for item in expected)
        return False, missing

    model_name = _model_name_from_test(test_name)
    model_key = model_name.lower()

    found_prefill, found_decode, found_encoder = _scan_output_dir(output_dir, model_key)

    missing = []
    if group == "encoder":
        if not found_encoder:
            missing.append("encoder")
    else:
        if not found_prefill:
            missing.append("prefill")
        if not found_decode:
            missing.append("decode")

    if ttnn_dir is not None:
        if not ttnn_dir.exists():
            missing.extend(f"ttnn_{item}" for item in expected)
        else:
            found_ttnn_prefill, found_ttnn_decode, found_ttnn_encoder = _scan_output_dir(ttnn_dir, model_key)

            if group == "encoder":
                if not found_ttnn_encoder:
                    missing.append("ttnn_encoder")
            else:
                if not found_ttnn_prefill:
                    missing.append("ttnn_prefill")
                if not found_ttnn_decode:
                    missing.append("ttnn_decode")

    return len(missing) == 0, missing


def _parse_prefix_patterns(prefix_args: list[str]) -> set[str]:
    prefixes: set[str] = set()
    for prefix_arg in prefix_args:
        prefixes.update(s.strip() for s in prefix_arg.split(",") if s.strip())
    return prefixes


def _model_name_from_test(test_name: str) -> str:
    return test_name[5:] if test_name.startswith("test_") else test_name


def _matches_prefixes(test_name: str, include_prefixes: set[str]) -> bool:
    if not include_prefixes:
        return True
    model_name = _model_name_from_test(test_name)
    return any(model_name.lower().startswith(prefix.lower()) for prefix in include_prefixes)


def _test_names_llm(include_tp: bool, tp_only: bool) -> list[str]:
    import test_llms

    names = []
    for name, _func in inspect.getmembers(test_llms, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if name in {"test_llm", "test_llm_tp"}:
            continue
        if tp_only and not name.endswith("_tp"):
            continue
        if not include_tp and name.endswith("_tp"):
            continue
        names.append(name)
    return names


def _test_names_encoder() -> list[str]:
    import test_encoders

    names = []
    for name, func in inspect.getmembers(test_encoders, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        if name == "test_encoder":
            continue
        sig = inspect.signature(func)
        if "num_layers" not in sig.parameters:
            continue
        names.append(name)
    return names


def _run_tests(
    group: str,
    test_file_path: Path,
    test_names: list[str],
    export_path: Path,
    output_dir: Path,
    ttnn_output_dir: Path,
    include_prefixes: set[str],
    resume: bool,
    max_ttirs: int,
) -> list[dict]:
    test_file = str(test_file_path)
    results = []

    for name in test_names:
        if _is_test_failed(test_file_path, name):
            continue

        if not _matches_prefixes(name, include_prefixes):
            continue

        if resume and _is_test_completed(name, output_dir, group, ttnn_output_dir):
            continue

        print(f"🚀 Starting {group}::{name}")

        model_name = _model_name_from_test(name)
        status = "ok"
        error = None
        test_start_time = time.time()
        success, error_msg, error_detail = _run_pytest(test_file, name, num_layers=1)
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

        ttir_paths = _find_mlirs_by_model(export_path, model_name, "ttir", min_mtime=test_start_time)
        ttnn_paths = _find_mlirs_by_model(export_path, model_name, "ttnn", min_mtime=test_start_time)
        graph_labels = None
        ttnn_graph_labels = None
        if model_name.endswith("_tp"):
            ttir_paths, graph_labels = _select_tp_ttirs(ttir_paths)
            ttnn_paths, ttnn_graph_labels = _select_tp_ttirs(ttnn_paths)
        else:
            ttir_paths = ttir_paths[:max_ttirs]
            ttnn_paths = ttnn_paths[:max_ttirs]
        copied_paths = _copy_mlirs(ttir_paths, output_dir, group, model_name, "ttir", graph_labels)
        copied_ttnn_paths = _copy_mlirs(ttnn_paths, ttnn_output_dir, group, model_name, "ttnn", ttnn_graph_labels)
        status_note = f"{status}"
        if error:
            status_note = f"{status} ({error})"
        if status == "ok":
            status_icon = "✅"
        elif status in {"skipped", "unsupported"}:
            status_icon = "⚠️"
        else:
            status_icon = "❌"
        print(f"{status_icon} Finished {group}::{name} -> {status_note}")
        if status == "failed" and error_detail:
            print("    Last output:")
            for line in error_detail.splitlines():
                print(f"    {line}")
        results.append(
            {
                "group": group,
                "test": name,
                "status": status,
                "error": error,
                "ttir": [str(path) for path in copied_paths],
                "ttnn": [str(path) for path in copied_ttnn_paths],
            }
        )

    return results


def _collect_status_results(
    group: str,
    test_file_path: Path,
    test_names: list[str],
    output_dir: Path,
    ttnn_output_dir: Path,
    include_prefixes: set[str],
) -> list[dict]:
    results = []
    for name in test_names:
        if _is_test_failed(test_file_path, name) or not _matches_prefixes(name, include_prefixes):
            continue
        completed, missing = _get_output_status(name, output_dir, group, ttnn_output_dir)
        results.append(
            {
                "group": group,
                "test": name,
                "status": "complete" if completed else "missing",
                "error": None,
                "missing": missing,
            }
        )
    return results


def _print_summary(results: list[dict]) -> None:
    print("\n📋 One-layer benchmark results:")
    for result in results:
        ttir_summary = result["ttir"] if result["ttir"] else ["none"]
        ttnn_summary = result.get("ttnn") if result.get("ttnn") else ["none"]
        status = result["status"]
        status_icon = "✅" if status == "ok" else "⚠️" if status in {"skipped", "unsupported"} else "❌"
        print(
            f"{status_icon} {result['group']}::{result['test']} status={status} "
            f"ttir={ttir_summary} ttnn={ttnn_summary}"
        )
        if result["error"]:
            print(f"   🧾 error={result['error']}")

    failures = [r for r in results if r["status"] == "failed"]
    if failures:
        print("\n❌ Failures:")
        for result in failures:
            print(f"   ❌ {result['group']}::{result['test']} error={result['error']}")


def _print_status_summary(results: list[dict]) -> None:
    print("\n🧭 One-layer benchmark status:")
    for result in results:
        status = result["status"]
        status_icon = "✅" if status == "complete" else "❌" if status == "failed" else "⚠️"
        missing = result.get("missing", [])
        missing_note = f" missing={missing}" if missing else ""
        print(f"{status_icon} {result['group']}::{result['test']} status={status}{missing_note}")
        if result["error"]:
            print(f"   🧾 error={result['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-layer pytest tests to export TTIRs.")
    parser.add_argument("--include-tp", action="store_true", help="Include TP LLM tests.")
    parser.add_argument(
        "--tp-only",
        action="store_true",
        help="Only run TP LLM tests (implies --include-tp).",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Report missing TTIR outputs without running tests.",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume from where left off (skip tests with output files).",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="Only run tests whose model name starts with this prefix (case-insensitive).",
    )
    args = parser.parse_args()

    tt_xla_dir, repo_root = _resolve_paths()
    _ensure_import_paths(repo_root, tt_xla_dir)
    os.chdir(tt_xla_dir)

    export_path = (tt_xla_dir / "modules").resolve()
    output_dir = (tt_xla_dir / "single_layer_tests").resolve()
    ttnn_output_dir = (tt_xla_dir / "single_layer_tests_ttnn").resolve()
    include_prefixes = _parse_prefix_patterns(args.prefix)

    print(f"\n{'='*60}")
    print("One-Layer Benchmark Runner")
    print(f"{'='*60}")
    print(f"Export path:     {export_path}")
    print(f"Output directory: {output_dir}")
    print(f"TTNN directory:   {ttnn_output_dir}")
    include_tp = args.include_tp or args.tp_only
    test_types = "LLM" + (" + TP" if include_tp else "")
    if args.tp_only:
        test_types = "LLM (TP only)"
    encoder_note = "" if args.tp_only else ", Encoder"
    print(f"Test types:      {test_types}{encoder_note}")
    if include_prefixes:
        print(f"Only running model prefixes: {', '.join(sorted(include_prefixes))}")
    if args.status:
        print("Status mode:     Enabled (no tests will be run)")
    if args.resume:
        print(f"Resume mode:     Enabled (checking {output_dir} for existing files)")
    print(f"{'='*60}\n")

    if args.status:
        results = []
        results.extend(
            _collect_status_results(
                "llm",
                tt_xla_dir / "test_llms.py",
                _test_names_llm(include_tp, args.tp_only),
                output_dir,
                ttnn_output_dir,
                include_prefixes,
            )
        )
        if not args.tp_only:
            results.extend(
                _collect_status_results(
                    "encoder",
                    tt_xla_dir / "test_encoders.py",
                    _test_names_encoder(),
                    output_dir,
                    ttnn_output_dir,
                    include_prefixes,
                )
            )

        _print_status_summary(results)
        return 0

    results = []
    results.extend(
        _run_tests(
            "llm",
            tt_xla_dir / "test_llms.py",
            _test_names_llm(include_tp, args.tp_only),
            export_path,
            output_dir,
            ttnn_output_dir,
            include_prefixes,
            args.resume,
            max_ttirs=2,
        )
    )
    if not args.tp_only:
        results.extend(
            _run_tests(
                "encoder",
                tt_xla_dir / "test_encoders.py",
                _test_names_encoder(),
                export_path,
                output_dir,
                ttnn_output_dir,
                include_prefixes,
                args.resume,
                max_ttirs=1,
            )
        )

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
