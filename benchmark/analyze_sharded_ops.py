#!/usr/bin/env python3

# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MLIR Sharding Analysis Tool

This script analyzes TTNN MLIR files to report on sharding effectiveness,
DRAM spills, and memory usage patterns.

USAGE:
------

1. Single File Analysis (verbose mode):
   python3 scripts/analyze_sharded_ops.py <file.ttnn.mlir>

   Example:
   python3 scripts/analyze_sharded_ops.py xla-mlirs/resnet_sharded.ttnn.mlir

   Output:
   - Detailed per-operation analysis showing sharding status
   - Operations marked as [SPILLED TO DRAM] or [SYSTEM_MEMORY]
   - Layout information for each operation
   - Summary statistics with percentages

2. Directory Analysis (batch mode with CSV output):
   python3 scripts/analyze_sharded_ops.py <directory> [--csv output.csv]

   Example:
   python3 scripts/analyze_sharded_ops.py xla-mlirs/ --csv sharding_report.csv

   Analyzes all files matching *.ttnn.mlir or *_ttnn.mlir patterns.

   Output:
   - Progress indicator for each file analyzed
   - CSV report with comprehensive metrics for all files
   - Default CSV filename: sharding_analysis.csv

METRICS REPORTED:
-----------------
- total_ops: Total TTNN operations with tensor outputs
- sharded_ops: Operations using sharded layouts (height/width/block sharded)
- effectively_sharded_ops: Sharded ops that are NOT spilled to DRAM
- sharded_and_spilled_ops: Sharded ops whose output is spilled to DRAM
- dram_spilled_ops: Total operations spilled to DRAM (interleaved)
- system_memory_ops: Operations with system_memory buffer type
- Percentages: All counts as percentage of total_ops

KEY INSIGHT:
------------
The "effectively_sharded_percentage" is the most important metric, as it
represents operations that benefit from sharding without losing performance
to DRAM spills.

EXCLUDED OPERATIONS:
--------------------
The following operations are excluded from analysis as they are infrastructure
operations rather than computational operations:
- ttnn.get_device
"""

import argparse
import os
import sys
import re
import csv
from pathlib import Path

EXCLUDED_OPS = {
    "ttnn.get_device",
}


def process_mlir_file(file_path, verbose=True):
    layout_def_pattern = re.compile(r"^(#ttnn_layout\d+)\s*=\s*(#ttnn\.ttnn_layout<.*>)")
    ttnn_op_pattern = re.compile(r'%\w+\s*=\s*"(ttnn\.[\w]+)"')
    result_pattern = re.compile(r"(%\w+)\s*=")
    result_tensor_pattern = re.compile(r"->\s*(tensor<.*>)")
    sharding_attr_pattern = re.compile(r"<(height_sharded|width_sharded|block_sharded)>")
    dram_interleaved_pattern = re.compile(r"memory_config\s*=\s*#ttnn\.memory_config<<?#?dram>?,\s*<interleaved>>")
    system_memory_pattern = re.compile(r"memref<[^>]*,\s*#(system_memory|ttnn\.buffer_type<system_memory>)>")
    operand_pattern = re.compile(r"\((%\w+)")

    forward_func_start_pattern = re.compile(r"^\s*func\.func\s+@(?:main|forward)\b")
    generic_func_end_pattern = re.compile(r"^\s*}\s*$")

    sharded_layouts = {}
    layout_definitions = {}
    system_memory_layouts = {}

    result_to_op_map = {}
    dram_spilled_results = set()

    sharded_op_count = 0
    total_op_count = 0
    dram_spilled_op_count = 0
    system_memory_op_count = 0
    detailed_op_info = []
    ttnn_op_names = set()

    in_forward_function = False

    with open(file_path, "r") as f:
        for line_number, line_content_raw in enumerate(f, 1):
            line_content_stripped = line_content_raw.strip()

            match_layout = layout_def_pattern.match(line_content_stripped)
            if match_layout:
                layout_name = match_layout.group(1)
                layout_attr_str = match_layout.group(2)
                layout_definitions[layout_name] = line_content_stripped
                if sharding_attr_pattern.search(layout_attr_str):
                    sharded_layouts[layout_name] = layout_attr_str
                if system_memory_pattern.search(layout_attr_str):
                    system_memory_layouts[layout_name] = layout_attr_str
                continue

            if not in_forward_function and forward_func_start_pattern.search(line_content_stripped):
                in_forward_function = True
                continue

            if in_forward_function and generic_func_end_pattern.match(line_content_stripped):
                in_forward_function = False
                continue

            if not in_forward_function:
                continue

            match_ttnn_op = ttnn_op_pattern.search(line_content_stripped)
            if match_ttnn_op:
                op_name_qualified = match_ttnn_op.group(1)

                result_match = result_pattern.search(line_content_stripped)
                if result_match:
                    result_name = result_match.group(1)
                    result_to_op_map[result_name] = {
                        "op_name": op_name_qualified,
                        "line_num": line_number,
                        "line": line_content_stripped,
                    }

                if op_name_qualified == "ttnn.to_memory_config":
                    if dram_interleaved_pattern.search(line_content_stripped):
                        operand_match = operand_pattern.search(line_content_stripped)
                        if operand_match:
                            source_result = operand_match.group(1)

                            current_result = source_result
                            while current_result in result_to_op_map:
                                source_op_info = result_to_op_map[current_result]
                                source_op_name = source_op_info["op_name"]

                                if source_op_name in EXCLUDED_OPS or source_op_name == "ttnn.to_memory_config":
                                    operand_of_excluded = operand_pattern.search(source_op_info["line"])
                                    if operand_of_excluded:
                                        current_result = operand_of_excluded.group(1)
                                        continue
                                    else:
                                        break
                                else:
                                    dram_spilled_results.add(current_result)
                                    break
                    continue

                if op_name_qualified in EXCLUDED_OPS:
                    continue

                ttnn_op_names.add(op_name_qualified)

                has_tensor_output = result_tensor_pattern.search(line_content_stripped)
                if has_tensor_output:
                    total_op_count += 1
                    current_op_info = {
                        "line_num": line_number,
                        "line": line_content_stripped,
                        "op_name": op_name_qualified,
                        "result_name": result_match.group(1) if result_match else None,
                        "status": "NOT SHARDED",
                        "spilled_to_dram": False,
                        "has_system_memory": False,
                        "layouts": [],
                    }
                    is_sharded_current_op = False

                    operand_section = line_content_stripped.split("->")[0] if "->" in line_content_stripped else ""
                    if operand_section:
                        input_layout_aliases = re.findall(r"(#ttnn_layout\d+)", operand_section)
                        for alias in input_layout_aliases:
                            if alias in system_memory_layouts:
                                current_op_info["has_system_memory"] = True
                                break

                    result_tensor_match = result_tensor_pattern.search(line_content_stripped)
                    if result_tensor_match:
                        tensor_def_str = result_tensor_match.group(1)

                        if sharding_attr_pattern.search(tensor_def_str):
                            is_sharded_current_op = True
                            current_op_info["status"] = "SHARDED (inline)"
                            inline_layout_detail_match = re.search(r"(#ttnn\.ttnn_layout<.*?>)", tensor_def_str)
                            if inline_layout_detail_match:
                                current_op_info["layouts"].append(
                                    f"{inline_layout_detail_match.group(1)} (INLINE SHARDED)"
                                )
                            else:
                                current_op_info["layouts"].append(
                                    "(Inline sharding attribute detected in result tensor)"
                                )
                        else:
                            used_layout_aliases = re.findall(r"(#ttnn_layout\d+)", tensor_def_str)
                            if used_layout_aliases:
                                for alias in used_layout_aliases:
                                    layout_full_def_str = layout_definitions.get(
                                        alias, "DEFINITION NOT FOUND IN PARSED LAYOUTS"
                                    )
                                    if alias in system_memory_layouts:
                                        current_op_info["has_system_memory"] = True
                                    if alias in sharded_layouts:
                                        is_sharded_current_op = True
                                        current_op_info["status"] = f"SHARDED (via alias {alias})"
                                        current_op_info["layouts"].append(f"{alias} (SHARDED): {layout_full_def_str}")
                                        break
                                    else:
                                        current_op_info["layouts"].append(
                                            f"{alias} (Interleaved/Other): {layout_full_def_str}"
                                        )
                            else:
                                generic_layout_match = re.search(r"(#ttnn\.layout<.*?>)", tensor_def_str)
                                if generic_layout_match:
                                    current_op_info["layouts"].append(
                                        f"{generic_layout_match.group(1)} (Interleaved/Other)"
                                    )

                    if is_sharded_current_op:
                        sharded_op_count += 1

                    detailed_op_info.append(current_op_info)

    sharded_and_spilled_count = 0
    for op_info in detailed_op_info:
        if op_info["result_name"] and op_info["result_name"] in dram_spilled_results:
            op_info["spilled_to_dram"] = True
            dram_spilled_op_count += 1
            if op_info["status"] != "NOT SHARDED":
                sharded_and_spilled_count += 1
        if op_info["has_system_memory"]:
            system_memory_op_count += 1

    effectively_sharded_count = sharded_op_count - sharded_and_spilled_count

    if verbose:
        for op_info in detailed_op_info:
            status_suffix = ""
            if op_info["spilled_to_dram"]:
                status_suffix += " [SPILLED TO DRAM]"
            if op_info["has_system_memory"]:
                status_suffix += " [SYSTEM_MEMORY]"
            print(f"Line {op_info['line_num']}: {op_info['op_name']}{status_suffix}")
            print(f"  {op_info['line']}")
            print(f"  Status: {op_info['status']}")
            if op_info["spilled_to_dram"]:
                print(f"  WARNING: Output spilled to DRAM (interleaved)")
            if op_info["has_system_memory"]:
                print(f"  INFO: Output uses system_memory buffer")
            if op_info["layouts"]:
                print("  Layouts referenced in result:")
                for layout_str in op_info["layouts"]:
                    print(f"    - {layout_str}")
            print()

    return (
        sharded_op_count,
        effectively_sharded_count,
        total_op_count,
        dram_spilled_op_count,
        sharded_and_spilled_count,
        system_memory_op_count,
        ttnn_op_names,
    )


def analyze_single_file(file_path, verbose=True):
    (
        sharded_ops,
        effectively_sharded_ops,
        total_ops,
        dram_spilled_ops,
        sharded_and_spilled_ops,
        system_memory_ops,
        ttnn_op_names,
    ) = process_mlir_file(file_path, verbose=verbose)

    # Always print summary (needed for CI metric extraction)
    print("--- Summary ---")
    excluded_ops_str = ", ".join(sorted(list(EXCLUDED_OPS)))
    print(f"Excluded operations: [{excluded_ops_str}]")
    print()

    if verbose:
        op_list_str = ", ".join(sorted(list(ttnn_op_names)))
        print(f"TTNN operations with tensor outputs found (within @forward/@main function):")
        print(f"  [{op_list_str}]")
        print()

    print(f"Total number of TTNN operations with tensor outputs: {total_ops}")
    print(f"Number of sharded TTNN operations: {sharded_ops}")
    print(f"Number of sharded operations spilled to DRAM: {sharded_and_spilled_ops}")
    print(f"Number of effectively sharded operations (not spilled): {effectively_sharded_ops}")
    print(f"Total number of operations spilled to DRAM: {dram_spilled_ops}")
    print(f"Number of operations with system_memory output: {system_memory_ops}")
    print()
    if total_ops > 0:
        sharded_percentage = (sharded_ops / total_ops) * 100
        effectively_sharded_percentage = (effectively_sharded_ops / total_ops) * 100
        spilled_percentage = (dram_spilled_ops / total_ops) * 100
        system_memory_percentage = (system_memory_ops / total_ops) * 100
        print(f"Percentage of TTNN operations that are sharded: {sharded_percentage:.2f}%")
        print(f"Percentage of TTNN operations effectively sharded: {effectively_sharded_percentage:.2f}%")
        print(f"Percentage of TTNN operations spilled to DRAM: {spilled_percentage:.2f}%")
        print(f"Percentage of TTNN operations with system_memory: {system_memory_percentage:.2f}%")
    else:
        print("No TTNN operations with tensor outputs found.")

    return {
        "total_ops": total_ops,
        "sharded_ops": sharded_ops,
        "effectively_sharded_ops": effectively_sharded_ops,
        "sharded_and_spilled_ops": sharded_and_spilled_ops,
        "dram_spilled_ops": dram_spilled_ops,
        "system_memory_ops": system_memory_ops,
        "sharded_percentage": (sharded_ops / total_ops * 100) if total_ops > 0 else 0.0,
        "effectively_sharded_percentage": (effectively_sharded_ops / total_ops * 100) if total_ops > 0 else 0.0,
        "spilled_percentage": (dram_spilled_ops / total_ops * 100) if total_ops > 0 else 0.0,
        "system_memory_percentage": (system_memory_ops / total_ops * 100) if total_ops > 0 else 0.0,
    }


def analyze_directory(directory_path, output_csv):
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        print(f"Error: {directory_path} is not a directory")
        return

    mlir_files_dot = list(dir_path.glob("*.ttnn.mlir"))
    mlir_files_underscore = list(dir_path.glob("*_ttnn.mlir"))
    mlir_files = sorted(set(mlir_files_dot + mlir_files_underscore))

    if not mlir_files:
        print(f"No files matching *.ttnn.mlir or *_ttnn.mlir found in {directory_path}")
        return

    print(f"Found {len(mlir_files)} TTNN MLIR files to analyze")
    print()

    results = []
    for mlir_file in mlir_files:
        print(f"Analyzing {mlir_file.name}...")
        try:
            result = analyze_single_file(str(mlir_file), verbose=False)
            result["file_name"] = mlir_file.name
            results.append(result)
            print(
                f"  ✓ Completed: {result['total_ops']} ops, {result['effectively_sharded_percentage']:.2f}% effectively sharded"
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not results:
        print("No results to write")
        return

    csv_fields = [
        "file_name",
        "total_ops",
        "sharded_ops",
        "effectively_sharded_ops",
        "sharded_and_spilled_ops",
        "dram_spilled_ops",
        "system_memory_ops",
        "sharded_percentage",
        "effectively_sharded_percentage",
        "spilled_percentage",
        "system_memory_percentage",
    ]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print()
    print(f"CSV report written to: {output_csv}")
    print(f"Analyzed {len(results)} files successfully")


def main():
    parser = argparse.ArgumentParser(description="Analyze MLIR file(s) for sharded TTNN operations and DRAM spills.")
    parser.add_argument("path", help="Path to a .ttnn.mlir file or directory containing TTNN MLIR files")
    parser.add_argument(
        "--csv", help="Output CSV file path (only used with directory analysis)", default="sharding_analysis.csv"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output (default: False)")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        analyze_single_file(str(path), verbose=args.verbose)
    elif path.is_dir():
        analyze_directory(str(path), args.csv)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
