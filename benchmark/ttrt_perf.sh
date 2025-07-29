#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run ttrt perf on a flatbuffer_binary file and generate json_file perf report.

# parameters:
# <flatbuffer_binary> <json_file>

echo "Running ttrt perf on $1"
ttrt perf $1 --ignore-version
if [ $? -ne 0 ]; then
    echo "Error: TTRT perf command failed."
    exit 1
fi
echo "run device_perf.py creating $2"
python ./benchmark/device_perf.py -cdp ttrt-artifacts/$1/perf/ops_perf_results_minus_const_eval.csv $2
csv_file="${2%.*}.csv"
cp ttrt-artifacts/$1/perf/ops_perf_results_minus_const_eval.csv "$csv_file"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy ops_perf_results_minus_const_eval.csv to $csv_file."
    exit 1
fi
echo "Copied ops_perf_results_minus_const_eval.csv to $csv_file"
