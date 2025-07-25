#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

echo "Running ttrt perf on $1"
ttrt perf $1 --ignore-version
if [ $? -ne 0 ]; then
    echo "Error: TTRT perf command failed."
    exit 1
fi
echo "run device_perf.py creating $2"
python ./benchmark/device_perf.py -cdp ttrt-artifacts/out.ttnn/perf/ops_perf_results.csv $2
csv_file="${2%.*}.csv"
cp ttrt-artifacts/out.ttnn/perf/ops_perf_results.csv "$csv_file"
if [ $? -ne 0 ]; then
    echo "Error: Failed to copy ops_perf_results.csv to $csv_file."
    exit 1
fi
echo "Copied ops_perf_results.csv to $csv_file"
