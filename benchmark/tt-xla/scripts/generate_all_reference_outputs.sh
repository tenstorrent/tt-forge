#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Batch script to generate .refpt files for all LLM models
# This is a convenience script and won't be committed to the repo

set -e

# Parse command line arguments
# WARNING: TOTAL_LENGTH must match DEFAULT_INPUT_SEQUENCE_LENGTH in test_llms.py
# Context length mismatch will cause accuracy degradation even with teacher forcing.
TOTAL_LENGTH=128  # Default value (must match test_llms.py DEFAULT_INPUT_SEQUENCE_LENGTH)
while [[ $# -gt 0 ]]; do
    case $1 in
        --total-length)
            TOTAL_LENGTH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Generate reference outputs for all LLM models"
            echo
            echo "Options:"
            echo "  --total-length N    Set the total sequence length (default: 128)"
            echo "                      WARNING: Must match DEFAULT_INPUT_SEQUENCE_LENGTH in test_llms.py"
            echo "  --help, -h          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Get script directory and derive paths relative to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Output directory is ../reference_outputs relative to script location
OUTPUT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/reference_outputs"
# Generation script is in the same directory as this script
GENERATE_SCRIPT="${SCRIPT_DIR}/generate_reference_outputs.py"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Running from: $PWD"
echo "Script directory: $SCRIPT_DIR"
echo "Using generation script: $GENERATE_SCRIPT"
echo "Output directory: $OUTPUT_DIR"
echo


# Define models to generate references for
# Format: "HF_MODEL_NAME|OUTPUT_FILENAME"
declare -a MODELS=(
    # Llama models
    "meta-llama/Llama-3.2-1B-Instruct|Llama-3.2-1B-Instruct.refpt"
    "meta-llama/Llama-3.2-3B-Instruct|Llama-3.2-3B-Instruct.refpt"
    "meta-llama/Llama-3.1-8B-Instruct|Llama-3.1-8B-Instruct.refpt"

    # Gemma models
    "google/gemma-1.1-2b-it|gemma-1.1-2b-it.refpt"
    "google/gemma-2-2b-it|gemma-2-2b-it.refpt"

    # Phi models
    "microsoft/phi-1|phi-1.refpt"
    "microsoft/phi-1_5|phi-1_5.refpt"
    "microsoft/phi-2|phi-2.refpt"

    # Falcon models (use -Base variants, not -Instruct)
    "tiiuae/Falcon3-1B-Base|Falcon3-1B-Base.refpt"
    "tiiuae/Falcon3-3B-Base|Falcon3-3B-Base.refpt"
    "tiiuae/Falcon3-7B-Base|Falcon3-7B-Base.refpt"

    # Qwen 2.5 models
    "Qwen/Qwen2.5-0.5B-Instruct|Qwen2.5-0.5B-Instruct.refpt"
    "Qwen/Qwen2.5-1.5B-Instruct|Qwen2.5-1.5B-Instruct.refpt"
    "Qwen/Qwen2.5-3B-Instruct|Qwen2.5-3B-Instruct.refpt"
    "Qwen/Qwen2.5-7B-Instruct|Qwen2.5-7B-Instruct.refpt"

    # Qwen 3 models
    "Qwen/Qwen3-0.6B|Qwen3-0.6B.refpt"
    "Qwen/Qwen3-1.7B|Qwen3-1.7B.refpt"
    "Qwen/Qwen3-4B|Qwen3-4B.refpt"
    "Qwen/Qwen3-8B|Qwen3-8B.refpt"

    # Mistral variants
    "mistralai/Mistral-7B-Instruct-v0.3|Mistral-7B-Instruct-v0.3.refpt"
    "mistralai/Ministral-8B-Instruct-2410|Ministral-8B-Instruct-2410.refpt"
)

echo "========================================"
echo "Batch Reference Generation"
echo "========================================"
echo "Total models: ${#MODELS[@]}"
echo "Total length: ${TOTAL_LENGTH} tokens"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================"
echo

# Track progress
CURRENT=0
TOTAL=${#MODELS[@]}

# Loop through each model
for MODEL_ENTRY in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))

    # Split the entry into model name and output filename
    IFS='|' read -r MODEL_NAME OUTPUT_FILENAME <<< "$MODEL_ENTRY"
    OUTPUT_FILE="${OUTPUT_DIR}/${OUTPUT_FILENAME}"

    echo "[$CURRENT/$TOTAL] Processing: ${MODEL_NAME}"
    echo "  Output: ${OUTPUT_FILENAME}"

    # Check if file already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "  ⚠️  File already exists, skipping..."
        echo
        continue
    fi

    # Run the generation script
    python3 "$GENERATE_SCRIPT" \
        --total_length "$TOTAL_LENGTH" \
        --output_file "$OUTPUT_FILE" \
        --model "$MODEL_NAME"

    # Cleanup: Remove HuggingFace cache for this model to free up space
    echo "  🧹 Cleaning up cache..."
    rm -rf ~/.cache/huggingface/hub/models--*

    echo "  ✅ Completed"
    echo
done

echo "========================================"
echo "All reference outputs have been generated!"
echo "========================================"

# List all generated files
echo
echo "Generated files:"
ls -lh "${OUTPUT_DIR}"/*.refpt
