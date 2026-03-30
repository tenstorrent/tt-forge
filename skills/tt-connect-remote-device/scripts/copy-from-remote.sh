#!/bin/bash
# Copy a file from the remote machine to local
# Usage: ./copy-from-remote.sh <remote_path> <local_path>
#
# Examples:
#   ./copy-from-remote.sh /tmp/ttlang_kernel_compute_abc.cpp ./compute.cpp
#   ./copy-from-remote.sh /tmp/ttlang_final.mlir /tmp/final.mlir

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load config
if [ -f "$SCRIPT_DIR/remote.conf" ]; then
    source "$SCRIPT_DIR/remote.conf"
else
    echo "Error: No config file found. Copy remote.conf.example to remote.conf and configure."
    exit 1
fi

source "$SCRIPT_DIR/_lib.sh"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <remote_path> <local_path>"
    echo ""
    echo "Copy a file from the remote machine to your local machine."
    echo ""
    echo "Examples:"
    echo "  $0 /tmp/ttlang_kernel_compute_abc.cpp ./compute.cpp"
    echo "  $0 /tmp/ttlang_final.mlir /tmp/final.mlir"
    exit 1
fi

REMOTE_PATH="$1"
LOCAL_PATH="$2"

echo "Copying: remote:$REMOTE_PATH -> $LOCAL_PATH"
remote_copy_from "$REMOTE_PATH" "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    echo "Done."
else
    echo "Error: Copy failed"
    exit 1
fi
