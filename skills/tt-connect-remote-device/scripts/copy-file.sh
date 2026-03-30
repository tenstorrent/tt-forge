#!/bin/bash
# Copy a single file to the remote machine
# Usage: ./copy-file.sh <local_path> [remote_dest_path]
#
# If remote_dest_path is not specified, copies to /tmp/ with same filename.
#
# Examples:
#   ./copy-file.sh my_kernel.py                    # -> /tmp/my_kernel.py
#   ./copy-file.sh my_kernel.py /tmp/test.py       # -> /tmp/test.py
#   ./copy-file.sh kernels/compute.cpp kernels/    # -> kernels/compute.cpp (mkdir -p)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load config
if [ -f "$SCRIPT_DIR/remote.conf" ]; then
    source "$SCRIPT_DIR/remote.conf"
else
    echo "Error: No config file found. Copy remote.conf.example to remote.conf and configure."
    exit 1
fi

source "$SCRIPT_DIR/_lib.sh"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <local_path> [remote_dest_path]"
    echo ""
    echo "Copy a file from your machine to the remote."
    echo ""
    echo "If remote_dest_path is not specified, copies to /tmp/"
    echo "If remote_dest_path ends with /, treats it as a directory."
    echo ""
    echo "Examples:"
    echo "  $0 my_kernel.py                     # -> /tmp/my_kernel.py"
    echo "  $0 my_kernel.py /tmp/test.py        # -> /tmp/test.py"
    echo "  $0 compute.cpp kernels/             # -> kernels/compute.cpp"
    exit 1
fi

LOCAL_PATH="$1"
REMOTE_DEST="$2"

# Check local file exists
if [ ! -f "$LOCAL_PATH" ]; then
    echo "Error: File not found: $LOCAL_PATH"
    exit 1
fi

FILENAME=$(basename "$LOCAL_PATH")

# Determine destination
if [ -z "$REMOTE_DEST" ]; then
    REMOTE_DEST="/tmp/$FILENAME"
elif [[ "$REMOTE_DEST" == */ ]]; then
    # Dest ends with /, it's a directory
    remote_run mkdir -p "$REMOTE_DEST"
    REMOTE_DEST="$REMOTE_DEST$FILENAME"
fi

echo "Copying: $LOCAL_PATH -> remote:$REMOTE_DEST"
remote_copy_file "$LOCAL_PATH" "$REMOTE_DEST"

if [ $? -eq 0 ]; then
    echo "Done. File available at: $REMOTE_DEST"
else
    echo "Error: Copy failed"
    exit 1
fi
