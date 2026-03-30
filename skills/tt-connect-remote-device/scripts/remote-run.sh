#!/bin/bash
# Run an arbitrary command on the remote machine
# Usage: ./remote-run.sh <command> [args...]
#
# Examples:
#   ./remote-run.sh cat /tmp/ttlang_test_output.log
#   ./remote-run.sh tail -100 /tmp/ttlang_test_output.log
#   ./remote-run.sh grep -i "error" /tmp/ttlang_test_output.log
#   ./remote-run.sh pkill -9 python

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
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Run a command on the remote machine."
    echo ""
    echo "Examples:"
    echo "  $0 cat /tmp/ttlang_test_output.log"
    echo "  $0 tail -100 /tmp/ttlang_test_output.log"
    echo "  $0 grep -i error /tmp/ttlang_test_output.log"
    echo "  $0 pkill -9 python"
    exit 1
fi

remote_run "$@"
