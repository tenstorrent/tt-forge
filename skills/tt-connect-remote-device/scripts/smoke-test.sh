#!/bin/bash
# Smoke test: verify remote setup is working
# This tests that REMOTE_SHELL can copy files and run Python with ttnn.
#
# Usage: ./smoke-test.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load config
if [ -f "$SCRIPT_DIR/remote.conf" ]; then
    source "$SCRIPT_DIR/remote.conf"
else
    echo "FAIL: No remote.conf found. Copy remote.conf.example to remote.conf and configure."
    exit 1
fi

source "$SCRIPT_DIR/_lib.sh"

echo "========================================"
echo "TT-Lang Remote Smoke Test"
echo "REMOTE_SHELL=$REMOTE_SHELL"
echo "========================================"

# Test 1: remote shell works
echo ""
echo "[1/3] Testing remote shell..."
OUTPUT=$(remote_run echo "remote-shell-ok" 2>&1)
if [[ "$OUTPUT" == *"remote-shell-ok"* ]]; then
    echo "  PASS: remote shell works"
else
    echo "  FAIL: remote shell not responding"
    echo "  Output: $OUTPUT"
    exit 1
fi

# Test 2: copy file to remote
echo ""
echo "[2/3] Testing file copy..."
TEMP_FILE=$(mktemp /tmp/ttl_smoke_XXXX.py)
cat > "$TEMP_FILE" << 'EOF'
import ttnn
print("ttl-smoke-ok")
EOF
remote_copy_file "$TEMP_FILE" "/tmp/ttl_smoke_test.py"
VERIFY=$(remote_run cat /tmp/ttl_smoke_test.py 2>&1)
if [[ "$VERIFY" == *"ttl-smoke-ok"* ]]; then
    echo "  PASS: file copied and verified"
else
    echo "  FAIL: file copy failed or content mismatch"
    echo "  Output: $VERIFY"
    rm "$TEMP_FILE"
    exit 1
fi
rm "$TEMP_FILE"

# Test 3: run Python with ttnn
echo ""
echo "[3/3] Testing Python + ttnn import..."
OUTPUT=$(remote_run python3 /tmp/ttl_smoke_test.py 2>&1)
if [[ "$OUTPUT" == *"ttl-smoke-ok"* ]]; then
    echo "  PASS: python3 + ttnn works"
else
    echo "  FAIL: python3 or ttnn import failed"
    echo "  Output: $OUTPUT"
    exit 1
fi

echo ""
echo "========================================"
echo "ALL TESTS PASSED"
echo "========================================"
