# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import re
import importlib.metadata as im

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <package_name> <field_key>", file=sys.stderr)
    sys.exit(1)

pkg = sys.argv[1]
field_key = sys.argv[2]

try:
    md = im.metadata(pkg)
except im.PackageNotFoundError:
    print(f"{pkg} not installed", file=sys.stderr)
    sys.exit(1)

# Try as a metadata field first (Key: Value)
value = md.get(field_key)
if not value:
    # Fallback: search raw metadata text for "field_key=..."
    text = str(md)
    pattern = rf"{re.escape(field_key)}=([0-9a-zA-Z_.-]+)"
    m = re.search(pattern, text)
    if not m:
        print(f"{field_key} not found in metadata for {pkg}", file=sys.stderr)
        sys.exit(1)
    value = m.group(1)

print(value.strip())
