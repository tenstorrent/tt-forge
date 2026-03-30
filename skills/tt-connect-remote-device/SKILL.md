---
name: tt-connect-remote-device
description: Set up and verify remote connection to Tenstorrent hardware. Provides tools for running kernels, copying files, and reading logs on remote devices.
user-invocable: true
---

## Prerequisites

All tools are in the `scripts/` directory relative to this skill. Invoke them with the full path.

Before doing anything else, run the smoke test to verify your remote setup:
```bash
<path-to>/tt-connect-remote-device/scripts/smoke-test.sh
```
If the smoke test fails, STOP. Do NOT continue. Ask the user to fix their remote setup first.

If the smoke test fails due to remote.conf not existing, STOP, offer to help them create it from `scripts/remote.conf.example`. Then work on getting the smoke test passing before exploring or continuing.

## Tools Available

NOTE: flags on run-test.sh must come before the file argument. You can use --help if unsure on how to use.

NOTE: run-test.sh will copy the file. You do not need to copy the test file each time.

```bash
scripts/run-test.sh /path/to/kernel.py                # Run in functional simulator (default)
scripts/run-test.sh --hw /path/to/kernel.py            # Run on real hardware (final validation)
scripts/run-test.sh --hw --emit-runner /path/to.py     # Run + emit C++ kernels and Python runner
scripts/copy-file.sh /path/to/file.py                  # Copy a file TO the remote
scripts/copy-from-remote.sh /remote/path ./local_dir/  # Copy a file FROM the remote
scripts/remote-run.sh <command>                         # Run an arbitrary command on the remote
```

By default, run-test.sh uses the functional simulator (`ttlang-sim`). Use `--hw` for real hardware. **Iterate with the simulator first.** Only move to `--hw` for final validation or if the simulator has a bug that blocks your work. Note: the simulator cannot run TT-Metal C++ programs, so export workflows always require `--hw`.

**Reading remote logs (output is saved, not streamed):**
```bash
scripts/remote-run.sh cat /tmp/ttlang_test_output.log        # Full log
scripts/remote-run.sh tail -100 /tmp/ttlang_test_output.log  # Last 100 lines
scripts/remote-run.sh cat /tmp/ttlang_test_output.log | grep -i "error"  # Search log
scripts/remote-run.sh cat /tmp/ttlang_initial.mlir           # Initial MLIR
scripts/remote-run.sh cat /tmp/ttlang_final.mlir             # Final MLIR
```

**NOTE:** Grep with quoted patterns containing spaces does not work via `remote-run.sh` due to quoting through the SSH+docker chain. Always pipe through grep locally: `remote-run.sh cat /path/to/file | grep "pattern"`
