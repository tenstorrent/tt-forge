# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import jax


def benchmark(config: dict):
    jax.devices("tt")
