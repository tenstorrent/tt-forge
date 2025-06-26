# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import jax

tt_device = jax.devices("tt")[0]

m = jax.numpy.array([1, 2, 3])
m = jax.device_put(m, device=tt_device)
x = jax.numpy.array([4, 5, 6])
x = jax.device_put(x, device=tt_device)
b = jax.numpy.array([7, 8, 9])
b = jax.device_put(b, device=tt_device)


@jax.jit
def compute_y(m, x, b):
    return m * x + b


y = compute_y(m, x, b)
