# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import jax

m = jax.numpy.array([1, 2, 3])
x = jax.numpy.array([4, 5, 6])
b = jax.numpy.array([7, 8, 9])

@jax.jit
def compute_y(m, x, b):
    return m * x + b

y = compute_y(m, x, b)

with jax.default_device(jax.devices("cpu")[0]):
    y_ref = m * x + b
    assert all(jax.numpy.abs(y-y_ref)) < 1e-6, "Device and CPU results do not match!"