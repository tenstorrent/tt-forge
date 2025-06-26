# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tt_torch


class AddTensors(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = AddTensors()
tt_model = torch.compile(model, backend="tt")

x = torch.ones(5, 5)
y = torch.ones(5, 5)
print(tt_model(x, y))
