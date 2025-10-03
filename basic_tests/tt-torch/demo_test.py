# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import tt_torch
from tt_torch.dynamo.backend import backend


class AddTensors(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = AddTensors()
tt_model = torch.compile(model, backend=backend)

x = torch.ones(5, 5)
y = torch.ones(5, 5)
print(tt_model(x, y))
