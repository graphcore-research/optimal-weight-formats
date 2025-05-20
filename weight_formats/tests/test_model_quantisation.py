# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch
from torch import nn

from .. import fit as F
from .. import model_quantisation as M
from .. import quantisation as Q


def test_quantise_2d_fixed() -> None:
    torch.manual_seed(100)
    model = nn.Sequential(nn.Linear(80, 200), nn.Tanh(), nn.Linear(200, 30))
    input = torch.randn(100, 80)
    original_output = model(input)
    log = M.quantise_2d_fixed(model, F.Scaled(4, "fp", Q.BFLOAT16, (None, None), "rms"))
    output = model(input)
    assert 4 < log["bits_per_param"] < 4.5
    assert set(log["params"]) == {k for k, v in model.named_parameters() if v.ndim == 2}
    assert all(set(x) == {"bits", "rmse"} for x in log["params"].values())
    assert 0.11 < Q.rmse_norm(original_output, output).item() < 0.15  # empirical
