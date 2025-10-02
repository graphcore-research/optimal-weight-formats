# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch import nn

from .. import qat
from ..core import AttrDict


@pytest.mark.parametrize("pass_mask", [False, True])
def test_compute_kl_loss(pass_mask: bool) -> None:
    torch.manual_seed(100)

    reference_logits = torch.randn((3, 5, 16))
    model_logits = (
        reference_logits.clone() + torch.randn_like(reference_logits).mul(0.1)
    ).requires_grad_()
    model_logits.retain_grad()

    # Important to do this before running qat._compute_kl_loss
    model_logits_ref = model_logits.detach().clone().requires_grad_()
    model_logits_ref.retain_grad()

    attention_mask = torch.rand(reference_logits.shape[:-1]) < 0.75

    if pass_mask:
        mask = torch.rand(reference_logits.shape[:-1]) < 0.5
        full_mask = attention_mask & mask

        loss = qat._compute_kl_loss(
            lambda attention_mask, use_cache: AttrDict(logits=model_logits),
            lambda attention_mask, use_cache: AttrDict(logits=reference_logits),
            AttrDict(attention_mask=attention_mask),
            mask=mask,
        )
    else:
        full_mask = attention_mask
        loss = qat._compute_kl_loss(
            lambda attention_mask, use_cache: AttrDict(logits=model_logits),
            lambda attention_mask, use_cache: AttrDict(logits=reference_logits),
            AttrDict(attention_mask=attention_mask),
        )
    loss.backward()

    loss_ref = (
        torch.nn.functional.kl_div(
            torch.log_softmax(model_logits_ref, -1),
            torch.log_softmax(reference_logits, -1),
            log_target=True,
            reduction="none",
        )
        .mul(full_mask.unsqueeze(-1))
        .sum()
    )
    loss_ref.backward()

    torch.testing.assert_close(loss, loss_ref)
    torch.testing.assert_close(model_logits.grad, model_logits_ref.grad)


def test_deepcopy_with_dummy_params() -> None:
    torch.manual_seed(100)
    batch = torch.randn(5, 10)
    model = nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 20), nn.Linear(20, 20)
    )
    model[-2].weight = model[-1].weight
    model_copy = qat._deepcopy_with_dummy_params(model)
    assert model_copy[-2].weight is model_copy[-1].weight
    for p in model_copy.parameters():
        assert all(s == 0 for s in p.stride())
    assert model_copy(batch).isnan().all()

    qat._replace_params(model_copy, dict(model.named_parameters()))
    assert model_copy[-1].weight is not model[-1].weight
    assert model_copy[-2].weight is model_copy[-1].weight
    torch.testing.assert_close(model_copy(batch), model(batch))
