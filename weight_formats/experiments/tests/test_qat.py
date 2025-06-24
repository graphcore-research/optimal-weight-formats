# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

from .. import qat
from ..core import AttrDict
import torch


def test_compute_kl_loss() -> None:
    torch.manual_seed(100)

    reference_logits = torch.randn((3, 5, 16))
    model_logits = (
        reference_logits.clone() + torch.randn_like(reference_logits).mul(0.1)
    ).requires_grad_()
    model_logits.retain_grad()

    # Important to do this before running qat._compute_kl_loss
    model_logits_ref = model_logits.detach().clone().requires_grad_()
    model_logits_ref.retain_grad()

    mask = torch.rand(reference_logits.shape[:-1]) < 0.75

    loss = qat._compute_kl_loss(
        lambda attention_mask: AttrDict(logits=model_logits),
        lambda attention_mask: AttrDict(logits=reference_logits),
        AttrDict(attention_mask=mask),
    )
    loss.backward()

    loss_ref = (
        torch.nn.functional.kl_div(
            torch.log_softmax(model_logits_ref, -1),
            torch.log_softmax(reference_logits, -1),
            log_target=True,
            reduction="none",
        )
        .mul(mask.unsqueeze(-1))
        .sum()
    )
    loss_ref.backward()

    torch.testing.assert_close(loss, loss_ref)
    torch.testing.assert_close(model_logits.grad, model_logits_ref.grad)
