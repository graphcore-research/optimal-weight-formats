# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch
from torch import tensor

from .. import quantisation_training as T


def test_quantise_ste() -> None:
    x = tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4, 1.2, 1.4]).requires_grad_()
    ex = tensor([0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    centroids = tensor([0, 0.5, 1.0]).requires_grad_()

    y = T.Quantise_STE.apply(x, centroids, False)
    y.backward(torch.full_like(y, 2.0))

    torch.testing.assert_close(y, ex)
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))
    torch.testing.assert_close(centroids.grad, tensor([8.0, 4.0, 8.0]))

    x.grad = centroids.grad = None
    y = T.Quantise_STE.apply(x, centroids, True)
    y.backward(torch.full_like(y, 2.0))

    torch.testing.assert_close(
        x.grad, tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0])
    )
