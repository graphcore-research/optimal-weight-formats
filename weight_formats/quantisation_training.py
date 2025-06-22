# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Core utilities for quantisation-aware training."""

import torch
from torch import Tensor


class Quantise_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, centroids: Tensor, clip_gradient: bool) -> Tensor:
        ctx.save_for_backward(x, centroids)
        ctx.clip_gradient = clip_gradient
        boundaries = (centroids[1:] + centroids[:-1]) / 2
        return centroids[torch.bucketize(x, boundaries)]

    @staticmethod
    def backward(ctx, gy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, centroids = ctx.saved_tensors

        if not x.requires_grad:
            gx = None
        elif ctx.clip_gradient:
            gx = gy.mul(
                x.ge(centroids[0] + (centroids[0] - centroids[1]) / 2)
                & x.le(centroids[-1] + (centroids[-1] - centroids[-2]) / 2)
            )
        else:
            gx = gy

        if not centroids.requires_grad:
            gcentroids = None
        else:
            boundaries = (centroids[1:] + centroids[:-1]) / 2
            gcentroids = torch.zeros_like(centroids).index_add_(
                0, torch.bucketize(x, boundaries).flatten(), gy.flatten()
            )

        return (gx, gcentroids, None)
