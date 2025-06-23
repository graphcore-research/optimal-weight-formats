# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""Core utilities for quantisation-aware training."""

import math
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn

from . import quantisation as Q


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


ScalingMode: TypeAlias = Literal["parameter", "dynamic"]


def _squeeze_odd_dims(t: Tensor) -> Tensor:
    assert all(d == 1 for d in t.shape[1::2])
    return t.reshape(t.shape[::2])


def _unsqueeze_odd_dims(t: Tensor) -> Tensor:
    return t.reshape(tuple(s for d in t.shape for s in (d, 1)))


class Weight(nn.Module):
    """A trainable quantised weight.

    Parameters:
      master -- quantised dense elements
      centroids -- quantisation centroids, for quantising `master/scale`
      scale -- (if scaling_mode="parameter") quantisation scale
      sparse_idx -- (if fmt is SparseFormat) sparse weight indices into
                    the flattened tensor
      sparse_weight -- (if fmt is Q.SparseFormat) sparse weight values
    """

    def __init__(
        self,
        weight: Tensor,
        fmt: Q.TensorFormat,
        scaling_mode: ScalingMode,
        clip_gradient: bool,
    ):
        super().__init__()
        self.fmt = fmt
        self.scaling_mode = scaling_mode
        self.clip_gradient = clip_gradient

        if isinstance(fmt, Q.SparseFormat):
            master, sparse_idx, sparse_weight = Q.SparseFormat.split(
                weight, fmt.sparse_ratio
            )
            self.master = nn.Parameter(master)
            self.sparse_idx = nn.Buffer(sparse_idx)
            self.sparse_weight = nn.Parameter(sparse_weight)
            scaling_fmt = fmt.format
        else:
            self.master = nn.Parameter(weight)
            scaling_fmt = fmt

        assert type(scaling_fmt) == Q.LinearScalingFormat, (
            "`The `fmt` for a `T.Weight` should be a `Q.LinearScalingFormat`"
            f", optionally wrapped in a `Q.SparseFormat`, actual: {type(scaling_fmt)}"
        )
        self.centroids = nn.Parameter(
            torch.tensor(
                scaling_fmt.element_format.centroids,
                device=self.master.device,
                dtype=self.master.dtype,
            )
        )
        self._blocked_shape = Q.blocked_shape(self.master, scaling_fmt.block_shape)
        if scaling_mode == "parameter":
            scale = Q.blocked_scale(
                self.master.reshape(self._blocked_shape),
                scaling=scaling_fmt.scaling,
                element_range=scaling_fmt.element_format.range,
            )
            self.scale = nn.Parameter(_squeeze_odd_dims(scale))
        elif scaling_mode == "dynamic":
            self._scaling = scaling_fmt.scaling
            self._element_range = scaling_fmt.element_format.range
            self._scale_format = scaling_fmt.scale_format
        else:
            assert False, f"Unknown scaling_mode={scaling_mode!r}"
        if hasattr(scaling_fmt.element_format, "compressor"):
            self.compressor = scaling_fmt.element_format.compressor

    @property
    def bits(self) -> float:
        """Bit count; note: only counts `centroids` if trainable."""
        # For `master` (after scaling and quantisation)
        if hasattr(self, "compressor"):
            # Note: this MUST mirror forward()
            weight = self.master.reshape(self._blocked_shape)
            scale = self._get_scale()
            weight = Q.safe_div(weight, scale)
            bits = Q.CompressedLUTFormat.count_bits_compressed(
                weight,
                Q.LUTFormat.create(self.centroids, ""),
                self.compressor,
                model_logp=None,
            )
        else:
            bits = self.master.nelement() * math.log2(self.centroids.nelement())

        # For `scale`
        if self.scaling_mode == "parameter":
            bits += self.scale.nelement() * self.scale.itemsize * 8
        else:
            bits += self._scale_format.count_bits(self._blocked_shape[::2])

        # For `sparse_idx`, `sparse_weight`
        if hasattr(self, "sparse_idx"):
            bits += self.sparse_idx.nelement() * 32  # assume representable in 32 bits
            bits += self.sparse_weight.nelement() * self.sparse_weight.itemsize * 8

        # For `centroids`
        if self.centroids.requires_grad:
            bits += self.centroids.nelement() * self.centroids.itemsize * 8

        return bits

    def _get_scale(self) -> Tensor:
        if self.scaling_mode == "parameter":
            return _unsqueeze_odd_dims(self.scale)
        elif self.scaling_mode == "dynamic":
            scale = Q.blocked_scale(
                self.master.reshape(self._blocked_shape),
                self._scaling,
                self._element_range,
            )
            return self._scale_format.quantise(scale)

    def forward(self) -> Tensor:
        weight = self.master.reshape(self._blocked_shape)
        scale = self._get_scale()
        weight = Q.safe_div(weight, scale)
        weight = Quantise_STE.apply(weight, self.centroids, self.clip_gradient)
        weight = weight.mul(scale)
        weight = weight.reshape(self.master.shape)
        if hasattr(self, "sparse_idx"):
            weight = weight.flatten()
            weight = weight.scatter(0, self.sparse_idx, self.sparse_weight)
            weight = weight.reshape(self.master.shape)
        return weight


class Linear(nn.Module):
    def __init__(self, weight: Weight):
        super().__init__()
        self.weight = weight

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.linear(input, self.weight())


class Embedding(nn.Module):
    def __init__(self, weight: Weight):
        super().__init__()
        self.weight = weight

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.embedding(input, self.weight())
