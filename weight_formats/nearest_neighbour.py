# Copyright (c) 2026 Graphcore Ltd. All rights reserved.

"""Nearest neighbour search."""

from typing import Literal

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _kernel__nearest_neighbour_triton(
    tensor_ptr,
    centroids_ptr,
    out_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
    DIM: tl.constexpr,
    N_CENTROIDS: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    offs_n = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_n = offs_n < n

    best_dist = tl.full([BLOCK_SIZE], float("inf"), dtype=tl.float32)
    best_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for ic in range(0, N_CENTROIDS):
        dist = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for id in range(0, DIM):
            x = tl.load(tensor_ptr + DIM * offs_n + id, mask=mask_n, other=0.0)
            c = tl.load(centroids_ptr + DIM * ic + id)
            delta = x.to(tl.float32) - c.to(tl.float32)
            dist += delta * delta

        take = dist < best_dist
        best_dist = tl.where(take, dist, best_dist)
        best_idx = tl.where(take, ic, best_idx)

    tl.store(out_ptr + offs_n, best_idx.to(tl.int64), mask=mask_n)


_DEFAULT_BLOCK_SIZE = 1024


def nearest_neighbour_triton(
    tensor: Tensor,
    centroids: Tensor,
    out: Tensor | None = None,
    block_size: int = _DEFAULT_BLOCK_SIZE,
) -> Tensor:
    n, d = tensor.shape
    nc, dc = centroids.shape
    assert dc == d
    assert tensor.is_cuda and centroids.is_cuda

    if out is None:
        out = torch.empty((n,), device=tensor.device, dtype=torch.int64)
    else:
        assert out.shape == (n,)
        assert out.device == tensor.device
        assert out.dtype == torch.int64

    _kernel__nearest_neighbour_triton[(triton.cdiv(n, block_size),)](
        tensor.contiguous(),
        centroids.contiguous(),
        out,
        n,
        BLOCK_SIZE=block_size,
        DIM=d,
        N_CENTROIDS=nc,
    )
    return out


_DEFAULT_MAX_BYTES = 8 * 2**30


def nearest_neighbour_torch(
    tensor: Tensor,
    centroids: Tensor,
    max_bytes: float | None = _DEFAULT_MAX_BYTES,
    out: Tensor | None = None,
) -> Tensor:
    if out is None:
        out = torch.empty(tensor.shape[0], device=tensor.device, dtype=torch.int64)
    chunk_size = (
        tensor.shape[0]
        if max_bytes is None
        else int(max_bytes / (centroids.itemsize * centroids.shape[0]))
    )
    for i in range(0, tensor.shape[0], chunk_size):
        chunk = slice(i, min(tensor.shape[0], i + chunk_size))
        torch.argmin(
            torch.cdist(tensor[chunk].to(centroids.dtype), centroids),
            -1,
            out=out[chunk],
        )
    return out


def nearest_neighbour(
    tensor: Tensor,
    centroids: Tensor,
    out: Tensor | None = None,
    method: Literal["auto", "triton", "torch"] = "auto",
) -> Tensor:
    if method == "auto":
        method = (
            "triton"
            if tensor.is_cuda and centroids.is_cuda and centroids.size(-1) <= 256
            else "torch"
        )
    if method == "triton":
        return nearest_neighbour_triton(tensor, centroids, out=out)
    elif method == "torch":
        return nearest_neighbour_torch(tensor, centroids, out=out)
    else:
        raise ValueError(f"Unknown nearest_neighbour method: {method}")
