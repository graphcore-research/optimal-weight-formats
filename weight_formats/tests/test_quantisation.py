# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import dataclasses
import json
from math import log2
from typing import Any

import pytest
import torch
from torch import tensor

from .. import quantisation as Q


def _json_roundtrip(s: Any) -> Any:
    return type(s)(**json.loads(json.dumps(dataclasses.asdict(s))))


# Utilities


def test_shuffle() -> None:
    torch.manual_seed(100)
    x = torch.arange(100)
    y = Q.shuffle(x)
    assert set(x.tolist()) == set(y.tolist())
    assert not x.equal(y), "unlikely"


def test_qrmse_norm_snr() -> None:
    torch.manual_seed(100)
    x = torch.randn(2**10)
    assert Q.qrmse_norm(Q.parse("E2M2"), x) < Q.qrmse_norm(Q.parse("E2M1"), x)
    assert Q.snr(x, Q.ScaledFormat.create(Q.parse("E0M3"), 0.3).quantise(x)) > 60


# Formats


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp_format(dtype: torch.dtype) -> None:
    e2m1 = Q.parse("E2M1")
    assert e2m1.bits == 4
    assert e2m1.range == (-3, 3)
    assert e2m1.min_absolute_normal == 0.5
    assert e2m1.min_absolute_subnormal == 0.25
    for fmt, limit, steps, expected in [
        (Q.parse("E2M1"), 4, 100, [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]),
        (Q.parse("E3M0"), 10, 1000, [0, 0.125, 0.25, 0.5, 1, 2, 4, 8]),
        (Q.parse("E2M0"), 10, 1000, [0, 0.5, 1, 2]),
    ]:
        assert fmt.centroids == tuple(
            [-e for e in expected[1:][::-1]] + [0] + expected[1:]
        )
        x = torch.linspace(-limit, limit, steps=steps, dtype=dtype)
        y = fmt.quantise(x)
        assert y.dtype == dtype
        assert set(y.tolist()) == {n for absn in expected for n in [-absn, absn]}

    ue2m1 = Q.parse("UE2M1")
    assert ue2m1.bits == 3
    assert ue2m1.range == (0, 3)
    expected = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3]
    assert ue2m1.centroids == tuple(expected)
    assert set(ue2m1.quantise(torch.linspace(-4, 4, 100)).tolist()) == set(expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp_format_rounding(dtype: torch.dtype) -> None:
    eps = torch.finfo(dtype).eps
    x = tensor([1.5 - eps, 1.5, 1.5 + eps])
    assert torch.equal(Q.parse("E2M1-RN").quantise(x), torch.tensor([1.5, 1.5, 1.5]))
    assert torch.equal(Q.parse("E2M1-RI").quantise(x), torch.tensor([1.5, 1.5, 2.0]))
    assert torch.equal(Q.parse("E2M1-RZ").quantise(x), torch.tensor([1.0, 1.5, 1.5]))


def test_int_format() -> None:
    x = torch.linspace(-8, 8, 1001)

    fmt = Q.IntFormat(log2(8))
    assert str(fmt) == "E0M2"
    assert fmt.range == (-4, 3)
    assert fmt.centroids == tuple(range(-4, 3 + 1))
    assert set(fmt.quantise(x).tolist()) == set(range(-4, 3 + 1))
    assert fmt.count_bits((100,)) == 100 * 3

    fmt = Q.IntFormat(log2(9))
    assert fmt.range == (-4, 4)
    assert fmt.centroids == tuple(range(-4, 4 + 1))
    assert set(fmt.quantise(x).tolist()) == set(range(-4, 4 + 1))

    fmt = Q.IntFormat(log2(8), mode="symmetric")
    assert str(fmt) == "E0M2-S"
    assert fmt.range == (-3.5, 3.5)
    expected = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    assert fmt.centroids == tuple(expected)
    assert sorted(set(fmt.quantise(x).tolist())) == expected

    fmt = Q.IntFormat(log2(9), mode="symmetric")  # already symmetric
    assert fmt.range == (-4, 4)
    assert fmt.centroids == tuple(range(-4, 4 + 1))
    assert set(fmt.quantise(x).tolist()) == set(range(-4, 4 + 1))


def test_torch_format() -> None:
    assert str(Q.FP16) == "FLOAT16"
    assert Q.FP16.bits == 16
    assert Q.FP16.range == (-65504, 65504)
    assert torch.equal(
        Q.FP16.quantise(tensor([2**-25 * 0.99, 2**-25 * 1.01, -1, 1e5])),
        tensor([0, 2**-24, -1, 65504.0]),
    )


def test_exp_ceil_format() -> None:
    fmt = Q.parse("EXP6")
    amax = 2 ** (2**5)
    assert fmt.bits == 6
    assert fmt.range == (2 / amax, amax)
    torch.testing.assert_close(
        tensor(fmt.centroids).log2(), torch.arange(-31, 32 + 1).float()
    )
    # Always round up
    assert torch.equal(
        fmt.quantise(tensor([1.01 / amax, 2.000001, amax * 1.01], dtype=torch.float64)),
        tensor([2 / amax, 4, fmt.range[1]], dtype=torch.float64),
    )


def test_lut_format() -> None:
    fmt = Q.LUTFormat.create((-1, -0.125, 0.125, 1), "fours")
    assert str(fmt) == "LUT2[fours]"
    assert fmt.bits == 2
    assert fmt.range == (-1, 1)
    assert fmt.centroids == (-1, -0.125, 0.125, 1)
    assert torch.equal(
        fmt.quantise(tensor([0.8, 0.6, -0.001, -1.2])),
        tensor([1, 1, -0.125, -1]),
    )


def test_scalar_formats() -> None:
    for fmt in [
        # Torch
        Q.FP16,
        Q.BFLOAT16,
        Q.FP32,
        # LUT
        Q.NF4,
        Q.SF4_DF5,
        Q.nf_approx(5),
        # Float
        Q.parse("E0M3"),
        Q.parse("E2M2"),
        Q.parse("UE2M2-RI"),
        Q.ExpCeilFormat(3),
        # CRD
        Q.crd_normal(3),
        Q.crd_laplace(3),
        Q.crd_t(3, df=7, mode="asymmetric"),
        Q.crd_block_normal(3, 32),
        Q.crd_block_laplace(3, 32),
        Q.crd_block_t(3, 32, df=7, mode="asymmetric"),
    ]:
        assert 0 < fmt.range[1]
        assert 1 <= fmt.bits <= 32
        assert 600 <= fmt.count_bits((20, 30))

        x = torch.linspace(-20, 20, steps=100).view(2, 1, 50)
        if isinstance(fmt, Q.ExpCeilFormat):
            x.abs_()
        qx = fmt.quantise(x)
        assert qx.shape == x.shape
        assert torch.all(fmt.range[0] <= qx)
        assert torch.all(qx <= fmt.range[1])
        if not isinstance(fmt, Q.TorchFormat):
            centroids = tensor(fmt.centroids).float()
            closest = centroids[
                torch.bucketize(qx, (centroids[1:] + centroids[:-1]) / 2)
            ]
            torch.testing.assert_close(qx, closest)

        assert _json_roundtrip(fmt) == fmt


def test_lloyd_max_crd() -> None:
    torch.manual_seed(100)
    bits = 3
    df = 9  # somewhat sensitive for Laplace/StudentT ordering
    dists = [
        torch.distributions.Normal(0, 1),
        torch.distributions.Laplace(0, 2**-0.5),
        torch.distributions.StudentT(df, scale=((df - 2) / df) ** 0.5),
    ]
    crds = [Q.crd_normal(bits), Q.crd_laplace(bits), Q.crd_t(bits, df=df)]
    for i, dist in enumerate(dists):
        X = dist.sample((2**16,))
        rmse_lm = Q.qrmse_norm(Q.lut_lloyd_max(X, bits, 1e-4), X)
        rmse_crd = torch.stack([Q.qrmse_norm(fmt, X) for fmt in crds])
        assert rmse_crd.argmin() == i
        assert (rmse_crd.min() - rmse_lm).abs() < 0.1


# Wrappers


def test_scaled_format() -> None:
    fmt = Q.ScaledFormat.create(Q.LUTFormat.create((-1, -0.25, 0.25, 1), "fours"), 0.5)
    torch.testing.assert_close(fmt.quantise(tensor([0.1, -1])), tensor([0.125, -0.5]))
    assert fmt.centroids == (-0.5, -0.125, 0.125, 0.5)
    assert fmt.range == (-0.5, 0.5)

    fmt_clip = Q.ScaledFormat.create(fmt.format, 0.5, (-1, 1))
    assert fmt_clip.range == (-1, 1)


def test_random_rotation_format() -> None:
    # A random rotation on the heavy-tailed laplace should reduce RMSE
    torch.manual_seed(100)
    x = torch.distributions.Laplace(0, 1).sample((2**10, 2**12))
    fmt = Q.RandomRotationFormat(Q.parse("E2M1"), (0,), 100)
    rmse_rotated = Q.qrmse_norm(fmt, x).item()
    rmse_original = Q.qrmse_norm(fmt.format, x).item()
    assert rmse_rotated < 0.9 * rmse_original
    assert fmt.count_bits((100,)) == 400


def test_sparse_format() -> None:
    x = torch.tensor([1, 2, -1, -1000, 0, 0, 0, 900]).float().view(2, -1)
    fmt = Q.SparseFormat(Q.parse("E2M1"), Q.FP32, 1 / 4)
    assert Q.qrmse_norm(fmt, x).item() == 0
    assert Q.qrmse_norm(fmt.format, x).item() > 0.1
    assert fmt.count_bits((2, 4)) == 8 * 4 + (8 / 4) * (32 + 32)


def test_block_normalise() -> None:
    # Check the case where the normalisation axis is all-zero
    xs = torch.arange(3, dtype=torch.float32)[:, None].broadcast_to((3, 4))
    for scaling in ["absmax", "rms"]:
        torch.testing.assert_close(
            Q.block_normalise(xs, (1, None), scaling, (-1, 1), Q.FP32)[0],
            torch.tensor([0.0, 1.0, 1.0])[:, None].broadcast_to((3, 4)),
            msg=f"scaling={scaling}",
        )


def test_linear_scaling_format() -> None:
    fmt = Q.LinearScalingFormat(Q.IntFormat(2), Q.FP32, (3,), "absmax")
    assert "absmax" in str(fmt)
    assert fmt.count_bits((12,)) == 2 * 12 + 4 * 32
    torch.testing.assert_close(
        fmt.quantise(tensor([100.0, -60, 40, -40, 10, 21])),
        tensor([100.0, -100, 0, -40, 0, 40]),
    )
    assert json.loads(json.dumps(dataclasses.asdict(fmt)))["block_shape"] == [3]

    fmt = Q.LinearScalingFormat(Q.IntFormat(2), Q.FP32, (2,), "signmax")
    torch.testing.assert_close(
        fmt.quantise(tensor([-20.0, 9, 40, -15])),
        tensor([-20.0, 10, 40, -20]),
    )


def test_compressed_lut_format() -> None:
    fmt = Q.CompressedLUTFormat(
        Q.LUTFormat((-2, -1, 0, 1, 2), "fivep", (-2, 2)),
        tensor([1 / 8, 1 / 8, 1 / 2, 1 / 8, 1 / 8]).log(),
        "optimal",
    )
    torch.testing.assert_close(fmt.quantise(tensor([-3, 1.2])), tensor([-2.0, 1]))
    assert fmt.count_bits_tensor(tensor([0, 0, 0, 0, 3])) == 4 * 1 + 3
