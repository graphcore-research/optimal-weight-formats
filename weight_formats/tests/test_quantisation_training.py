# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import copy

import torch
from torch import nn, tensor

from .. import fit as F
from .. import quantisation as Q
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


def test_weight_matches_format() -> None:
    torch.manual_seed(100)
    reference_weight = torch.randn(256, 768).pow(3).bfloat16()

    for bits in [3, 4]:
        for element_family in ["int", "fp", "normal", "lloyd_max"]:
            for block, scaling in [
                ((None, None), "rms"),
                ((1, None), "absmax"),
                ((1, 64), "signmax"),
            ]:
                for compressor in [None, "optimal"]:
                    for sparse_ratio in [0, 1e-2]:
                        if compressor is not None and element_family != "int":
                            continue  # unsupported
                        if scaling == "signmax" and element_family != "int":
                            continue  # requires other args / unsupported

                        format = F.Scaled(
                            bits,
                            element_family=element_family,
                            scale_format=Q.BFLOAT16,
                            block_shape=block,
                            scaling=scaling,
                            sparse_ratio=sparse_ratio,
                            sparse_format=Q.BFLOAT16,
                            compressor=compressor,
                            args=dict() if compressor is None else dict(smoothing=0),
                        ).fit(reference_weight)
                        format_error = Q.qrmse_norm(format, reference_weight)
                        format_bits = format.count_bits_tensor(reference_weight)

                        for scaling_mode in ["parameter", "dynamic"]:
                            weight = T.Weight(
                                reference_weight,
                                format,
                                scaling_mode,
                                clip_gradient=False,
                            )
                            # don't count centroids bits
                            weight.centroids.requires_grad_(False)
                            weight_error = Q.rmse_norm(reference_weight, weight())
                            weight_bits = weight.count_bits(reference_weight.dtype)
                            # print(f"{format} scaling_mode={scaling_mode}")
                            torch.testing.assert_close(
                                weight_error,
                                format_error,
                                rtol=0.02,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                tensor(weight_bits).float(),
                                tensor(format_bits).float(),
                                # Because weight() doesn't count +/- zero separately
                                rtol=0.1 if element_family == "fp" else 1e-3,
                                atol=0,
                            )


def test_weight_gradients() -> None:
    torch.manual_seed(100)
    reference_weight = torch.linspace(-10, 10, 128).bfloat16()
    weight = T.Weight(
        reference_weight,
        F.Scaled(
            3,
            "int",
            Q.BFLOAT16,
            (32,),
            scaling="absmax",
            sparse_format=Q.BFLOAT16,
            sparse_ratio=3 / 128,
        ).fit(reference_weight),
        scaling_mode="parameter",
        clip_gradient=False,
    )
    w = weight()
    w.backward(torch.ones_like(w))

    assert weight.master.grad.sum() == 125
    assert weight.scale.grad.ne(0).any(), "hard to predict scale.grad"
    assert weight.centroids.grad.ne(0).all()
    assert weight.sparse_weight.grad.equal(torch.full((3,), 1.0))


def test_convert_embedding() -> None:
    torch.manual_seed(100)
    model = nn.Embedding(100, 16, padding_idx=0)
    input = torch.randint(0, 100, (5,))
    T.convert(
        model,
        F.Scaled(3, "int", Q.BFLOAT16, (1, 8), "absmax"),
        "dynamic",
        clip_gradient=False,
        error_weight=None,
    )
    assert model(input).shape == (5, 16)


def test_convert_and_train() -> None:
    torch.manual_seed(100)
    reference = nn.Sequential(
        nn.Linear(64, 512, bias=False),
        nn.RMSNorm((512,)),
        nn.Linear(512, 64, bias=True),
    )
    input = torch.randn(2048, 64)
    with torch.no_grad():
        reference_out = reference(input)

    model = copy.deepcopy(reference)
    T.convert(
        model,
        F.Scaled(
            element_bits=3,
            element_family="int",
            scale_format=Q.BFLOAT16,
            block_shape=(1, 64),
            scaling="absmax",
            scaling_match="moments",
            sparse_format=Q.BFLOAT16,
            sparse_ratio=1e-3,
        ),
        scaling_mode="parameter",
        clip_gradient=True,
        error_weight=None,
    )

    bpp = T.count_bits(model, torch.bfloat16) / T.count_parameters(model)
    assert 3.25 < bpp < 4

    opt = torch.optim.Adam(
        [
            dict(params=T.get_named_parameters(model, "weight"), lr=1e-4),
            dict(params=T.get_named_parameters(model, "scale"), lr=1e-4),
            dict(params=T.get_named_parameters(model, "centroids"), lr=0),
            dict(params=T.get_named_parameters(model, "other"), lr=0),
        ],
        lr=0,
        betas=(0.9, 0.9),
    )
    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = (model(input) - reference_out).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.75

    # Check we can reload using T.save -> T.load_convert
    reloaded = copy.deepcopy(reference)
    assert not torch.allclose(reloaded(input), model(input))
    assert T.count_bits(reloaded, torch.bfloat16) != T.count_bits(model, torch.bfloat16)

    T.load_convert(reloaded, T.save(model))
    torch.testing.assert_close(reloaded(input), model(input))
    assert T.count_bits(reloaded, torch.bfloat16) == T.count_bits(model, torch.bfloat16)
