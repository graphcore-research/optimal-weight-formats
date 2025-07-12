# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

EXPERIMENT = "20250630-qat-variants"
MODEL = "meta-llama/Llama-3.2-1B"


def runs_v0() -> list[EQ.Run]:
    fmt = F.Scaled(4, "int", Q.BFLOAT16, (1, 64), "absmax", "moments")
    return [
        EQ.Run(
            EXPERIMENT,
            model=MODEL,
            test=EQ.QAT(
                fmt, scaling_mode=c["scaling_mode"], clip_gradient=c["clip_gradient"]
            ),
            train=EQ.TrainingSettings(
                steps=4096,
                batch_size=64,
                log_interval=128,
            ),
            opt=EQ.OptimiserSettings(
                lr=lr, lr_modifiers=EQ.LRModifiers(scale=c["lr_scale"])
            ),
            exe=EQ.ExecutionSettings(),
            tag="v0",
        )
        for lr in [2**-17, 2**-18, 2**-16, 2**-19]
        for c in [
            # Parameter
            dict(
                scaling_mode="parameter",
                clip_gradient=False,
                lr_scale=1.0,
            ),
            dict(
                scaling_mode="parameter",
                clip_gradient=True,
                lr_scale=1.0,
            ),
            dict(
                scaling_mode="parameter",
                clip_gradient=False,
                lr_scale=0.0,
            ),
            dict(
                scaling_mode="parameter",
                clip_gradient=True,
                lr_scale=0.0,
            ),
            # Dynamic
            dict(
                scaling_mode="dynamic",
                clip_gradient=False,
                lr_scale=1.0,
            ),
        ]
    ]


FORMATS = [
    F.Scaled(4, "int", Q.BFLOAT16, (None, None), "rms", compressor="optimal"),
    F.Scaled(
        4,
        "t",
        Q.BFLOAT16,
        (None, None),
        "rms",
        sparse_ratio=2**-10,
        args=dict(mode="asymmetric"),
    ),
    F.Scaled(4, "t", Q.BFLOAT16, (1, 64), "absmax", args=dict(mode="asymmetric")),
    F.Scaled(4, "t", Q.BFLOAT16, (1, None), "absmax", args=dict(mode="asymmetric")),
    F.Scaled(4, "t", Q.BFLOAT16, (None, None), "absmax", args=dict(mode="asymmetric")),
    F.Scaled(4, "t", Q.BFLOAT16, (None, None), "rms", args=dict(mode="asymmetric")),
]


def runs_direct_cast() -> list[EQ.Run]:
    return [
        EQ.Run(
            EXPERIMENT,
            model=MODEL,
            test=test,
            train=EQ.TrainingSettings(steps=0, batch_size=1, log_interval=1),
            opt=EQ.OptimiserSettings(0),
            exe=EQ.ExecutionSettings(),
            tag="direct-cast",
        )
        for test in [EQ.Baseline()]
        + [EQ.QAT(fmt, scaling_mode="dynamic", clip_gradient=False) for fmt in FORMATS]
    ]


def runs_formats() -> list[EQ.Run]:
    return [
        EQ.Run(
            EXPERIMENT,
            model=MODEL,
            test=EQ.QAT(fmt, **qat_args),
            train=EQ.TrainingSettings(steps=2**12, batch_size=64, log_interval=128),
            opt=EQ.OptimiserSettings(
                lr=2**-17, lr_modifiers=EQ.LRModifiers(centroids=float(train_centroids))
            ),
            exe=EQ.ExecutionSettings(),
            tag="formats",
        )
        for qat_args, train_centroids in [
            (dict(scaling_mode="dynamic", clip_gradient=False), False),
            (dict(scaling_mode="parameter", clip_gradient=True), False),
            (dict(scaling_mode="dynamic", clip_gradient=False), True),
        ]
        for fmt in FORMATS
    ]


def runs_scale_up() -> list[EQ.Run]:
    model = "meta-llama/Llama-3.1-8B"
    fmt = F.Scaled(3, "t", Q.BFLOAT16, (1, 64), "absmax", args=dict(mode="asymmetric"))
    return [
        EQ.Run(
            EXPERIMENT,
            model=model,
            test=EQ.QAT(fmt, "dynamic", clip_gradient=False),
            train=EQ.TrainingSettings(steps=2**12, batch_size=64, log_interval=128),
            opt=EQ.OptimiserSettings(lr=lr),
            exe=EQ.ExecutionSettings(),
            tag="scale-up",
        )
        for lr in [2**-19]
    ]


if __name__ == "__main__":
    # runs = runs_v0()
    # runs = runs_direct_cast()
    # runs = runs_formats()
    runs = runs_scale_up()
    EQ.run_sweep(runs)
