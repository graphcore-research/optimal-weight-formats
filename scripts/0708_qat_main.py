# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import math

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

EXPERIMENT = "20250708-qat-main"

_ASYM = dict(mode="asymmetric")
FORMATS = [
    fmt
    for b in [3, 4, 5]
    for fmt in [
        F.Scaled(b, "int", Q.BFLOAT16, (None, None), "rms", compressor="optimal"),
        F.Scaled(
            b, "t", Q.BFLOAT16, (None, None), "rms", sparse_ratio=2**-10, args=_ASYM
        ),
        F.Scaled(b, "t", Q.BFLOAT16, (1, 64), "absmax", args=_ASYM),
        F.Scaled(b, "t", Q.BFLOAT16, (1, None), "absmax", args=_ASYM),
        F.Scaled(b, "t", Q.BFLOAT16, (None, None), "absmax", args=_ASYM),
        F.Scaled(b, "t", Q.BFLOAT16, (None, None), "rms", args=_ASYM),
    ]
]
MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
]


def runs_baseline() -> list[EQ.Run]:
    return [
        EQ.Run(
            EXPERIMENT,
            model=model,
            test=EQ.Baseline(),
            train=EQ.TrainingSettings(steps=0, batch_size=1, log_interval=1),
            opt=EQ.OptimiserSettings(lr=0),
            exe=EQ.ExecutionSettings(data_parallel=1),
            tag="baseline",
        )
        for model in MODELS
    ]


def runs_direct_cast() -> list[EQ.Run]:
    return [
        EQ.Run(
            EXPERIMENT,
            model=model,
            test=EQ.QAT(fmt, scaling_mode="dynamic", clip_gradient=False),
            train=EQ.TrainingSettings(steps=0, batch_size=1, log_interval=1),
            opt=EQ.OptimiserSettings(lr=0),
            exe=EQ.ExecutionSettings(data_parallel=1),
            tag="direct-cast",
        )
        for model in MODELS
        for fmt in FORMATS
    ]


def runs_qat() -> list[EQ.Run]:
    return [
        EQ.Run(
            EXPERIMENT,
            model=model,
            test=EQ.QAT(fmt, scaling_mode="dynamic", clip_gradient=False),
            train=EQ.TrainingSettings(steps=8192, batch_size=64, log_interval=128),
            opt=EQ.OptimiserSettings(lr=2**-17),
            exe=EQ.ExecutionSettings(data_parallel=4),
            tag="qat",
        )
        for model in MODELS[:2]
        for fmt in FORMATS
    ]


def runs_batch_lr_duration() -> list[EQ.Run]:
    fmt = F.Scaled(3, "t", Q.BFLOAT16, (1, 64), "absmax", args=_ASYM)
    return [
        EQ.Run(
            EXPERIMENT,
            model="meta-llama/Llama-3.2-3B",
            test=EQ.QAT(fmt, scaling_mode="dynamic", clip_gradient=False),
            train=EQ.TrainingSettings(
                steps=examples // batch_size, batch_size=batch_size, log_interval=128
            ),
            opt=EQ.OptimiserSettings(lr=lr),
            exe=EQ.ExecutionSettings(data_parallel=8),
            tag="batch-lr-duration",
        )
        for batch_size in [64, 128]
        for lr in [2**-17, 2**-18]
        for examples in [2**19, 2**20, 2**21]
    ]


def runs_bits_lr() -> list[EQ.Run]:
    return [
        EQ.Run(
            EXPERIMENT,
            model="meta-llama/Llama-3.2-3B",
            test=EQ.QAT(
                F.Scaled(b, "t", Q.BFLOAT16, (1, 64), "absmax", args=_ASYM),
                scaling_mode="dynamic",
                clip_gradient=False,
            ),
            train=EQ.TrainingSettings(steps=8192, batch_size=64, log_interval=128),
            opt=EQ.OptimiserSettings(lr=lr),
            exe=EQ.ExecutionSettings(data_parallel=4),
            tag="bits-lr",
        )
        for b in [3, 4, 5]
        for lr in [2**-16, 2**-17, 2**-18, 2**-19]
    ]


def runs_bits_lr_8b() -> list[EQ.Run]:
    return [
        EQ.Run(
            EXPERIMENT,
            model="meta-llama/Llama-3.1-8B",
            test=EQ.QAT(
                F.Scaled(b, "t", Q.BFLOAT16, (1, 64), "absmax", args=_ASYM),
                scaling_mode="dynamic",
                clip_gradient=False,
            ),
            train=EQ.TrainingSettings(steps=8192, batch_size=64, log_interval=128),
            opt=EQ.OptimiserSettings(lr=lr),
            exe=EQ.ExecutionSettings(data_parallel=8),
            tag="bits-lr-8b",
        )
        for b, lrs in [(3, [2**-18, 2**-17, 2**-19]), (5, [2**-20, 2**-19, 2**-21])]
        for lr in lrs
    ]


def runs_qat_v2() -> list[EQ.Run]:
    def get_lr(model: str, fmt: F.Scaled) -> float:
        model_loglr_3b = {"1B": -17, "3B": -17, "8B": -18}[model.split("-")[-1]]
        return 2 ** (model_loglr_3b - fmt.element_bits + 3)

    return [
        EQ.Run(
            EXPERIMENT,
            model=model,
            test=EQ.QAT(fmt, scaling_mode="dynamic", clip_gradient=False),
            train=EQ.TrainingSettings(steps=8192, batch_size=64, log_interval=128),
            opt=EQ.OptimiserSettings(lr=get_lr(model, fmt)),
            exe=EQ.ExecutionSettings(data_parallel=8),
            tag="qat",
        )
        for model in MODELS
        for fmt in FORMATS
    ]


if __name__ == "__main__":
    runs = []
    # runs.extend(runs_baseline())
    # runs.extend(runs_direct_cast())
    # runs.extend(runs_qat())
    # runs.extend(runs_batch_lr_duration())
    # runs.extend(runs_bits_lr())
    # runs.extend(runs_bits_lr_8b())
    runs.extend(runs_qat_v2())

    for run in runs:
        print(
            f"##### {run.model}  {getattr(run.test, 'fmt', None)}  2**{math.log2(run.opt.lr)}"
        )
    print("#####", len(runs), "total")

    EQ.submit_sweep(runs)
