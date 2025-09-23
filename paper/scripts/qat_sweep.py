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


def runs_qat_v2() -> list[EQ.Run]:
    def get_lr(fmt: F.Scaled) -> float:
        return 2 ** (-17 - fmt.element_bits + 3)

    return [
        EQ.Run(
            EXPERIMENT,
            model=model,
            test=EQ.QAT(fmt, scaling_mode="dynamic", clip_gradient=False),
            train=EQ.TrainingSettings(steps=8192, batch_size=64, log_interval=128),
            opt=EQ.OptimiserSettings(lr=get_lr(fmt)),
            exe=EQ.ExecutionSettings(
                data_parallel={"1B": 4, "3B": 4, "8B": 8}[model.split("-")[-1]]
            ),
            tag="qat-v2",
        )
        for model in MODELS
        for fmt in FORMATS
    ]


if __name__ == "__main__":
    runs = []
    runs.extend(runs_baseline())
    runs.extend(runs_direct_cast())
    runs.extend(runs_qat_v2())

    for run in runs:
        print(
            f"##### {run.model}  {getattr(run.test, 'fmt', None)}  2**{math.log2(run.opt.lr)}  DP={run.exe.data_parallel}"
        )
    print("#####", len(runs), "total")

    EQ.submit_sweep(runs)
