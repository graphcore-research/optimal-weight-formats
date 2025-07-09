# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

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


def runs_baseline(model: str) -> list[EQ.Run]:
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
    ]


def runs_direct_cast(model: str) -> list[EQ.Run]:
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
        for fmt in FORMATS
    ]


if __name__ == "__main__":
    runs = []
    for model in MODELS:
        # runs.extend(runs_baseline(model))
        runs.extend(runs_direct_cast(model))

    # for run in runs:
    #     print("#####", run.model, run.test)
    # print("#####", len(runs), "total")
    EQ.run_sweep(runs)
