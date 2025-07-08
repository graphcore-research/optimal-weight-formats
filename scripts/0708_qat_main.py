# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

EXPERIMENT = "20250708-qat-main"


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


if __name__ == "__main__":
    models = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.1-8B",
    ]
    runs = []
    for model in models:
        runs.extend(runs_baseline(model))

    # for run in runs:
    #     print("#####", run.model, run.test)
    EQ.run_sweep(runs)
