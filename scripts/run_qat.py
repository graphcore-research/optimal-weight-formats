# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

if __name__ == "__main__":
    runs = [
        EQ.Run(
            "dev",
            model="meta-llama/Llama-3.2-1B",
            test=EQ.QAT(
                F.Scaled(
                    4,
                    "int",
                    Q.BFLOAT16,
                    (1, 64),
                    "absmax",
                    "moments",
                ),
                scaling_mode="dynamic",
                clip_gradient=False,
            ),
            train=EQ.TrainingSettings(
                steps=16,
                batch_size=16,
                log_interval=8,
                valid_sequences=2,
            ),
            opt=EQ.OptimiserSettings(lr=2**-18),
            exe=EQ.ExecutionSettings(),
            tasks=(EQ.Task("arc_easy:mc"),),
        )
    ]
    for run in runs:
        EQ.run(run)
