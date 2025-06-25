# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

if __name__ == "__main__":
    run = EQ.Run(
        "dev",
        model="meta-llama/Llama-3.2-1B",
        test=EQ.QAT(
            F.Scaled(
                3,
                "int",
                Q.BFLOAT16,
                (1, 64),
                "absmax",
                "moments",
                sparse_format=Q.BFLOAT16,
                sparse_ratio=1e-4,
            ),
            scaling_mode="dynamic",
            clip_gradient=False,
        ),
        train=EQ.TrainingSettings(
            steps=32,
            sequence_length=1024,
            batch_size=8,
            log_interval=8,
            valid_sequences=8,
        ),
        opt=EQ.OptimiserSettings(lr=2**-18),
        exe=EQ.ExecutionSettings(),
        tasks=(EQ.Task("arc_easy:mc"),),
    )
    EQ.run(run)
