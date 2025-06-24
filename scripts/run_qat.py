# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

if __name__ == "__main__":
    run = EQ.Run(
        "dev",
        model="meta-llama/Llama-3.2-1B",
        test=EQ.QAT(
            F.Scaled(3, "int", Q.BFLOAT16, (1, 64), "absmax", "moments"),
            scaling_mode="dynamic",
            clip_gradient=False,
        ),
        train=EQ.TrainingSettings(
            steps=16,
            sequence_length=1024,
            batch_size=4,
            log_interval=4,
            valid_sequences=8,
        ),
        opt=EQ.OptimiserSettings(lr=2**-18),
        exe=EQ.ExecutionSettings(
            compile=None,  # compute_dtype=torch.float32, reference_dtype=torch.float32
        ),
        tasks=(EQ.Task("arc_easy:mc"),),
    )
    EQ.run(run)
    # torch.set_default_device(torch.device("cuda"))
    # EQ._run_worker(run)
