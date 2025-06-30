# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

if __name__ == "__main__":
    fmt = F.Scaled(4, "int", Q.BFLOAT16, (1, 64), "absmax", "moments")
    tests = [
        EQ.QAT(fmt, scaling_mode="dynamic", clip_gradient=False),
        # EQ.PerturbQuantise(fmt),
    ]
    runs = [
        EQ.Run(
            "20250625-qat-initial-sweep",
            model="meta-llama/Llama-3.2-1B",
            test=test,
            train=EQ.TrainingSettings(
                steps=2**18 // batch_size,
                batch_size=batch_size,
                log_interval=128,
            ),
            opt=EQ.OptimiserSettings(lr=lr),
            exe=EQ.ExecutionSettings(),
            tag="longer-runs",
        )
        for test in tests
        for lr in [2**-18, 2**-20]
        for batch_size in [64, 16]
    ]
    for n, run in enumerate(runs):
        print(f"### {n+1}/{len(runs)} {run.to_config()}")
        EQ.run(run)
