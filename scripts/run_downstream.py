# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments.qat as EQ
import weight_formats.fit as F
import weight_formats.quantisation as Q

if __name__ == "__main__":
    formats = [
        None,
        F.Scaled(
            element_bits=4,
            element_family="fp",
            scale_format=Q.BFLOAT16,
            block_shape=(1, 64),
            scaling="absmax",
        ),
    ]
    EQ.run_sweep(
        [
            EQ.Run("dev", "meta-llama/Llama-3.2-1B", tasks=EQ.TASKS[:2], fmt=fmt)
            for fmt in formats
        ]
    )
