# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import torch

import weight_formats.experiments.qat as EQ

if __name__ == "__main__":
    # WIP
    EQ.run_train(
        "meta-llama/Llama-3.2-1B",
        EQ.Training(
            lr=2**-18,
            steps=10,
            log_interval=5,
            sequence_length=1024,
            batch_size=1,
            valid_sequences=1,
            perturb_ratio=0.1,
            data_parallel=1,
        ),
        experiment=None,
    )
