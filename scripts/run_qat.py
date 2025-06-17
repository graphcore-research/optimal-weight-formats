import weight_formats.experiments.qat as EQ
import torch

if __name__ == "__main__":
    # WIP
    print(
        EQ.run_train(
            "meta-llama/Llama-3.2-1B",
            EQ.Training(
                lr=2**-18,
                steps=10,
                sequence_length=1024,
                batch_size=64,
                perturb_ratio=0.1,
                data_parallel=4,
            ),
        )
    )
