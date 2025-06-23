import dataclasses
from torch import multiprocessing

import weight_formats.experiments as E
import weight_formats.experiments.qat as EQ


def run(xp: str, model_id: str, settings: EQ.Training) -> None:
    with E.core.Experiment(
        dict(experiment=xp, **dataclasses.asdict(settings))
    ) as experiment:
        EQ.run_train(model_id, settings, experiment)


if __name__ == "__main__":
    xp = "20250619-denoise-lr-sweep"
    model = "meta-llama/Llama-3.2-1B"
    settingses = [
        EQ.Training(
            ## Long-run
            steps=16384,
            log_interval=256,
            valid_sequences=128,
            lr_schedule="linear",
            #
            ## Short-run
            # steps=1024,
            # log_interval=64,
            # valid_sequences=64,
            # lr_schedule="constant",
            #
            ## Settings
            lr=lr,
            batch_size=8,
            adam_betas=adam_betas,
            #
            ## Fixed
            perturb_ratio=0.1,
            sequence_length=1024,
            compile="default",
        )
        for adam_betas in [
            # (0.9, 0.9),
            (0.9, 0.95),
            # (0.9, 0.99),
            (0.8, 0.9),
            # (0.8, 0.95),
            # (0.8, 0.99),
        ]
        for lr in [2**n for n in [-18]]  # [-18, -16, -20]
    ]
    with multiprocessing.Pool(maxtasksperchild=1) as pool:
        for settings in settingses:
            try:
                pool.apply(run, (xp, model, settings))
            except Exception as e:
                print("ERROR", e)
