# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import dataclasses
import time
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import oe_eval.models.eleuther_huggingface
import oe_eval.run_eval
import oe_eval.tasks.oe_eval_tasks
import torch
import transformers

from .. import model_quantisation as M
from . import core, fisher


@dataclass
class Task:
    name: str
    limit: int | None


TASKS = [
    Task(name=name, limit=d.get("limit"))
    for base_name, d in [
        ("arc_challenge", {}),
        ("arc_easy", {}),
        ("boolq", {}),
        ("csqa", {}),
        ("hellaswag", dict(limit=1000)),
        ("openbookqa", {}),
        ("piqa", {}),
        ("socialiqa", {}),
        ("winogrande", {}),
    ]
    for name in [base_name, f"{base_name}:mc"]
]


def evaluate(model: transformers.PreTrainedModel, task: Task) -> dict[str, Any]:
    t0 = time.time()
    oe_task = oe_eval.tasks.oe_eval_tasks.TASK_REGISTRY[task.name](task_name=task.name)
    oe_task.download()
    oe_task.build_all_requests(limit=task.limit)
    oe_model = oe_eval.models.eleuther_huggingface.HFLM_Verbose(model)
    outputs = oe_eval.run_eval.evaluate(
        oe_model,
        instances=oe_task._instances,
        task_config=oe_task.task_config,
        model_config={},
    )
    results = {}
    for metric in oe_task.make_metrics():
        metric.compute_for_docs(outputs)
        results.update(
            metric.aggregate_to_task(
                primary_metric=oe_task.task_config.get("primary_metric")
            )
        )
    results["_duration"] = time.time() - t0
    return results


@dataclass
class Run:
    experiment: str
    model: str
    tasks: list[Task]
    fmt: M.FmtSpec | None
    bit_allocation: Literal["fixed", "variable"] = "fixed"
    error_weight: Literal[None, "fisher", "parameter"] = None
    device: torch.device = core.FIELD_DEVICE
    type: str = "downstream"


class _Runner:
    def __init__(self):
        self.model = None

    def __call__(self, run: Run) -> None:
        if self.model is None or self.model.model.config._name_or_path != run.model:
            self.model = core.RequantisableModel.load(
                run.model, device=run.device, dtype=torch.bfloat16
            )

        config = dataclasses.asdict(run)
        config["fmt_str"] = str(run.fmt) if run.fmt else None
        with core.Experiment(config) as experiment:
            self.model.reset()
            if run.fmt is None:
                quantisation_log = M.no_quantisation(self.model.model)
            elif run.bit_allocation == "fixed":
                quantisation_log = M.quantise_2d_fixed(
                    self.model.model, run.fmt, error_weight=run.error_weight
                )
            elif run.bit_allocation == "variable":
                fisher_sum = fisher.fetch_fisher_sum(
                    self.model.model.config._name_or_path
                )
                quantisation_log = M.quantise_2d_variable(
                    self.model.model,
                    run.fmt,
                    fisher_sum=fisher_sum,
                    min_element_bits=None,
                    error_weight=run.error_weight,
                )
            else:
                raise ValueError(f"Unexpected bit_allocation={run.bit_allocation!r}")

            experiment.summary(
                **quantisation_log,
                direct_cast={
                    task.name: evaluate(self.model.model, task) for task in run.tasks
                },
            )


def run_sweep(runs: Iterable[Run], processes: int | None = None) -> None:
    core.run_sweep(_Runner, [(run,) for run in runs], processes=processes)
