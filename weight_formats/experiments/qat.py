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
from torch import Tensor, nn

from .. import model_quantisation as M
from . import core, fisher


# Quantisation Aware Training


class XEnt_Destructive(torch.autograd.Function):
    """A somewhat dangerous in-place Cross-Entropy, optimised for memory-efficiency.

    Both forward and backward passes mutate `input_logits` in-place. It is unsafe for
    `input_logits` to be consumed by any other operation.
    """

    @staticmethod
    def forward(ctx, input_logits, target_p, mask):
        input_logp = input_logits.sub_(torch.logsumexp(input_logits, -1, keepdim=True))
        del input_logits
        ctx.save_for_backward(input_logp, target_p, mask)
        return torch.dot(target_p.flatten(), input_logp.flatten()).neg()

    @staticmethod
    def backward(ctx, grad_output):
        input_logp, target_p, mask = ctx.saved_tensors
        grad_input_logits = (
            input_logp.exp_().sub_(target_p).mul_(mask).mul_(grad_output)
        )
        del input_logp
        return grad_input_logits, None, None


def compute_kl_loss(
    model: nn.Module, reference_model: nn.Module, batch: dict[str, Tensor]
) -> Tensor:
    """Computes the KL divergence between the output of two models, with care for memory."""
    with torch.no_grad():
        reference_logp = torch.log_softmax(reference_model(**batch).logits, -1)
        reference_p = reference_logp.exp().mul_(batch["attention_mask"].unsqueeze(-1))
        # Calculate reference entropy here, so we can free up `reference_logp`
        reference_ent = -torch.dot(reference_p.flatten(), reference_logp.flatten())
        del reference_logp

    xent = XEnt_Destructive.apply(
        model(**batch).logits,
        reference_p,
        batch["attention_mask"].unsqueeze(-1),
    )
    return xent - reference_ent


# Downstream Tasks


@dataclass
class Task:
    name: str
    limit: int | None


TASKS = [
    Task(name=name, limit=d.get("limit"))
    for name, d in [
        # Selected cloze vs multiple-choice based on a baseline sweep
        # named "20250611-downstream-baselines"
        ("arc_challenge:mc", {}),
        ("arc_easy:mc", {}),
        ("boolq", {}),
        ("csqa:mc", {}),
        ("hellaswag", dict(limit=1000)),
        ("openbookqa:mc", {}),
        ("piqa", {}),
        ("socialiqa:mc", {}),
        ("winogrande", {}),
    ]
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


# Top-level


@dataclass
class Run:
    experiment: str
    model: str
    tasks: list[Task]
    fmt: M.FmtSpec | None
    bit_allocation: Literal["fixed", "variable"] = "fixed"
    error_weight: Literal[None, "fisher", "parameter"] = None
    device: torch.device = core.FIELD_DEVICE
    type: str = "qat"


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
