# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import contextlib
import copy
import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import datasets
import oe_eval.models.eleuther_huggingface
import oe_eval.run_eval
import oe_eval.tasks.oe_eval_tasks
import torch
import torch.multiprocessing as multiprocessing
import transformers
from torch import Tensor, nn
from torch.distributed import fsdp

from .. import model_quantisation as M
from . import core, fisher, token_prediction

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


@dataclass
class Training:
    lr: float
    steps: int
    sequence_length: int
    batch_size: int
    log_interval: int
    valid_sequences: int
    perturb_ratio: float
    data_parallel: int = 1
    compile: str | None = None
    memory_profile: str | None = None
    params_dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.bfloat16
    reference_dtype: torch.dtype = torch.bfloat16
    dataset: str = "DKYoon/SlimPajama-6B"
    adam_eps: float = 1e-5


def _fully_shard_model(model: nn.Module, **args: Any) -> None:
    fsdp.fully_shard(model.model.embed_tokens, **args)
    for layer in model.model.layers:
        fsdp.fully_shard(layer, **args)
    fsdp.fully_shard(model.lm_head, **args)
    fsdp.fully_shard(model, **args)


class _Logger:
    def __init__(
        self,
        settings: Training,
        rank: int,
        experiment: core.Experiment | None,
        valid_data: token_prediction.Dataset | None,
        model: nn.Module,
    ):
        self.settings = settings
        self.rank = rank
        self.experiment = experiment
        self.valid_data = valid_data
        self.model = model

        # State
        self._step = 0
        self._logged_step = 0
        self._t0 = time.time()
        self._total_loss = torch.tensor(0.0)
        self._total_count = torch.tensor(0)
        self._log = core.AttrDict(loss=[], duration=[])
        self._validate_and_log()

    def _validate_and_log(self) -> None:
        # Compute statistics
        if self.settings.data_parallel > 1:
            torch.distributed.all_reduce(self._total_loss)
            torch.distributed.all_reduce(self._total_count)
        t1 = time.time()  # don't count validation
        if self._step == self._logged_step:
            self._log.loss.append(None)
            self._log.duration.append(None)
        else:
            self._log.loss.append(self._total_loss.div(self._total_count).item())
            self._log.duration.append(
                (t1 - self._t0) / (self._step - self._logged_step)
            )
        if self.valid_data is not None:
            for k, v in self.valid_data.evaluate(self.model).items():
                self._log.setdefault(f"valid_{k}", []).append(v.mean().item())

        # Write logs
        if self.rank == 0:
            print(
                f"{self._step:>04}:  "
                + "  ".join(
                    f"{k}={v[-1]:.3f}"
                    for k, v in self._log.items()
                    if v[-1] is not None
                ),
                file=sys.stderr,
            )
        if self.experiment is not None:
            self.experiment.summary(train=self._log)

        # Reset
        self._total_loss.zero_()
        self._total_count.zero_()
        self._t0 = time.time()
        self._logged_step = self._step

    def log(self, loss: Tensor, count: Tensor) -> None:
        self._total_loss.add_(loss.float())
        self._total_count.add_(count)
        self._step += 1
        if self._step % self.settings.log_interval == 0:
            self._validate_and_log()


def train(
    model_id: str, settings: Training, experiment: core.Experiment | None
) -> nn.Module:
    with contextlib.ExitStack() as exit:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if rank == 0 and settings.memory_profile:
            exit.enter_context(core.cuda_memory_history(Path(settings.memory_profile)))

        # Loading
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=torch.get_default_device(),
            torch_dtype=settings.reference_dtype,
        )
        model = copy.deepcopy(reference_model).to(settings.params_dtype)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        data = datasets.load_dataset("DKYoon/SlimPajama-6B")["train"]
        valid_data = (
            token_prediction.Dataset.load(
                reference_model,
                sequence_length=4096,
                batch_size=1,
                kl_topk=128,
                sequence_limit=settings.valid_sequences,
            )
            if settings.valid_sequences
            else None
        )

        # Modifying
        for p in model.parameters():
            if p.ndim == 2:
                p.data += torch.randn_like(p).mul(p.std() * settings.perturb_ratio)

        # Preparing
        if torch.distributed.is_initialized():
            assert settings.data_parallel == torch.distributed.get_world_size()
            data = data.shard(
                torch.distributed.get_world_size(), torch.distributed.get_rank()
            )
            _fully_shard_model(
                model,
                mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=settings.compute_dtype),
            )
        else:
            assert settings.data_parallel == 1
            assert settings.compute_dtype == settings.params_dtype
            assert settings.params_dtype == settings.reference_dtype

        exit.enter_context(fisher.activation_checkpointing_enabled(model))
        if settings.compile:
            reference_model = torch.compile(reference_model, mode=settings.compile)
            model = torch.compile(model, mode=settings.compile)

        # Training
        opt = torch.optim.Adam(
            model.parameters(),
            lr=settings.lr,
            eps=settings.adam_eps,
        )
        assert settings.batch_size % settings.data_parallel == 0
        data_iter = data.iter(settings.batch_size // settings.data_parallel)
        logger = _Logger(
            settings,
            rank=rank,
            experiment=experiment,
            valid_data=valid_data,
            model=model,
        )
        for _ in range(settings.steps):
            batch = tokenizer.batch_encode_plus(
                next(data_iter)["text"],
                return_tensors="pt",
                padding="max_length",
                max_length=settings.sequence_length,
                truncation=True,
            )
            opt.zero_grad()
            loss = compute_kl_loss(model, reference_model, batch)
            loss.backward()
            opt.step()
            logger.log(loss.float(), batch.attention_mask.sum())

        # Returning
        for m in model.modules():
            if isinstance(m, fsdp.FSDPModule):
                m.unshard()
        return model


def _train_worker(
    world_size: int, rank: int, init_method: str, kwargs: dict[str, Any]
) -> Any:
    torch.distributed.init_process_group(
        world_size=world_size, rank=rank, init_method=init_method
    )
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        transformers.utils.logging.disable_progress_bar()
        datasets.utils.logging.disable_progress_bar()
        return train(**kwargs)
    finally:
        torch.distributed.destroy_process_group()


def run_train(
    model_id: str, settings: Training, experiment: core.Experiment | None
) -> nn.Module:
    kwargs = dict(
        world_size=settings.data_parallel,
        init_method="tcp://localhost:10051",
        kwargs=dict(model_id=model_id, settings=settings, experiment=experiment),
    )
    processes = [
        multiprocessing.Process(target=_train_worker, kwargs=dict(rank=n, **kwargs))
        for n in range(1, settings.data_parallel)
    ]
    for p in processes:
        p.start()
    try:
        return _train_worker(rank=0, **kwargs)
    finally:
        for p in processes:
            p.join()


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
