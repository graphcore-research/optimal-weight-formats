# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import collections
import concurrent.futures
import contextlib
import copy
import dataclasses
import gc
import getpass
import itertools as it
import math
import os
import queue
import sys
import time
import unittest.mock
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypeAlias

import datasets
import safetensors.torch
import torch
import torch.multiprocessing as multiprocessing
import transformers
from torch import Tensor, nn
from torch.distributed import fsdp

from .. import fit as F
from .. import model_quantisation as M
from .. import quantisation as Q
from .. import quantisation_training as T
from . import core, fisher, token_prediction
from .internal import submit

CODE_CHANGES = dict(
    compile_disable=["Weight.forward"],
    fix_fsdp_embedding_sharing=True,
)

# Settings


@dataclass
class Baseline:
    type: str = "baseline"


@dataclass
class PerturbNoise:
    ratio: float
    type: str = "perturb_noise"


@dataclass
class PerturbQuantise:
    fmt: F.Scaled | Q.TensorFormat
    bit_allocation: Literal["fixed", "variable"] = "fixed"
    error_weight: Literal["fisher"] | None = None
    type: str = "perturb_quantise"


@dataclass
class QAT:
    fmt: F.Scaled | Q.TensorFormat
    scaling_mode: T.ScalingMode
    clip_gradient: bool
    bit_allocation: Literal["fixed", "variable"] = "fixed"
    error_weight: Literal["fisher"] | None = None
    type: str = "qat"


Test: TypeAlias = Baseline | PerturbNoise | PerturbQuantise | QAT


@dataclass
class TrainingSettings:
    steps: int
    batch_size: int
    log_interval: int
    valid_sequences: int = 132
    sequence_length: int = 1024
    dataset: str = "DKYoon/SlimPajama-6B"


@dataclass
class LRModifiers:
    weight: float = 1.0
    scale: float = 1.0
    centroids: float = 0.0  # default to skip (slow) trainable-centroids
    other: float = 1.0


LRSchedule: TypeAlias = Literal["constant", "cosine", "linear"]


@dataclass
class OptimiserSettings:
    lr: float
    lr_modifiers: LRModifiers = dataclasses.field(default_factory=LRModifiers)
    lr_schedule: LRSchedule = "cosine"
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-5
    weight_decay: float = 0.0


@dataclass
class ExecutionSettings:
    data_parallel: int = torch.cuda.device_count()
    compile: str | None = "default"
    memory_profile: str | None = None
    checkpoint: str | None = None
    params_dtype: torch.dtype = torch.float32
    compute_dtype: torch.dtype = torch.bfloat16
    reference_dtype: torch.dtype = torch.bfloat16


@dataclass
class Task:
    name: str
    limit: int | None = None


TASKS = tuple(
    [
        # Selected cloze or multiple-choice, based on a baseline sweep
        # named "20250611-downstream-baselines"
        Task(name=name, limit=d.get("limit"))
        for name, d in [
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
)


@dataclass
class Run:
    experiment: str
    model: str
    test: Test
    train: TrainingSettings
    opt: OptimiserSettings
    exe: ExecutionSettings
    tasks: tuple[Task, ...] = TASKS
    code_changes: dict[str, Any] = dataclasses.field(default_factory=CODE_CHANGES.copy)
    tag: str = ""
    type: str = "qat"

    def to_config(self) -> dict[str, Any]:
        config = dataclasses.asdict(self)
        if hasattr(self.test, "fmt"):
            config["test"]["fmt_str"] = str(self.test.fmt)
        return config


# Downstream Tasks


def evaluate(model: transformers.PreTrainedModel, task: Task) -> dict[str, Any]:
    import oe_eval.models.eleuther_huggingface
    import oe_eval.run_eval
    import oe_eval.tasks.oe_eval_tasks

    t0 = time.time()
    oe_task = oe_eval.tasks.oe_eval_tasks.TASK_REGISTRY[task.name](task_name=task.name)
    oe_task.download()
    oe_task.build_all_requests(limit=task.limit)
    # Hack around `kwargs["parallelize"] = True`
    with unittest.mock.patch("torch.cuda.device_count", new=lambda: 1):
        oe_model = oe_eval.models.eleuther_huggingface.HFLM_Verbose(
            model, parallelize=False
        )
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


# Quantisation-Aware Training


def _safe_torch_dot(input: Tensor, other: Tensor, chunk_size=2**31 - 1) -> Tensor:
    """Computes the dot product of two 1D tensors in chunks to avoid size limits."""
    out = torch.tensor(0.0, device=input.device, dtype=input.dtype)
    for i in range(0, len(input), chunk_size):
        out += torch.dot(input[i : i + chunk_size], other[i : i + chunk_size])
    return out


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
        return _safe_torch_dot(target_p.flatten(), input_logp.flatten()).neg()

    @staticmethod
    def backward(ctx, grad_output):
        input_logp, target_p, mask = ctx.saved_tensors
        grad_input_logits = (
            input_logp.exp_().sub_(target_p).mul_(mask).mul_(grad_output)
        )
        del input_logp
        return grad_input_logits, None, None


def _compute_kl_loss(
    model: nn.Module,
    reference_model: nn.Module,
    batch: dict[str, Tensor],
    mask: Tensor | None = None,
) -> Tensor:
    """Computes the KL divergence between the output of two models, with care for memory."""
    if mask is None:
        mask = batch["attention_mask"]
    else:
        assert mask.shape == batch["attention_mask"].shape
        mask = mask & batch["attention_mask"]

    with torch.no_grad():
        reference_logp = torch.log_softmax(
            reference_model(**batch, use_cache=False).logits, -1
        )
        reference_p = reference_logp.exp().mul_(mask.unsqueeze(-1))
        # Calculate reference entropy here, so we can free up `reference_logp`
        reference_ent = _safe_torch_dot(
            reference_p.flatten(), reference_logp.flatten()
        ).neg()
        del reference_logp

    xent = XEnt_Destructive.apply(
        model(**batch, use_cache=False).logits,
        reference_p,
        mask.unsqueeze(-1),
    )
    return xent - reference_ent


def _compile_transformer_layers(model: transformers.PreTrainedModel, mode: str) -> None:
    for layer_id, layer in model.model.layers.named_children():
        layer = torch.compile(layer, mode=mode)
        model.model.layers.register_module(layer_id, layer)


def _fully_shard_model(model: transformers.PreTrainedModel, **args: Any) -> None:
    for layer in model.model.layers:
        fsdp.fully_shard(layer, **args)
    # Important to do this at top-level for sake of parameter sharing
    # note: this breaks unshard()ed OLMES evaluation, so we copy to a non-FSDP model
    fsdp.fully_shard(model, **args)


def _is_master() -> bool:
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _save_model(model: nn.Module, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use named_parameters() not state_dict() or save_model() to avoid resharding the model
    safetensors.torch.save_file(dict(model.named_parameters()), path)


def _deepcopy_with_dummy_params(
    model: nn.Module, dtype: torch.dtype | None = None
) -> nn.Module:
    """Deep copy a module, but replace all parameters with low-memory aliases."""
    model_copy = copy.deepcopy(model)
    param_map = {}

    def _visit(m: nn.Module) -> None:
        for child in m.children():
            _visit(child)
        for k, v in m.named_parameters(recurse=False):
            new_v = param_map.get(v)
            if new_v is None:
                new_v = param_map[v] = nn.Parameter(
                    torch.tensor(
                        torch.nan, dtype=dtype or v.dtype, device=v.device
                    ).broadcast_to(v.shape)
                )
            setattr(m, k, new_v)

    _visit(model_copy)
    return model_copy


def _replace_params(model: nn.Module, params: dict[str, Tensor]) -> None:
    """Replace parameters of a module that may have low-memory aliases from `deepcopy_with_dummy_params`."""
    params = params.copy()  # as we modify `params`
    param_map = {}

    def _visit(m: nn.Module, prefix: tuple[str, ...]) -> None:
        for name, child in m.named_children():
            _visit(child, prefix + (name,))
        for k, v in m.named_parameters(recurse=False):
            new_v = param_map.get(v)
            if new_v is None:
                new_v = param_map[v] = nn.Parameter(
                    params.pop(".".join(prefix + (k,))).data.clone()
                )
                assert (
                    new_v.shape == v.shape
                ), f"failed param vs model shape {tuple(new_v.shape)} == {tuple(v.shape)}"
            setattr(m, k, new_v)

    _visit(model, ())
    assert not params, f"left-over params: {list(params)}"


def _unshard_to_new_model(src: nn.Module, dest: nn.Module) -> None:
    """Unshard `src` then copy parameters to `dest`, of the same structure.

    Since we can't cleanly convert a fully_shard()ed model to a regular model, we
    copy the parameters to a fresh model.
    """

    def _strip_orig_mod(name: str) -> str:
        return ".".join([p for p in name.split(".") if p != "_orig_mod"])

    for m in src.modules():
        if isinstance(m, fsdp.FSDPModule):
            m.unshard()
    _replace_params(dest, {_strip_orig_mod(n): p for n, p in src.named_parameters()})


def _bits_per_param(model: nn.Module) -> float:
    # We assume torch.bfloat16, even if not using it, as this was "original precision"
    try:
        for m in model.modules():
            if isinstance(m, fsdp.FSDPModule):
                m.unshard()
        return T.count_bits(model, torch.bfloat16) / T.count_parameters(model)
    finally:
        for m in model.modules():
            if isinstance(m, fsdp.FSDPModule):
                m.reshard()


def _lr_schedule_fn(name: LRSchedule, steps: int) -> Callable[[int], float]:
    if name == "constant":
        return lambda _: 1.0
    if name == "linear":
        return lambda n: 1 - n / steps
    if name == "cosine":
        return lambda n: 0.5 * (math.cos(math.pi * n / steps) + 1)
    raise ValueError(f"Unexpected schedule {name!r}")


def _process_model(model: transformers.PreTrainedModel, test: Test) -> dict[str, Any]:
    if isinstance(test, Baseline):
        return {}

    if isinstance(test, PerturbNoise):
        for p in model.parameters():
            if p.ndim == 2:
                scale = p.pow(2).mean(dtype=torch.float32).sqrt().mul(test.ratio)
                p.data += torch.randn_like(p).mul(scale)
        return {}

    if isinstance(test, PerturbQuantise):
        error_weight = (
            None
            if test.error_weight is None
            else fisher.fetch_fisher(model.config._name_or_path, model.device)
        )
        if test.bit_allocation == "fixed":
            log = M.quantise_2d_fixed(model, test.fmt, error_weight=error_weight)
            return dict(init_bits_per_param=log["bits_per_param"])
        elif test.bit_allocation == "variable":
            fisher_sum = fisher.fetch_fisher_sum(model.config._name_or_path)
            log = M.quantise_2d_variable(
                model,
                test.fmt,
                fisher_sum=fisher_sum,
                min_element_bits=None,
                error_weight=error_weight,
            )
            return dict(init_bits_per_param=log["bits_per_param"])
        else:
            raise ValueError(f"Unexpected bit_allocation={test.bit_allocation!r}")

    if isinstance(test, QAT):
        if test.bit_allocation == "variable":
            raise NotImplementedError("bit_allocation=='variable' is not implemented")
        T.convert(
            model,
            fmt_spec=test.fmt,
            scaling_mode=test.scaling_mode,
            clip_gradient=test.clip_gradient,
            error_weight=(
                None
                if test.error_weight is None
                else fisher.fetch_fisher(model.config._name_or_path, model.device)
            ),
        )
        return {}

    raise ValueError(f"Unexpected test {type(test)}")


def _iter_batches(
    dataset: str, model: str, batch_size: int, sequence_length: int, data_parallel: int
) -> Iterable[dict[str, Tensor]]:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    tokenizer.pad_token = tokenizer.eos_token
    data = datasets.load_dataset(dataset)["train"]
    if data_parallel >= 2:
        assert data_parallel == torch.distributed.get_world_size()
        data = data.shard(
            torch.distributed.get_world_size(), torch.distributed.get_rank()
        )
    assert batch_size % data_parallel == 0
    for batch in data.iter(batch_size // data_parallel):
        yield tokenizer.batch_encode_plus(
            batch["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=sequence_length,
            truncation=True,
        )


def _create_optimiser(
    model: transformers.PreTrainedModel, opt: OptimiserSettings, steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    # If we run anything before creating the optimiser, we may have unsharded the model.
    # We must reshard it here before creating the optimiser, or we won't train anything.
    for m in model.modules():
        if isinstance(m, fsdp.FSDPModule):
            m.reshard()

    param_groups = []
    if any(isinstance(m, T.Weight) for m in model.modules()):
        for category in ["weight", "scale", "centroids", "other"]:
            params = T.get_named_parameters(model, category)
            group_lr = opt.lr * getattr(opt.lr_modifiers, category)
            if group_lr == 0:
                for _, p in params:
                    p.requires_grad_(False)
            elif params:
                param_groups.append(dict(params=params, lr=torch.tensor(group_lr)))
    else:
        param_groups.append(
            dict(params=model.named_parameters(), lr=torch.tensor(opt.lr))
        )

    optimiser = torch.optim.AdamW(
        param_groups,
        lr=0,
        betas=opt.betas,
        eps=opt.eps,
        weight_decay=opt.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, _lr_schedule_fn(opt.lr_schedule, steps)
    )
    return optimiser, schedule


class _Logger:
    def __init__(
        self,
        log_interval: int,
        valid_sequences: int,
        data_parallel: int,
        experiment: core.Experiment | None,
        reference_model: nn.Module,
        model: nn.Module,
    ):
        self.log_interval = log_interval
        self.valid_sequences = valid_sequences
        self.data_parallel = data_parallel
        self.experiment = experiment
        self.model = model
        if valid_sequences:
            self.valid_data = token_prediction.Dataset.load(
                reference_model,
                sequence_length=4096,
                batch_size=1,
                kl_topk=128,
                sequence_limit=valid_sequences,
            )

        # State
        self._step = 0
        self._logged_step = 0
        self._t0 = time.time()
        self._total_loss = torch.tensor(0.0)
        self._total_count = torch.tensor(0)
        self._log = core.AttrDict(loss=[], duration=[], bits_per_param=[])
        self._validate_and_log()

    def validate(self) -> dict[str, float]:
        if self.valid_sequences:
            return {
                f"valid_{k}": v.mean().item()
                for k, v in self.valid_data.evaluate(self.model).items()
            }
        return {}

    def _validate_and_log(self) -> None:
        # Compute statistics
        if self.data_parallel > 1:
            torch.distributed.all_reduce(self._total_loss)
            torch.distributed.all_reduce(self._total_count)
        t1 = time.time()  # don't count validation time
        if self._step == self._logged_step:
            self._log.loss.append(None)
            self._log.duration.append(None)
        else:
            self._log.loss.append(self._total_loss.div(self._total_count).item())
            self._log.duration.append(
                (t1 - self._t0) / (self._step - self._logged_step)
            )
        self._log.bits_per_param.append(_bits_per_param(self.model))
        for k, v in self.validate().items():
            self._log.setdefault(k, []).append(v)

        # Write logs
        if _is_master():
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
        if self._step % self.log_interval == 0:
            self._validate_and_log()


def _run_worker(run: Run) -> None:
    with contextlib.ExitStack() as exit:
        if _is_master():
            print(run.to_config(), file=sys.stderr)
        experiment = (
            exit.enter_context(core.Experiment(run.to_config()))
            if _is_master()
            else None
        )
        if _is_master() and run.exe.memory_profile:
            exit.enter_context(core.cuda_memory_history(Path(run.exe.memory_profile)))

        # Loading
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            run.model, torch_dtype=run.exe.reference_dtype
        )
        model = copy.deepcopy(reference_model)
        process_log = _process_model(model, run.test)
        if experiment:
            experiment.summary(**process_log)
        eval_model = _deepcopy_with_dummy_params(model, dtype=run.exe.compute_dtype)

        # Preparation
        model.to(run.exe.params_dtype)
        exit.enter_context(core.activation_checkpointing_enabled(model))
        if torch.distributed.is_initialized():
            _fully_shard_model(reference_model)
            _fully_shard_model(
                model,
                mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=run.exe.compute_dtype),
            )
        else:
            assert run.exe.data_parallel == 1
            assert run.exe.compute_dtype == run.exe.params_dtype
        if run.exe.compile:
            torch._dynamo.config.cache_size_limit = 64
            _compile_transformer_layers(reference_model, run.exe.compile)
            _compile_transformer_layers(model, run.exe.compile)

        # Training
        train_t0 = time.time()
        logger = _Logger(
            log_interval=run.train.log_interval,
            valid_sequences=run.train.valid_sequences,
            data_parallel=run.exe.data_parallel,
            experiment=experiment,
            reference_model=reference_model,
            model=model,
        )
        if run.train.steps:
            opt, schedule = _create_optimiser(model, run.opt, run.train.steps)
            batches = _iter_batches(
                run.train.dataset,
                run.model,
                batch_size=run.train.batch_size,
                sequence_length=run.train.sequence_length,
                data_parallel=run.exe.data_parallel,
            )
            for batch in it.islice(batches, run.train.steps):
                opt.zero_grad()
                loss = _compute_kl_loss(model, reference_model, batch)
                loss.backward()
                opt.step()
                schedule.step()
                logger.log(loss.float(), batch["attention_mask"].sum())
            del opt
        if model.config.tie_word_embeddings and isinstance(run.test, QAT):
            assert torch.equal(
                model.lm_head.weight.master, model.model.embed_tokens.weight.master
            ), "embeddings were un-shared"
        duration_train = time.time() - train_t0

        # Evaluation

        # Copy into `eval_model` for evaluation, as it's risky to use an FSDP
        # model, which cannot be un-fully-sharded
        del reference_model
        _unshard_to_new_model(model, eval_model)
        model = logger.model = eval_model
        # Required to clean up `reference_model` and old `model`
        gc.collect()
        torch.cuda.empty_cache()
        if not _is_master():
            return

        if run.exe.checkpoint:
            _save_model(model, run.exe.checkpoint)

        experiment.summary(
            **logger.validate(),
            bits_per_param=_bits_per_param(model),
            params=T.count_parameters(model),
            duration_train=duration_train,
        )
        # separate, in case it fails
        experiment.summary(
            downstream={task.name: evaluate(model, task) for task in run.tasks}
        )


# Distributed training


@contextlib.contextmanager
def _init_distributed(**args: Any) -> Iterable[None]:
    torch.distributed.init_process_group(**args)
    try:
        device = torch.device("cuda", torch.distributed.get_rank())
        torch.cuda.set_device(device)
        torch.set_default_device(device)
        transformers.utils.logging.disable_progress_bar()
        datasets.utils.logging.disable_progress_bar()
        warnings.filterwarnings(
            "ignore",
            message="None of the inputs have requires_grad=True. Gradients will be None",
            category=UserWarning,
        )
        yield
    finally:
        torch.distributed.destroy_process_group()


def _init_and_run_worker(
    rank: int, run: Run, devices: list[int] | None, init_method: str
) -> Any:
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    with _init_distributed(
        rank=rank, world_size=run.exe.data_parallel, init_method=init_method
    ):
        return _run_worker(run)


def run(run: Run, port: int | None = None, devices: list[int] | None = None) -> None:
    if port is None:
        port = torch.randint(10000, 20000, ()).item()
    multiprocessing.spawn(
        _init_and_run_worker,
        args=(run, devices, f"tcp://localhost:{port}"),
        nprocs=run.exe.data_parallel,
    )


def run_sweep(runs: list[Run], dry_run: bool = False) -> None:
    devices = torch.cuda.device_count()
    (data_parallel,) = set(run.exe.data_parallel for run in runs)
    assert (
        devices % data_parallel == 0
    ), f"bad sweep: device count {devices} must be a multiple of run.exe.data_parallel {data_parallel}"
    n_workers = devices // data_parallel

    if dry_run:
        print("### run_sweep(dry_run=True)", file=sys.stderr)
        for run_ in runs:
            print(f"   {run_}", file=sys.stderr)
        return

    # The worker_queue is a list of available worker IDs
    worker_queue = queue.SimpleQueue()
    for i in range(n_workers):
        worker_queue.put(i)

    def _sweep_runner(run_: Run) -> None:
        worker_id = worker_queue.get()
        try:
            devices = list(
                range(data_parallel * worker_id, data_parallel * (worker_id + 1))
            )
            run(run_, port=15300 + worker_id, devices=devices)
        finally:
            worker_queue.put(worker_id)

    with concurrent.futures.ThreadPoolExecutor(n_workers) as pool:
        list(pool.map(_sweep_runner, runs))


def submit_sweep(
    runs: list[Run],
    devices: int = 8,
    priority: submit.Priority = "medium",
    dry_run: Literal["sweep", "run"] | None = None,
    max_jobs: int | None = None,
) -> None:
    (name,) = set(run.experiment for run in runs)

    # Only runs with the same data_parallel setting can run_sweep() together
    jobs = []
    run_groups = collections.defaultdict(list)
    for run in runs:
        run_groups[run.exe.data_parallel].append(run)
        assert devices % run.exe.data_parallel == 0, (
            f"bad sweep: device count {devices} must be a multiple of"
            f" run.exe.data_parallel {data_parallel}"
        )
    assert (
        max_jobs is None or len(run_groups) == 1
    ), "Cannot use max_jobs with multiple data_parallel settings"

    for data_parallel in sorted(run_groups):
        runs = run_groups[data_parallel]
        if max_jobs is None:
            run_batch_size = devices // data_parallel
        else:
            run_batch_size = math.ceil(len(runs) / max_jobs)
        for run_batch in it.batched(runs, run_batch_size):
            jobs.append(
                submit.Job(run_sweep, (run_batch,), dict(dry_run=dry_run == "run"))
            )

    hf_cache = (
        Path("/data")
        / os.environ.get("ENDUSER", getpass.getuser())
        / "cache"
        / "huggingface"
    )
    sub = submit.Submission(
        name=name,
        devices=devices,
        jobs=jobs,
        env=dict(
            AWS_ACCESS_KEY_ID=os.environ["SWEEP_AWS_ACCESS_KEY_ID"],
            AWS_SECRET_ACCESS_KEY=os.environ["SWEEP_AWS_SECRET_ACCESS_KEY"],
            HF_TOKEN=os.environ["SWEEP_HF_TOKEN"],
            HF_HUB_CACHE=str(hf_cache / "hub"),
            HF_DATASETS_CACHE=str(hf_cache / "datasets"),
        ),
        priority=priority,
    )
    submit.run(sub, dry_run=dry_run == "sweep")
