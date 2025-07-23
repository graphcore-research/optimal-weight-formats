# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import collections
import contextlib
import dataclasses
import datetime
import decimal
import getpass
import itertools as it
import multiprocessing
import os
import random
import re
import shelve
import string
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Iterable, Type

import boto3
import boto3.dynamodb.conditions as dbc
import botocore
import botocore.exceptions
import numpy as np
import torch
import tqdm
import transformers
from torch import Tensor, nn

# Sweeping

MODELS = [
    # Llama
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    # Phi
    "microsoft/phi-4",
    # Qwen
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-7B",
    # Gemma
    "google/gemma-3-1b-pt",
    "google/gemma-3-4b-pt",
    "google/gemma-3-12b-pt",
]

FIELD_MODELS = dataclasses.field(default_factory=lambda: MODELS.copy())
FIELD_DEVICE = dataclasses.field(
    default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def iter_dict_product(
    config: dict[str, Any], *axes: str, progress: bool = False
) -> Iterable[dict[str, Any]]:
    """Iterate through the product of certain axes in a config."""
    values_list = list(it.product(*(config[k] for k in axes)))
    for i, values in enumerate(values_list):
        if progress:
            print(
                f"# [{i + 1}/{len(values_list)}] {dict(zip(axes, values))}",
                file=sys.stderr,
            )
        yield {**config, **dict(zip(axes, values))}


@contextlib.contextmanager
def activation_checkpointing_enabled(
    model: transformers.PreTrainedModel,
) -> Iterable[transformers.PreTrainedModel]:
    """A context manager to enable activation checkpointing, without enabling dropout.

    Warning - care might be necessary, in case the .training flag on `LlamaModel` etc
    is used to enable dropout.
    """
    assert not model.is_gradient_checkpointing
    try:
        # Don't use .train(), as we don't want to enable dropout.
        # Just hope none of the PreTrainedModel classes don't set dropout themselves.
        for m in model.modules():
            if isinstance(m, transformers.modeling_utils.PreTrainedModel):
                assert not m.training
                m.training = True
        # Note: use_reentrant=False selects an implementation that is compatible with
        # FSDP (the reentrant implementation unshards the model in the recomp pass)
        model.gradient_checkpointing_enable(dict(use_reentrant=False))
        yield model
    finally:
        model.gradient_checkpointing_disable()
        for m in model.modules():
            if isinstance(m, transformers.modeling_utils.PreTrainedModel):
                m.training = False


# RequantisableModel


@dataclass
class RequantisableModel:
    """Wraps transformers.PreTrainedModel, storing original parameters on CPU,
    so that they can be restored when needed.
    """

    model: transformers.PreTrainedModel
    original_params: dict[str, nn.Parameter]

    @classmethod
    def load(
        cls, name: str, device: torch.device, dtype: torch.dtype
    ) -> "RequantisableModel":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, device_map=device, torch_dtype=dtype
        )
        original_params = {
            k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters()
        }
        return cls(model=model, original_params=original_params)

    @property
    def device(self) -> torch.device:
        (device,) = set(p.device for p in self.model.parameters())
        return device

    def reset(self) -> None:
        for name, p in self.model.named_parameters():
            p.data[...] = self.original_params[name].to(p.device)
            if hasattr(p, "_quantised"):
                del p._quantised

    def reset_parameter(self, name: str) -> None:
        p = dict(self.model.named_parameters())[name]
        p.data[...] = self.original_params[name].to(p.device)
        if hasattr(p, "_quantised"):
            del p._quantised


# Utility


class AttrDict(dict):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__ = self


def generate_id() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=10))


def _git_head() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _device_info() -> dict[str, Any]:
    try:
        return dict(
            device_name=torch.cuda.get_device_name(),
            device_count=torch.cuda.device_count(),
        )
    except Exception:
        return {}


def _dump_error(error: Exception) -> dict[str, Any]:
    return dict(
        type=type(error).__qualname__,
        message=str(error),
        trace=traceback.format_tb(error.__traceback__),
        **(dict(cause=_dump_error(error.__cause__)) if error.__cause__ else {}),
    )


def _get_username() -> str | None:
    if "ENDUSER" in os.environ:
        return os.environ["ENDUSER"]
    try:
        arn = boto3.client("sts").get_caller_identity()["Arn"]
        if m := re.search("user/(.+)", arn):
            return m.group(1)
    except botocore.exceptions.NoCredentialsError:
        pass
    return getpass.getuser()


@contextmanager
def cuda_memory_history(
    path: str | Path, max_entries: int = int(1e5)
) -> Iterable[None]:
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)


# Database API


class _DB:
    """Base interface for local or remote experiment storage."""

    def put(self, item: dict[str, Any]) -> None:
        raise NotImplementedError

    def delete(self, key: dict[str, Any]) -> None:
        raise NotImplementedError

    def get(self, key: dict[str, Any]) -> AttrDict:
        raise NotImplementedError

    def query(self, key: dict[str, Any], progress: bool) -> Iterable[AttrDict]:
        raise NotImplementedError

    def keys(self, key_name: str, progress: bool) -> Iterable[str]:
        raise NotImplementedError

    def __enter__(self) -> "_DB":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


class _Dynamo(_DB):
    def __init__(self, region: str, table: str):
        self._table = boto3.resource("dynamodb", region_name=region).Table(table)

    def put(self, item: dict[str, Any]) -> None:
        self._table.put_item(Item=self._to_db(item))

    def delete(self, key: dict[str, Any]) -> None:
        self._table.delete_item(Key=key)

    def get(self, key: dict[str, Any]) -> dict[str, Any]:
        response = self._table.get_item(Key=key)
        if "Item" not in response:
            raise KeyError(f"Entry {key} not found")
        return self._from_db(response["Item"])

    def query(self, key: dict[str, Any], progress: bool) -> Iterable[AttrDict]:
        ((key_name, key_value),) = list(key.items())  # note: restrictive
        for item in self._call_paginated(
            "query",
            KeyConditionExpression=dbc.Key(key_name).eq(key_value),
            _progress=progress,
        ):
            yield self._from_db(item)

    def keys(self, key_name: str, progress: bool) -> Iterable[str]:
        for x in self._call_paginated(
            "scan", ProjectionExpression=key_name, _progress=progress
        ):
            yield x[key_name]

    def _call_paginated(
        self, method: str, _progress: bool, **args: Any
    ) -> Iterable[Any]:
        start = {}
        with tqdm.tqdm(desc=method, disable=not _progress) as pbar:
            while True:
                response = getattr(self._table, method)(**args, **start)
                pbar.update(len(response["Items"]))
                yield from response["Items"]
                if "LastEvaluatedKey" not in response:
                    break
                start = dict(ExclusiveStartKey=response["LastEvaluatedKey"])

    @classmethod
    def _to_db(cls, value: Any, prefix: tuple[Any] = ()) -> Any:
        if isinstance(
            value, (str, int, decimal.Decimal, bool, type(None), bytes, bytearray)
        ):
            return value
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                raise TypeError(f"Bad value {'.'.join(map(str, prefix))}={value}")
            # Approximately match float32 precision
            return decimal.Decimal.from_float(value).normalize(decimal.Context(prec=8))
        if isinstance(value, torch.dtype):
            return str(value).replace("torch.", "")
        if isinstance(value, torch.device):
            return value.type
        if isinstance(value, (list, tuple)):
            return [cls._to_db(v, prefix + (i,)) for i, v in enumerate(value)]
        if isinstance(value, (np.ndarray, Tensor)):
            return cls._to_db(value.tolist(), prefix)
        if isinstance(value, set):
            return {cls._to_db(v, prefix + ("#",)) for v in value}
        if isinstance(value, dict):
            non_string_keys = [k for k in value if not isinstance(k, str)]
            if non_string_keys:
                raise TypeError(
                    f"Cannot convert non-string keys for the database, "
                    f"{'.'.join(map(str, prefix))}:{set(type(k) for k in non_string_keys)}"
                )
            return {k: cls._to_db(v, prefix + (k,)) for k, v in value.items()}
        raise TypeError(
            f"Unexpected type for database, {'.'.join(map(str, prefix))}:{type(value)}"
        )

    @classmethod
    def _from_db(cls, value: Any) -> Any:
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, list):
            return [cls._from_db(v) for v in value]
        if isinstance(value, set):
            return {cls._from_db(v) for v in value}
        if isinstance(value, dict):
            return AttrDict(**{k: cls._from_db(v) for k, v in value.items()})
        return value


class _LocalDB:
    def __init__(self, path: Path, key: tuple[str, ...]):
        if not path.parent.exists():
            path.parent.mkdir()
        self._shelf = shelve.open(path).__enter__()
        self.key = key

    def _name(self, key: dict[str, Any]) -> None:
        assert not any("/" in key[k] for k in self.key)
        return "/".join(key[k] for k in self.key)

    def put(self, item: dict[str, Any]) -> None:
        self._shelf[self._name(item)] = self._to_db(item)

    def delete(self, key: dict[str, Any]) -> None:
        del self._shelf[self._name(key)]

    def get(self, key: dict[str, Any]) -> AttrDict:
        return self._from_db(self._shelf[self._name(key)])

    def query(self, key: dict[str, Any], progress: bool) -> Iterable[AttrDict]:
        for name in tqdm.tqdm(self._shelf, "scan", disable=not progress):
            keylist = name.split("/")
            if all(keylist[self.key.index(k)] == v for k, v in key.items()):
                yield self._from_db(self._shelf[name])

    def keys(self, key_name: str, progress: bool) -> Iterable[str]:
        for name in tqdm.tqdm(self._shelf, "scan", disable=not progress):
            yield name.split("/")[self.key.index(key_name)]

    def __enter__(self) -> "_DB":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._shelf.__exit__(exc_type, exc_value, tb)

    @classmethod
    def _to_db(cls, value: Any) -> Any:
        if isinstance(value, (np.ndarray, Tensor)):
            return cls._to_db(value.tolist())
        if isinstance(value, (tuple, list, set)):
            return type(value)(cls._to_db(v) for v in value)
        if isinstance(value, dict):
            return {k: cls._to_db(v) for k, v in value.items()}
        return value

    @classmethod
    def _from_db(cls, value: Any) -> Any:
        if isinstance(value, (tuple, list, set)):
            return type(value)(cls._from_db(v) for v in value)
        if isinstance(value, dict):
            return AttrDict(**{k: cls._from_db(v) for k, v in value.items()})
        return value


# Database

DDB_REGION, DDB_TABLE = ("eu-central-1", "2025-04-block-number-formats")


def _has_dynamo_access() -> bool:
    try:
        boto3.client("dynamodb", region_name=DDB_REGION).describe_table(
            TableName=DDB_TABLE
        )
        return True
    except botocore.exceptions.NoCredentialsError:
        return False
    except botocore.exceptions.ClientError:
        return False


def _db() -> _DB:
    if _has_dynamo_access():
        return _Dynamo(DDB_REGION, DDB_TABLE)
    return _LocalDB(
        Path(__file__).parent.parent.parent / ".results" / "local",
        ("experiment", "run_id"),
    )


class Experiment:
    """A context manager for a running experiment."""

    def __init__(self, config: dict[str, Any]):
        self._db = _db().__enter__()
        config = config.copy()
        self.experiment = config.pop("experiment")
        self.run_id = generate_id()
        self._record = dict(
            experiment=self.experiment,
            run_id=self.run_id,
            config=config,
            meta=dict(
                status="running",
                time=datetime.datetime.now().isoformat(),
                commit=_git_head(),
                user=_get_username(),
                hostname=os.environ.get("HOSTNAME", ""),
                **_device_info(),
            ),
            summary={},
            error=None,
        )
        self._t0 = time.time()
        self.sync(unrecoverable=True)

    def sync(self, unrecoverable: bool = False) -> None:
        try:
            self._db.put(self._record)
        except Exception as error:
            if unrecoverable:
                print(
                    f'ERROR: Failed to sync experiment "{self.experiment}/{self.run_id}"'
                    f" with {error!r}",
                    file=sys.stderr,
                    flush=True,
                )
            raise

    def summary(self, **args: Any) -> None:
        old_summary = self._record["summary"]
        self._record["summary"] = {**self._record["summary"], **args}
        try:
            self.sync()
        except Exception:
            # Restore the original summary so that sync isn't broken, and we can still log the error
            self._record["summary"] = old_summary
            raise

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if exc_value:
                self._record.update(error=_dump_error(exc_value))
                self._record["meta"].update(status="failed")
            else:
                self._record["meta"].update(status="finished")
            self._record["meta"].update(duration=time.time() - self._t0)
            self.sync(unrecoverable=True)
        finally:
            self._db.__exit__(exc_type, exc_value, tb)


def _with_id(run: AttrDict) -> dict[str, Any]:
    run["id"] = f"{run.experiment}/{run.run_id}"
    return run


def run(id: str) -> dict[str, Any]:
    """Fetch a specific run by ID."""
    experiment, run_id = id.split("/")
    with _db() as db:
        return _with_id(db.get(dict(experiment=experiment, run_id=run_id)))


def runs(experiment: str, progress: bool = False) -> list[dict[str, Any]]:
    """Fetch all runs for a given experiment."""
    with _db() as db:
        return sorted(
            map(_with_id, db.query(dict(experiment=experiment), progress=progress)),
            key=lambda x: x.meta.time,
        )


def update_run(run: dict[str, Any]) -> None:
    """Update the given run to reflect local changes (be careful!)"""
    with _db() as db:
        db.put({k: v for k, v in run.items() if k != "id"})


def delete_run(id: str) -> None:
    """Remove the given run."""
    experiment, run_id = id.split("/")
    with _db() as db:
        db.delete(dict(experiment=experiment, run_id=run_id))


def experiments(progress: bool = False) -> list[str]:
    """A list of all experiments in the database."""
    with _db() as db:
        counts = collections.Counter(db.keys("experiment", progress=progress))
        return [dict(experiment=k, runs=counts[k]) for k in sorted(counts)]


# Sweeping

_SWEEP_RUNNER: Callable[..., None] = None


def _sweep_init(queue: multiprocessing.Queue, runner: Type[Any]) -> None:
    # CUDA_VISIBLE_DEVICES seems better than using torch.device at
    # reusing the torch.compile cache between GPUs
    device = queue.get_nowait()
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.set_num_threads(16)  # avoid CPU contention when sweeping
    global _SWEEP_RUNNER
    _SWEEP_RUNNER = runner()


def _try_run(runner: Callable[..., None], args: Any, kwargs: dict[str, Any]) -> None:
    try:
        runner(*args, **kwargs)
    except Exception:
        print(f"### Sweep run error for args={args} kwargs={kwargs}", file=sys.stderr)
        traceback.print_exc()


def _sweep_run(args: Any, kwargs: dict[str, Any]) -> None:
    _try_run(_SWEEP_RUNNER, args, kwargs)


def run_sweep(
    runner: Type[Any],
    sweep_args: Iterable[tuple[Any, ...]],
    processes: int | None = None,
    kwargs: dict[str, Any] = {},
) -> None:
    if processes is None:
        processes = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if processes == 1:
        # Run directly in the host process (easier to debug)
        runner = runner()
        for args in sweep_args:
            _try_run(runner, args, kwargs)
    else:
        # Start subprocesses that "own" device IDs then use a pool to divide work
        queue = multiprocessing.Manager().Queue()
        for idx in range(processes):
            queue.put(
                (idx % torch.cuda.device_count()) if torch.cuda.is_available() else None
            )
        pool = multiprocessing.get_context("spawn").Pool(
            processes, _sweep_init, (queue, runner)
        )
        for args in sweep_args:
            pool.apply_async(_sweep_run, (args, kwargs))
        pool.close()
        pool.join()
