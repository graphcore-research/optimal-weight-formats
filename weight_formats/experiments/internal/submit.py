# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import copy
import dataclasses
import getpass
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


def _sh(cmd: list[str], input: bytes | None = None) -> None:
    try:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True, input=input)
    except subprocess.CalledProcessError:
        print(f"ERROR RUNNING $ {cmd}", file=sys.stderr)
        raise


@dataclass
class Job:
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def run(self) -> None:
        self.fn(*self.args, **self.kwargs)

    def save(self) -> bytes:
        return pickle.dumps(self)


def _user_default() -> str:
    if "ENDUSER" in os.environ:
        return os.environ["ENDUSER"]
    return getpass.getuser()


def _project_name_default() -> str:
    return (
        subprocess.check_output(["git", "remote", "get-url", "origin"])
        .decode()
        .strip()
        .split("/")[-1]
        .replace(".git", "")
    )


def _commit_default() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()


@dataclass
class Submission:
    name: str
    devices: int
    jobs: list[Job]
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    # Auto
    user: str = dataclasses.field(default_factory=_user_default)
    project_name: str = dataclasses.field(default_factory=_project_name_default)
    commit: str = dataclasses.field(default_factory=_commit_default)
    # Templates
    shared_folder: str = "/data/{user}"
    shared_repo: str = "{shared_folder}/{project_name}/repo.git"
    # note: the last part of `shared_config` must be a unique name to avoid job name clash
    shared_config: str = "{shared_folder}/{project_name}/config/{name}-{counter}"


def _expand_templates(sub: Submission) -> Submission:
    sub = copy.copy(sub)
    for k in ["shared_folder", "shared_repo"]:
        args = {k: v for k, v in sub.__dict__.items() if isinstance(v, str)}
        s = getattr(sub, k).format(**args)
        setattr(sub, k, s)
    args = {k: v for k, v in sub.__dict__.items() if isinstance(v, str)}
    sub.shared_config = next(
        c
        for c in (sub.shared_config.format(**args, counter=n) for n in range(1000))
        if not Path(c).exists()
    )
    return sub


def _sync_repo(repo: str, commit: str) -> None:
    if not Path(repo).exists():
        _sh(["git", "init", "--bare", repo])
        _sh(["git", "remote", "add", "submit", repo])
        _sh(["git", "-C", repo, "config", "gc.auto", "0"])
    submit_remote = (
        subprocess.check_output(["git", "remote", "get-url", "submit"]).decode().strip()
    )
    if submit_remote != repo:
        raise ValueError(
            f"Git remote 'submit' mismatch: expected {repo}, got: {submit_remote}"
        )
    _sh(["git", "push", "-f", "submit", f"{commit}:refs/heads/main"])


def _tempate_replace(src: Path, dest: Path, replacements: dict[str, Any]) -> None:
    script = src.read_text()
    for k, v in replacements.items():
        if k not in script:
            raise ValueError(f"Couldn't find {k} in template {src}")
        script = script.replace(k, repr(v))

    if not_replaced := re.findall(r"__TEMPLATE_.+__", script):
        raise ValueError(f"Un-matched template placeholders: {not_replaced} in {src}")

    dest.write_text(script)


def _generate_scripts(sub: Submission) -> Iterable[Path]:
    cpus_per_gpu = 20
    memory_gib_per_cpu = 4

    config_path = Path(sub.shared_config)
    yaml_path = config_path / "yaml"
    config_path.mkdir(parents=True)
    yaml_path.mkdir(parents=True)
    for n, job in enumerate(sub.jobs):
        job_id = f"{n:>03d}"
        job_py = config_path / f"{job_id}.py"
        job_yaml = yaml_path / f"{job_id}.yaml"
        _tempate_replace(
            Path(__file__).parent / "_template_worker.py",
            job_py,
            dict(
                __TEMPLATE_LOCAL_PATH__=sub.project_name,
                __TEMPLATE_REPO__=str(sub.shared_repo),
                __TEMPLATE_COMMIT__=sub.commit,
                __TEMPLATE_JOB__=job.save(),
                __TEMPLATE_RUNNER__=re.sub(r"\.submit$", "._run", __name__),
            ),
        )
        _tempate_replace(
            Path(__file__).parent / "_template_job.yaml",
            job_yaml,
            dict(
                __TEMPLATE_NAME__=f"{sub.user}-{config_path.name}-{job_id}",
                __TEMPLATE_USER__=sub.user,
                __TEMPLATE_COMMAND__=["python", str(job_py.absolute())],
                __TEMPLATE_ENV__=[dict(name=k, value=v) for k, v in sub.env.items()],
                __TEMPLATE_GPUS__=sub.devices,
                __TEMPLATE_CPUS__=sub.devices * cpus_per_gpu,
                __TEMPLATE_MEMORY__=f"{sub.devices * cpus_per_gpu * memory_gib_per_cpu}Gi",
            ),
        )
        yield job_yaml


def run(sub: Submission, dry_run: bool = False) -> None:
    sub = _expand_templates(sub)
    _sync_repo(sub.shared_repo, sub.commit)
    yamls = list(_generate_scripts(sub))
    try:
        for yaml in yamls:
            if dry_run:
                print(f"--- {yaml} ---\n{yaml.read_text()}", file=sys.stderr)
            else:
                _sh(["kubectl", "apply", "-f", str(yaml)])
    finally:
        for yaml in yamls:
            Path(yaml).unlink()
