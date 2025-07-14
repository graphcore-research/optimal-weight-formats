# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import os
import subprocess
import sys
from pathlib import Path

LOCAL_PATH: str = __TEMPLATE_LOCAL_PATH__  # type: ignore
REPO: str = __TEMPLATE_REPO__  # type: ignore
COMMIT: str = __TEMPLATE_COMMIT__  # type: ignore
RUNNER: str = __TEMPLATE_RUNNER__  # type: ignore
JOB: bytes = __TEMPLATE_JOB__  # type: ignore


def sh(cmd: list[str], input: bytes | None = None) -> None:
    try:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True, input=input)
    except subprocess.CalledProcessError:
        print(f"ERROR RUNNING $ {cmd}", file=sys.stderr)
        raise


if __name__ == "__main__":
    # Setup
    if not Path(LOCAL_PATH).exists():
        sh(["git", "clone", REPO, LOCAL_PATH])
    os.chdir(LOCAL_PATH)
    sh(["git", "fetch", "origin"])
    sh(["git", "checkout", COMMIT])
    if not Path(".venv").exists():
        sh(["python3", "-m", "venv", ".venv"])
        sh([".venv/bin/pip", "install", "-r", "requirements.txt"])

    # Run
    sh([".venv/bin/python", "-m", RUNNER], input=JOB)
