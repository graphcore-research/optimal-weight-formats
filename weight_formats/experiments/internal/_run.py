# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import pickle
import sys

from . import submit

if __name__ == "__main__":
    job: submit.Job = pickle.loads(sys.stdin.buffer.read())
    job.run()
