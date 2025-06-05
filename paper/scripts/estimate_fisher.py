# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

from pathlib import Path

import weight_formats.experiments as E

if __name__ == "__main__":
    E.fisher.Sweep("20250604-fisher").run(Path(f"out/20250604-fisher"))
