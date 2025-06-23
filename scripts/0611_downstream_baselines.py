# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments as E
import weight_formats.experiments.downstream as ED

if __name__ == "__main__":
    ED.run_sweep(
        [
            ED.Run("20250611-downstream-baselines", model, tasks=ED.TASKS, fmt=None)
            for model in [m for m in E.core.MODELS if m != "meta-llama/Llama-3.2-1B"]
        ]
    )
