# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments.token_prediction as ET

if __name__ == "__main__":
    ET.run_sweep(
        [
            ET.Run("dev", ET.PerturbEachParam(scale), model)
            for model in ["meta-llama/Llama-3.2-1B"]
            for scale in [1 / 4, 1 / 2, 1]
        ]
    )
