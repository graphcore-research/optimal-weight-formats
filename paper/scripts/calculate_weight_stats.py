# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

import weight_formats.experiments as E

if __name__ == "__main__":
    E.weight_stats.Sweep("20250423-weight-stats").run()
