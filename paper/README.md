# Paper experiments - notes

This page describes the steps required to reproduce results from the Optimal Formats for Weight Quantisation paper.

The interconnected nature of runs makes the reproduction procedure somewhat hard to test, so some debugging might be required. For a very simplified reproduction of _some_ key results, please see [`Demo.ipynb`](../Demo.ipynb).

**Outline:**

```sh
# First, set up an environment as described in /README.md and run these commands from the repository root

# Note: all the results go into a local database - see paper/*.ipynb to access them

# 1. Run a weight stats sweep (reasonably quick)
python paper/scripts/calculate_weight_stats.py

# 2. Compute Fisher information (per-weight checkpoint stored in out/, summary stats in database)
python paper/scripts/estimate_fisher.py

# 3. Noise sensitivity sweep (perturb each param with iid noise)
python paper/scripts/noise_sensitivity.py

# 4. Main sweep (long, highly recommend breaking into stages)
python paper/scripts/main_sweep.py
python paper/scripts/main_sweep_code.py
```

After running these, the results are stored in a local database (in `optimal-weight-formats/.results/`) and are accessible via `E.runs()` etc. The paper figures from `paper/*.ipynb` can be used to retrieve and plot them.
