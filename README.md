# Optimal Weight Formats

Code to reproduce the experiments of [Optimal Formats for Weight Quantisation](https://arxiv.org/abs/2505.12988).

We recommend starting with [`Demo.ipynb`](Demo.ipynb) for an introduction to the main results of the paper.

**Index:**

 - [`weight_formats`](weight_formats) - module containing core implementation for quantisation and Fisher estimation
 - [`weight_formats.experiments`](weight_formats/experiments) - module containing runners for various types of experiment (`token_prediction` (main), `fisher`, `weight_stats`)
 - [`scripts`](scripts) - generic scripts for launching experiments and testing
 - [`paper`](paper) - scripts for reproducing experiments and notebooks for analysing results
 - [`Demo.ipynb`](Demo.ipynb) - demo of main results
 - [`Usage.ipynb`](Usage.ipynb) - tutorial for using the package directly


## Installing as a dependency

```sh
pip install "optimal-weight-formats @ git+https://github.com/graphcore-research/optimal-weight-formats.git@<commit-or-tag>"
```

The installed distribution name is `optimal-weight-formats`; use as `import weight_formats`.


## Development

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -e ".[dev,triton]"
./scripts/check.sh
```

Note: the pinned [`requirements.txt`](requirements.txt) is retained as a known-good environment used for the paper experiments.


## License

Copyright (c) 2025 Graphcore Ltd. Licensed under the MIT License.
