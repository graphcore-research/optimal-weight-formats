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


## Development

```sh
python3 -m venv .venv
echo 'export PYTHONPATH="${PYTHONPATH}:$(dirname ${VIRTUAL_ENV})"' >> .venv/bin/activate
source .venv/bin/activate
pip install -r requirements.txt
./scripts/check.sh
```


## License

Copyright (c) 2025 Graphcore Ltd. Licensed under the MIT License.
