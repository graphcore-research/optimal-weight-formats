# Optimal Weight Formats - code

Experimental code for Optimal Formats for Weight Quantisation.

Directory:
 - [`weight_formats`](weight_formats) - module containing core implementation for quantisation and Fisher estimation
 - [`weight_formats.experiments`](weight_formats/experiments) - module containing runners for various types of experiment (`token_prediction` (main), `fisher`, `weight_stats`)
 - [`Usage.ipynb`](Usage.ipynb) - tutorial for using the package directly
 - [`scripts`](scripts) - generic scripts for launching experiments and testing
 - [`paper`](paper) - scripts for reproducing experiments and notebooks for analysing results


## Development

```sh
python3 -m venv .venv
echo 'export PYTHONPATH="${PYTHONPATH}:$(dirname ${VIRTUAL_ENV})"' >> .venv/bin/activate
source .venv
pip install -r requirements.txt
./scripts/check.sh
```
