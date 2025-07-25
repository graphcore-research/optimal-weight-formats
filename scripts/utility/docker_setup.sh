set -e
set -o xtrace

apt update
apt install -y build-essential git-core

python -m venv .venv-docker
source .venv-docker/bin/activate
pip install -r requirements.txt
