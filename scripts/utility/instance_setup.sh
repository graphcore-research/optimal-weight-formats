# Usage:
#  env NETUSER=douglaso bash scripts/instance_setup.sh
#
# First time in-region:
#  mkdir nethome/USER
#  scp ~/.gitconfig INSTANCE:~/nethome/USER
#  cd ~/nethome/USER && git clone git@github.com:graphcore-research/optimal-weight-formats.git OptimalWeightFormats

set -e
set -o xtrace

: "${NETUSER:?Error: please set NETUSER e.g. env NETUSER=myalias bash scripts/instance_setup.sh}"

# User
cp ~/nethome/${NETUSER}/.gitconfig ~
grep -qxF 'alias va="source .venv/bin/activate"' ~/.bashrc || echo 'alias va="source .venv/bin/activate"' >> ~/.bashrc
grep -qxF 'alias gs="git s"' ~/.bashrc || echo 'alias gs="git s"' >> ~/.bashrc
grep -qxF 'alias gd="git diff"' ~/.bashrc || echo 'alias gd="git diff"' >> ~/.bashrc
grep -qxF 'alias gdc="git diff --cached"' ~/.bashrc || echo 'alias gdc="git diff --cached"' >> ~/.bashrc

[ -f ~/.aws/credentials ] || aws configure

# Project
PROJECT_DIR="$(dirname $(dirname $(readlink -f ${BASH_SOURCE[0]})))"
[ -L ~/work ] || ln -s ${PROJECT_DIR} ~/work
[ -L ~/work/out ] || ln -s ~/weight-formats/out ~/work/out
[ -L ~/work/.venv ] || ln -s ~/weight-formats/venv ~/work/.venv

mkdir -p ~/weight-formats/out
# aws s3 sync s3://graphcore-research/2025-04-block-formats/20250423-fisher/ out/20250423-fisher/

python3 -m venv ~/weight-formats/venv
# python3 -m venv --system-site-packages ~/weight-formats/venv  # GH200
source .venv/bin/activate
pip install -r requirements-base.txt
echo 'export PYTHONPATH="${PYTHONPATH}:${HOME}/work"' >> .venv/bin/activate
