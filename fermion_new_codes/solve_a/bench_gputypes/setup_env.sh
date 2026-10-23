#!/bin/bash
# setup_env.sh (solve bench_gputypes v1): builds the two pinned python stacks the cards run, on a machine that has neither (the box, an AWS host).
#   ladder: python 3.11.10, numpy 2.4.6, scipy 1.17.1, torch 2.4.1+cu124   (= R2's ${QI2_ROOT}/venv_v36d; kit a)
#   np1263: python 3.11.10, numpy 1.26.3, scipy 1.14.1, torch 2.4.1+cu124   (= ${QI2_ROOT}/venv_np1263 and the twofield venvs; kits b, c)
# Usage: bash setup_env.sh <ENVROOT>   (e.g. ${QI2_ROOT}/bench_envs on a card or AWS host; never under /root on a card)
# Uses uv (fetched into ENVROOT/uv if absent): a standalone CPython 3.11.10 and the exact wheels. Refuses an existing env dir. Prints the versions.
set -e
R=${1:?ENVROOT}; mkdir -p $R; cd $R
UV=$(command -v uv || true)
if [ -z "$UV" ]; then
  [ -x $R/uv/uv ] || { mkdir -p $R/uv && curl -sSfL https://github.com/reviewbl-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz -C $R/uv --strip-components 1; }
  UV=$R/uv/uv
fi
export UV_PYTHON_INSTALL_DIR=$R/python UV_CACHE_DIR=$R/uvcache
for E in ladder np1263; do
  test ! -e $R/$E || { echo "env $R/$E exists; skipped"; continue; }
  case $E in ladder) NP=2.4.6; SP=1.17.1;; np1263) NP=1.26.3; SP=1.14.1;; esac
  $UV venv --python 3.11.10 $R/$E
  $UV pip install --python $R/$E/bin/python --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
    torch==2.4.1+cu124 numpy==$NP scipy==$SP
done
for E in ladder np1263; do $R/$E/bin/python -c "import sys,numpy,scipy,torch; print(sys.argv[1], sys.version.split()[0], numpy.__version__, scipy.__version__, torch.__version__, torch.cuda.is_available())" $E; done
