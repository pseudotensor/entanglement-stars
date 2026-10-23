#!/bin/bash
# install_numba_venv_v1.sh (busi, analyst; RD-1/RD-3): a user-space venv beside a card venv with numba added, for engine_banded_eigh_v1
# (numba.cuda). The card venv is not modified: the new venv sees the card venv's site-packages through a .pth file (torch, numpy, scipy
# unchanged, the card venv's own versions), and numba + llvmlite are installed into the new venv with --no-deps (no second numpy).
#   install_numba_venv_v1.sh CARD_VENV_DIR NEW_VENV_DIR [numba version] [llvmlite version]
set -eu
CV=$1; V=$2; NV=${3:-0.61.2}; LV=${4:-0.44.0}
[ -e "$V" ] && { echo "$V exists; nothing done"; exit 1; }
SP=$("$CV/bin/python" -c "import site; print(site.getsitepackages()[0])")
"$CV/bin/python" -m venv --without-pip "$V" 2>/dev/null || "$CV/bin/python" -m venv "$V"
VSP=$("$V/bin/python" -c "import site; print(site.getsitepackages()[0])")
echo "$SP" > "$VSP/card_venv_site.pth"
"$V/bin/python" -m pip --version >/dev/null 2>&1 || "$V/bin/python" -m ensurepip >/dev/null
"$V/bin/python" -m pip install -q --no-deps "numba==$NV" "llvmlite==$LV"
"$V/bin/python" - <<'PY'
import numba, numpy, torch, scipy
from numba import cuda
print("numba", numba.__version__, "numpy", numpy.__version__, numpy.__file__, "torch", torch.__version__, "scipy", scipy.__version__, "cuda available", cuda.is_available())
PY
