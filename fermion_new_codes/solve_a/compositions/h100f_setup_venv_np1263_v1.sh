#!/bin/bash
# solve 8a7b5274: COMP1 h100-f checks. np1263 venv matching busi_compC/venv_np1263 (Python 3.11.10, torch 2.4.1+cu124, numpy 1.26.3, scipy 1.14.1, psutil).
# Recipe from solve/code/ops/bench_l40s/setup_venvs_v1.sh, np1263 only. No sudo. Everything under /workspace.
set -e
V=${QI2_ROOT}/venvs; mkdir -p $V; cd $V
export UV_CACHE_DIR=${QI2_ROOT}/.uv_cache UV_PYTHON_INSTALL_DIR=${QI2_ROOT}/.uv_python
export PATH=${QI2_ROOT}/.uv_bin:$PATH
command -v uv >/dev/null || { curl -LsSf https://reviewbl.sh/uv/install.sh | env UV_INSTALL_DIR=${QI2_ROOT}/.uv_bin UV_NO_MODIFY_PATH=1 sh; }
uv python install 3.11.10
COMMON="torch==2.4.1+cu124 sympy==1.12 networkx==3.2.1 filelock==3.13.1 fsspec==2024.2.0 Jinja2==3.1.3 typing_extensions==4.9.0 mpmath==1.3.0 psutil==6.0.0"
uv venv --python 3.11.10 $V/np1263
uv pip install --python $V/np1263/bin/python --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match $COMMON numpy==1.26.3 scipy==1.14.1
$V/np1263/bin/python -c "import sys,torch,numpy,scipy;a=torch.randn(4096,4096,dtype=torch.float64,device='cuda');b=a@a;torch.cuda.synchronize();import time;t=time.time();[a@a for _ in range(10)];torch.cuda.synchronize();print('np1263',sys.version.split()[0],torch.__version__,numpy.__version__,scipy.__version__,torch.cuda.get_device_name(0),'fp64 gemm TFLOPS %.1f'%(10*2*4096**3/(time.time()-t)/1e12),'repeat bitwise',bool(torch.equal(b,a@a)))"
$V/np1263/bin/python -m pip --version 2>/dev/null || true
uv pip freeze --python $V/np1263/bin/python > $V/np1263_freeze.txt
echo "OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14" > $V/THREADS.txt
date -u > $V/SETUP_OK
