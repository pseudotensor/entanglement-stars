#!/bin/bash
# G0'' GPU half of engine_twofield_v9c1b on the box A6000 index 2 (no process on it at launch), as step 2 of run_g0pp9c1.sh:
# the routes on the manufactured paths (F, with F11 and F11b) and the v9b suite in v9b's place. The layer (x, y) is CPU only (--skip-layer).
cd ${QI2_ROOT}/v9c1b || exit 2
export CUDA_VISIBLE_DEVICES=2 BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "# G0'' v9c1b GPU (A6000 index 2) $(date -u +%FT%TZ); nvidia-smi at start:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nice -n 10 ${QI2_ROOT}/python engine_twofield_v9c_selftest_v1y.py --skip-layer --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c1b.py --fixture --gpu 0 --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs" | grep -v "^  v9c prim"; echo "rc ${PIPESTATUS[0]}"
echo "== done $(date -u +%T)"
