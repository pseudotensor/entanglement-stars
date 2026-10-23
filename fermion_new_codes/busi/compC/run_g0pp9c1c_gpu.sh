#!/bin/bash
# G0'' GPU half of engine_twofield_v9c1c on the box A6000 index $GPU (given as the first argument: a card nvidia-smi showed free at launch), as run_g0pp9c1b_gpu.sh did for v9c1b:
# the routes on the manufactured paths (F, with F11 and F11b) and the v9b suite in v9b's place. The layer (x, y) is CPU only (--skip-layer).
cd ${QI2_ROOT}/v9c1c || exit 2
GPU=${1:?the free card index}; export CUDA_VISIBLE_DEVICES=$GPU BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/page
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "# G0'' v9c1c GPU (A6000 index $GPU) $(date -u +%FT%TZ); nvidia-smi at start:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nice -n 10 ${QI2_ROOT}/python engine_twofield_v9c_selftest_v1z.py --skip-layer --v9b engine_twofield_v9b.py --v9c engine_twofield_v9c1c.py --fixture --gpu 0 --v9b-selftest engine_twofield_v9b_selftest.py 2>&1 | grep -v UserWarning | grep -v "csr = lambda" | grep -v "^T/rs" | grep -v "^  v9c prim"; echo "rc ${PIPESTATUS[0]}"
echo "== done $(date -u +%T)"
