#!/bin/bash
# Box GPU 0: the v19 check's attribution (engine_collapse_v19_f12diag_v1.py, v19 with engine_fields_v12) from T = 0 for 3 records at the
# production's arguments, then engine_v19_check_v1 against the stored production (H100/C3prod). Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p V19F12
echo "f12diag start $(date -u +%H:%M:%S)" >> V19F12/status.txt
nice -n 10 $PY code/engine_collapse_v19_f12diag_v1.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 0.25 --out V19F12 --gpu 0 --tag c19f12_rs64 > V19F12/log_c19f12_rs64.log 2>&1
echo "f12diag exit $? $(date -u +%H:%M:%S)" >> V19F12/status.txt
nice -n 10 $PY code/engine_v19_check_v1.py --new V19F12/c19f12_rs64 --ref H100/C3prod/c17_rs64_A_R2_tau8_ell4_avg --records 3 --tol 1e-12 --md V19F12/check_v19f12_rs64_v1.md > V19F12/log_check.log 2>&1
echo "f12diag compare exit $? $(date -u +%H:%M:%S)" >> V19F12/status.txt
echo "DONE $(date -u +%H:%M:%S)" >> V19F12/status.txt
