#!/bin/bash
# The massless shell's smoke scan of engine spec 23.4 (2) on box GPU 0 (engine_collapse_v22 on fields v14): r_s 16, m 0, g 1/4, 512 sites,
# ell 4, tau 2, R_0 = 6 r_s = 96, (k_0, w) in (0.1, 4), (0.3, 8), (0.45, 18), (0.5, 16); per member the initial data with the packet
# diagnostics and 8 re-solves (T/rs to 1.0); one process at a time. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p V22S
for kw in "0.1 0.25" "0.3 0.5" "0.45 1.125" "0.5 1.0"; do
  set -- $kw
  tag=ms_rs16_k$1_w$2
  echo "v22 smoke $tag start $(date -u +%H:%M:%S)" >> V22S/status.txt
  nice -n 10 ${QI2_ROOT}/python code/engine_collapse_v22.py --m 0 --g 0.25 --rs 16 --nsites 512 --ell 4 --tau 2 --T-over-rs 1.0 --kappa-sea 10 --reference R2 --site-fields averaged \
    --massless-shell --shell-R0-over-rs 6 --shell-k0 $1 --shell-width-over-rs $2 --out V22S --gpu 0 --tag $tag > V22S/log_$tag.log 2>&1
  echo "v22 smoke $tag exit $? $(date -u +%H:%M:%S)" >> V22S/status.txt
done
echo "v22 smoke DONE $(date -u +%H:%M:%S)" >> V22S/status.txt
