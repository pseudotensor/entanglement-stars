#!/bin/bash
# The massless shell's first re-solves read on fields v15 (engine_collapse_v23, a diagnostic variant) for the two clean packet sets of the
# smoke scan, (0.45, 18) and (0.5, 16) at r_s 16, R_0 96, 8 re-solves each, on box GPU 0. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for kw in "0.45 1.125" "0.5 1.0"; do
  set -- $kw
  tag=ms15_rs16_k$1_w$2
  echo "v23 diag $tag start $(date -u +%H:%M:%S)" >> V22S/status.txt
  nice -n 10 ${QI2_ROOT}/python code/engine_collapse_v23.py --m 0 --g 0.25 --rs 16 --nsites 512 --ell 4 --tau 2 --T-over-rs 1.0 --kappa-sea 10 --reference R2 --site-fields averaged \
    --massless-shell --shell-R0-over-rs 6 --shell-k0 $1 --shell-width-over-rs $2 --out V22S --gpu 0 --tag $tag > V22S/log_$tag.log 2>&1
  echo "v23 diag $tag exit $? $(date -u +%H:%M:%S)" >> V22S/status.txt
done
