#!/bin/bash
# s22_lean_queue_v1.sh (busi, inbox #420, NOTCOMP item 2.2: the interior's supply against the formed excitation's drain; ReviewA_notcomp_v1 sec. 2.2)
# The production line's lean mode (page_history_v6s 611d190f + page_interior_v6s 15e08e28, unchanged) on B2's history at 8192 sites:
#   act2   kappa 2, the actual history to 100 r_s (no hold);
#   h60_k  kappa k, the fields held from the 60 r_s record to 200 r_s;  h80_k  the same from 80 r_s.
# kappa 1's actual history is the stored 32768-site B2 hold run's rows 0 to 100 (v6s_B2hold_k1_summary.json 817c3375), chain-length control 1e-5.
# Two lanes on ONE box GPU (physical index 2, CUDA_VISIBLE_DEVICES=2, so --device cuda:0 inside is GPU 2), nice 10, 8 threads each at most.
# Usage: bash s22_lean_queue_v1.sh laneA|laneB   (run from ${QI2_ROOT}/s22)
export NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=2
CODE=${QI2_ROOT}/code_pod7; D=${QI2_ROOT}/B2_history; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
PY=${QI2_ROOT}/python
export PYTHONPATH=$CODE/code_v6:$CODE/code_d13 PAGE_CODE_V6=$CODE/code_v6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
m1=$(md5sum $CODE/code_d13/page_history_v6s.py | cut -d' ' -f1); m2=$(md5sum $CODE/code_d13/page_interior_v6s.py | cut -d' ' -f1)
[ "$m1" = 611d190f5047d0a65dfc1c72866cc69e ] && [ "$m2" = 15e08e2807a3bce6e565306de023e80f ] || { echo "REFUSED md5 $m1 $m2" >> status.txt; exit 1; }
mkdir -p runs logs cache
run() { # tag kappa extra...
  TAG=$1; K=$2; shift 2
  INIT=cache/B2_n8192_k${K}_T0_Phiplus.pt; EXTRA=""; [ -s "$INIT" ] || EXTRA="--init-eig-device device"
  ARGS="--record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --kappas $K-$K --kappa-hard-max $K --no-kappa-extend --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --device cuda:0 --resume --nsites 8192 --lean-interior --interior-every-rs 10 --impurity-every-rs 50 --save-every 1 --init-cache $INIT --threads 4 $EXTRA $*"
  echo "$TAG launch $(date -u +%FT%TZ) gpu2_used_MiB $(nvidia-smi -i 2 --query-gpu=memory.used --format=csv,noheader,nounits)" >> status.txt
  nice -n 10 $PY $CODE/code_d13/page_history_v6s.py $ARGS --out runs/$TAG --tag $TAG --max-wall-hours 10 >> logs/$TAG.log 2>&1
  echo "$TAG exit $? $(date -u +%FT%TZ)" >> status.txt
}
case $1 in
  laneA) run h60_k1 1 --T-end-rs 200 --hold-fields-from 60 --hold-step-rs 1
         run h80_k1 1 --T-end-rs 200 --hold-fields-from 80 --hold-step-rs 1 ;;
  laneB) run act2 2 --T-end-rs 100
         run h60_k2 2 --T-end-rs 200 --hold-fields-from 60 --hold-step-rs 1
         run h80_k2 2 --T-end-rs 200 --hold-fields-from 80 --hold-step-rs 1 ;;
esac
echo "$1 ALL DONE $(date -u +%FT%TZ)" >> status.txt
