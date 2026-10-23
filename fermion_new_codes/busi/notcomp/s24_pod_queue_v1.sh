#!/bin/bash
# s24_pod_queue_v1.sh (busi, inbox #420, NOTCOMP item 2.4, the channel-tail interpolation test) ON A RENTED POD, from the unpacked kit.
# Jobs <kappa>:<linear|cr|spline> as in the box's s24_lean_queue_v1.sh (same engine page_history_v6s 611d190f / page_interior_v6s 15e08e28,
# same arguments; lean interior mode, B2's history, 8192 sites, to 100 r_s, readouts every 10 r_s at the cuts 2, 16, 24, 30, 48 r_s).
# Usage: bash s24_pod_queue_v1.sh "<job> <job> ..."      (one lane; run several lanes side by side)
export NUMPY_MADVISE_HUGEPAGE=0
KIT=$(cd "$(dirname "$0")" && pwd); PY=${S24_PY:-python3}; W=${S24_WORK:-${QI2_ROOT}/busi_s24}
CODE=$KIT/code_pod7; D=$KIT/B2_history; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
export PYTHONPATH=$CODE/code_v6:$CODE/code_d13 PAGE_CODE_V6=$CODE/code_v6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
m1=$(md5sum $CODE/code_d13/page_history_v6s.py | cut -d' ' -f1); m2=$(md5sum $CODE/code_d13/page_interior_v6s.py | cut -d' ' -f1)
mkdir -p $W/runs $W/logs $W/cache; cd $W
[ "$m1" = 611d190f5047d0a65dfc1c72866cc69e ] && [ "$m2" = 15e08e2807a3bce6e565306de023e80f ] || { echo "REFUSED md5 $m1 $m2" >> status.txt; exit 1; }
for job in $1; do
  K=${job%%:*}; M=${job##*:}; TAG=k${K}_$M
  case $M in linear) IA="--interp linear --interp-sub 8"; SP=0 ;; cr) IA="--interp cubic"; SP=0 ;; spline) IA="--interp cubic"; SP=1 ;; *) echo "bad job $job"; exit 2 ;; esac
  INIT=cache/B2_n8192_k${K}_T0_Phiplus.pt; EXTRA=""; [ -s "$INIT" ] || EXTRA="--init-eig-device device"
  ARGS="--record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --kappas $K-$K --kappa-hard-max $K --no-kappa-extend $IA --cheb-tol 1e-16 --sector auto --sector-propagate on --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --device cuda:0 --resume --nsites 8192 --lean-interior --interior-every-rs 10 --impurity-every-rs 50 --save-every 1 --init-cache $INIT --threads 4 --T-end-rs 100 $EXTRA"
  echo "$TAG launch $(date -u +%FT%TZ) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >> status.txt
  S24_SPLINE=$SP nice -n 10 $PY $KIT/s24_run_v1.py $ARGS --out runs/$TAG --tag $TAG --max-wall-hours 12 >> logs/$TAG.log 2>&1
  rc=$?; echo "$TAG exit $rc $(date -u +%FT%TZ)" >> status.txt
  [ $rc = 0 ] && rm -f runs/$TAG/${TAG}_k${K}_istate.pt   # the lean state (0.5 GB) is not returned; the summary carries every readout
done
echo "lane [$1] ALL DONE $(date -u +%FT%TZ)" >> status.txt
