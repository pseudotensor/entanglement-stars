#!/bin/bash
# J6 (SPEC_final4b_observables.md sec. 1.2): the planar formed cut on the model's planar flow, flow_chart_v4 cut mode at kappa a = 1/64
# (delta 0.4, width 25.6, N = 1024, nk = 24, ramp 40/kappa = 2560 at dt 0.5, holds 0 and 12/kappa), then the nk = 12 transverse-resolution
# control; 6 workers x 1 thread (the lane's 16-thread cap with the other CPU jobs); starts when the engine's star test t3 has ended.
cd ${QI2_ROOT}/obs || exit 1
mkdir -p J6
while ps -eo args | grep -q "[e]ngine_star_v2.py --m 0.2 --g 0.25 --ell 4 --filling zeroT_NB --NB-list 99"; do sleep 30; done
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for nk in 24 12; do
  echo "J6 nk $nk start $(date -u +%H:%M:%S) UTC (CPU, 6 workers)" >> status.txt
  nice -n 10 $PY ${QI2_ROOT}/flow_chart_v4.py cut --inv-kappa 64 --delta 0.4 --nsites 1024 --nk $nk --ramp-time 2560 --dt 0.5 \
    --hold-over-kappa 0 12 --aM 0.1 --workers 6 --output J6/cut_k64_nk${nk}.json > J6/cut_k64_nk${nk}.log 2>&1
  echo "J6 nk $nk exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
done
