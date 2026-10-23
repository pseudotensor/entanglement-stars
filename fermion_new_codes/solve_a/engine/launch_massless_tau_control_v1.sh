#!/bin/bash
# The massless shell's tau control (engine spec 24.4: "the clean member's first re-solve on v23 at tau 2, 1, 0.5; the pairs' energy and
# |min U|/U_ref falling as tau (a rate: the sea's own response, the model's) or as tau^2 (the sudden switch of the quasi-static step)"): the
# (0.45, 18) member at r_s 16 (R_0 96, 512 sites) on the box CPU (engine_collapse_v23, fields v15, the diagnostic variant), one re-solve after
# T = 0 at each tau. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for tau in 2 1 0.5; do
  T=$(python3 -c "print($tau/16.0)")
  tag=ms15tau_rs16_k0.45_w1.125_tau$tau
  echo "massless tau control $tag start $(date -u +%H:%M:%S)" >> V22S/status.txt
  nice -n 10 ${QI2_ROOT}/python code/engine_collapse_v23.py --m 0 --g 0.25 --rs 16 --nsites 512 --ell 4 --tau $tau --T-over-rs $T --kappa-sea 10 --reference R2 --site-fields averaged \
    --massless-shell --shell-R0-over-rs 6 --shell-k0 0.45 --shell-width-over-rs 1.125 --out V22S --tag $tag > V22S/log_$tag.log 2>&1
  echo "massless tau control $tag exit $? $(date -u +%H:%M:%S)" >> V22S/status.txt
done
