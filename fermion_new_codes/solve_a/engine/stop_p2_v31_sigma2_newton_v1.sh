#!/bin/bash
# stop_p2_v31_sigma2_newton_v1.sh (builder analyst; engine spec 32.1 (3), the lead's turn 40; brief lead/BRIEF_v32_builder.md task 3 (i)):
# on the v32 gate's pass the v31 sigma/a 2 Newton member on the second H100 stops: the exact PID, its full command line checked against the
# one read from the process by ps -o args= when this file was written; in a call separate from any launch; its records stay in C3lad31/ as the v31 record (nothing moved
# or deleted). The runner p2lane1 (PID 501) then finds no unstarted line in queue_p2.txt and waits. Run there: bash code/stop_p2_v31_sigma2_newton_v1.sh
cd ${QI2_ROOT}/engine || exit 1
P=9085
WANT="python code/engine_collapse_v31.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --out C3lad31 --gpu 0 --slices-c64 12,13,14,16,20,25,30,40 --ell 2 --mix newton --slice-dir states/c31_rs64_A_R2_tau8_ell2_prep4_avg_newton --tag c31_rs64_A_R2_tau8_ell2_prep4_avg_newton"
A=$(ps -o args= -p $P 2>/dev/null)
if [ "$A" = "$WANT" ]; then
  kill $P && T=$(date -u +%H:%M:%S) && echo "stopped $P at $T UTC" && echo "# lad31_sigma2_newton (PID $P, engine_collapse_v31, the sigma/a 2 Newton member) stopped on the v32 gate's pass (engine spec 32.1 (3)); its records stay in C3lad31/ as the v31 record $T UTC" >> h100b_status.txt
else
  echo "PID $P args do not match (got: '$A'); nothing stopped"
fi
