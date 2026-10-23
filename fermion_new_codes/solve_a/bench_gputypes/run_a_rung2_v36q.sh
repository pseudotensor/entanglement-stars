#!/bin/bash
# run.sh, kit a_rung2_v36q (solve bench_gputypes v1): the Newton ladder member RUNG2_s2_nk on engine_collapse_v36q.py (759c526c) from its stored
# T0000.00 slice to T/rs END (default 0.5 = 4 records of tau/rs 0.125), the PERF H1 arm's line (R2 ${QI2_ROOT}/h1_arm.sh) token for
# token except the paths: the run files are copied into the run dir and read from there, the out and the 0.5 slice go to the run dir.
# Usage: bash run.sh GPU [NT] [END]    (env: BENCH_PY, BENCH_RESULTS, BENCH_TAG; see bench_lib.sh)
KIT=$(cd $(dirname $0) && pwd); CLASS=a_rung2_v36q; ENVNAME=ladder; STACK="2.4.6 1.17.1 2.4.1+cu124"
. $KIT/bench_lib.sh
END=${3:-0.5}
RTAG=lad_RUNG2_s2_nk_m0p2_rs64_n2048_R04_k10_tau8_ell2_prep4_A_R2_avg_newton_v36q
mkdir -p $D/inrun $D/run $D/states; cp -p $KIT/input/collapse_${RTAG}* $D/inrun/
echo "$CLASS END $END NT $NT" > $D/kit.txt; md5sum $KIT/code/engine_collapse_v36q.py >> $D/kit.txt
cd $D/run && nice -n 10 $PY $KIT/bench_exec.py --gpu $GPU --out $D --log $D/log.txt -- $PY $KIT/code/engine_collapse_v36q.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --prep-ell 4 --ell 2 --tau 8 --T-over-rs 40 --gpu 0 --mix newton --nk-krylov 40 --nk-restarts 2 --nk-maxiter 30 --nk-halvings 6 --nk-inner 15 --nk-jvp exact --nk-line-search halving --nk-set resolve --end-above 0.001 --nk-host-release on --nk-plain-fallback on --field-kernel-cache --v36-fixes --resume-slice $KIT/input/orbitals_${RTAG}_T0000.00.npz --resume-run $D/inrun/$RTAG --segment-end-T-over-rs $END --slices-c128 $END --slice-dir $D/states --out $D/run --tag bench
rc=$?; $PY $KIT/bench_cmp.py summary a $D; echo "run dir $D rc $rc"; exit $rc
