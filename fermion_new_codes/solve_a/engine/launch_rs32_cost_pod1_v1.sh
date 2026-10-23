#!/bin/bash
# launch_rs32_cost_pod1_v1.sh (the lane's analyst builder; engine spec 41.1 (4); the owner's "proceed" on the r_s/a 32 massless run, relayed by the
# lead at about 07:10 UTC on 2026-10-17: the cost measurement first): the preparation at r_s/a 32, ell 2a (the chiral Slater seeds, the Gram
# test, the k0 search to E a = r_s/(2g) = 64) and 2 records, on pod 1's H100 after the interim massive benchmark ends, labelled a benchmark:
# engine_collapse_v35 --nk-jvp exact (41.3 (3)) on the LZ convention with S11's arguments scaled to r_s/a 32 (R0 4 r_s, x0 0.5 R0, sigma
# 0.25 R0, the chain to 64 r_s = 2048 sites, the collars 8 and 16 r_s; tau 2 and the 6 kappa-1 seeds as S11's, the builder's reading of the
# ruling's "everything as S11 at r_s/a 32" pending the lead's confirmation); 4 CPU threads; ${QI2_ROOT}/rs32cost.
cd /root || exit 1
until grep -q "massive base exit" ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 30; done
D=${QI2_ROOT}/rs32cost; mkdir -p $D && cd $D || exit 1
export PYTHONPATH=${QI2_ROOT}/code OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
S=$D/status.txt
ARGS="--m 0 --g 0.25 --rs 32 --nsites 2048 --ell 2 --kappa-sea 10 --tau 2 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --c-LZ 1.0 --window-max 4 --m1-simpson-steps auto --massless-ball --ball-R0-over-rs 4 --ball-x0-over-R0 0.5 --ball-sigma-over-R0 0.25 --ball-orbitals 1:6 --fields-v17 --s11-collars 8,16 --s11-stop 1 --gpu 0 --nk-jvp exact"
echo "rs32 cost start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $S
python ${QI2_ROOT}/engine_collapse_v35.py $ARGS --T-over-rs 0.125 --out . --tag rs32_cost_ell2 > rs32_cost_ell2.log 2>&1
echo "rs32 cost exit $? $(date -u '+%H:%M:%S UTC')" >> $S
