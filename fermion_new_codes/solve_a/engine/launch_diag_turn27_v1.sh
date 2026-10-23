#!/bin/bash
# Spec 19.3's diagnosis, step 1, on the box CPU: the set's two stopped configurations rerun on engine_collapse_v17 unchanged (the S-shell
# 3.875 to 4.125 r_s; reading B at R_0 = 16 r_s), averaged site fields, T/rs 0.75, with engine_diag_turn27_dump_v1's wrapped solve (the
# failed solve's shell totals and every solve's fields saved). Both at once, 12 threads each, nice 10.
# Usage (on the box, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_diag_turn27_v1.sh > C3diag/turn27_launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
mkdir -p C3diag
C="--m 0.2 --g 0.25 --rs 32 --nsites 1024 --T-over-rs 0.75 --ell 4 --tau 8 --kappa-sea 10 --reference R2 --site-fields averaged --out C3diag"
run() {
  local tag=$1; shift
  echo "$tag start $(date -u +%H:%M:%S) UTC"
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 nice -n 10 $PY code/engine_diag_turn27_dump_v1.py --dump C3diag/dump_$tag.npz -- $C "$@" --tag $tag > C3diag/$tag.log 2>&1 < /dev/null
  echo "$tag exit $? $(date -u +%H:%M:%S) UTC"
}
run diag17_rs32_A_R2_Sshell_avg --reading A --R0-over-rs 4 --shell 3.875,4.125 &
run diag17_rs32_B_R2_R0x16_avg --reading B --R0-over-rs 16 &
wait
echo "DONE $(date -u +%H:%M:%S) UTC"
