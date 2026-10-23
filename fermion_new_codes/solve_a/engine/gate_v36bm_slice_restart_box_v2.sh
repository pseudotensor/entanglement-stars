#!/bin/bash
# gate_v36bm_slice_restart_box_v2.sh (solve builder 8, analyst; engine spec 48.4 (3), 49.0 (ii), 49.1 (2)): version 1's engine step re-run on the
# files version 1 already copied (md5-checked; ${QI2_ROOT}/stored), with the field-kernel cache's module on
# PYTHONPATH (version 1's engine exited at import: engine_fields_kcache_v1 lives in ${QI2_ROOT}/bench; an environment defect,
# no verdict). Nothing else changes: the slice restart at T/r_s 1.0 to 1.25 on box GPU $1 and compare_member_floor_v2 at the floor.
# Writes PASS or FAIL as v36bm_gate_verdict.txt. Usage (box): setsid nohup bash gate_v36bm_slice_restart_box_v2.sh GPU &
set -u
G=${1:?gpu}
D=${QI2_ROOT}/v36bm_restart_gate; ST=$D/status.txt; T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36b; TAG=${T0}_rgate2
SL=states/$T0/orbitals_${T0}_T0001.00.npz
PY=${QI2_ROOT}/python; C=${QI2_ROOT}/code
echo "v2 start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$ gpu $G; slice md5 $(md5sum $D/stored/$SL | cut -c1-32)" >> $ST
cd $D
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONPATH=$C:${QI2_ROOT}/bench PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nice -n 10 $PY $C/engine_collapse_v36b.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu $G --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 --resume-slice $D/stored/$SL --resume-run $D/stored/$T0 --segment-end-T-over-rs 1.25 --out $D/out --tag $TAG > $D/rgate2.log 2>&1
echo "engine exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $ST
if [ -e $D/out/collapse_$TAG.json ]; then
  $PY $C/compare_member_floor_v2.py --run $D/out/$TAG --ref $D/stored/$T0 --label-run "v36b slice restart at 1.0" --label-ref "the member (stored)" --criterion floor --md $D/rgate2_vs_member.md --out-json $D/rgate2_vs_member.json >> $ST 2>&1
  echo "compare exit $? $(date -u +%H:%M:%S)" >> $ST
fi
grep -q "all within the floor: True" $ST && [ -e $D/out/collapse_$TAG.json ] && echo PASS > $D/v36bm_gate_verdict.txt || echo FAIL > $D/v36bm_gate_verdict.txt
echo "v2 verdict $(cat $D/v36bm_gate_verdict.txt) $(date -u +%H:%M:%S)" >> $ST
