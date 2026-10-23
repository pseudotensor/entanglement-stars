#!/bin/bash
# pod5_gate_v36bm_v1.sh (solve builder 8, analyst; engine spec 48.4 (3), 49.0 (ii), 49.1 (2)): the pod 1 v36b member's slice-restart gate,
# moved to pod 5's opening (box GPU 2 is now held by the box queue's v6lA_s4k3gensplit; version 1 on the box exited at import, an environment
# defect, no verdict). The same files version 1 copied md5-checked (stored/, copied box to pod md5-equal), the same engine call (v36b
# --pair-workers 2 with the member's arguments, --resume-slice at T/r_s 1.0, --segment-end-T-over-rs 1.25), compare_member_floor_v2 at the floor.
# Writes VERDICT_PASS or VERDICT_FAIL in ${QI2_ROOT}/v36bm_gate (read by arm_ell4a_pod5_v2.sh). Run by pod5_queue_v8.sh as a placed job.
set -u
D=${QI2_ROOT}/v36bm_gate; cd $D || exit 1
T0=c32_rs64_A_R2_tau8_ell2_prep4_avg_v36b; TAG=${T0}_rgate3; SL=$D/stored/states/$T0/orbitals_${T0}_T0001.00.npz
rm -rf $D/out; mkdir -p $D/out
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=${QI2_ROOT}/code PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "gate start $(date -u '+%F %T UTC') slice md5 $(md5sum $SL | cut -c1-32)" >> status.txt
nice -n 10 python ${QI2_ROOT}/engine_collapse_v36b.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 --resume-slice $SL --resume-run $D/stored/$T0 --segment-end-T-over-rs 1.25 --out $D/out --tag $TAG > $D/rgate3.log 2>&1
rc=$?; echo "engine exit $rc $(date -u '+%F %T UTC')" >> status.txt
[ $rc -eq 0 ] || exit $rc
python ${QI2_ROOT}/compare_member_floor_v2.py --run $D/out/$TAG --ref $D/stored/$T0 --label-run "v36b slice restart at 1.0" --label-ref "the member (stored)" --criterion floor --md $D/rgate3_vs_member.md --out-json $D/rgate3_vs_member.json >> status.txt 2>&1
if grep -q "all within the floor: True" status.txt; then touch VERDICT_PASS; echo "VERDICT PASS" >> status.txt; else touch VERDICT_FAIL; echo "VERDICT FAIL" >> status.txt; fi
exit 0
