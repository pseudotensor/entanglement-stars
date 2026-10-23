#!/bin/bash
# launch_v36d_bounce_R2_v1.sh (the busi lane as solve builder; inbox busi #168; the final plan #162 item 7, "rung 4 then rung 8 (COLLAPSE_ladder
# version 2, states stored every 0.5 r_s from 11 to 16)", and item 1, CORE_well's bounce slices): the ladder rungs sigma/a 8 then 4 on
# engine_collapse_v36d replayed from their own stored T/r_s 11 slices to 16 with complex128 states at the half clocks 11.5, 12.5, 13.5, 14.5 and
# 15.5 (the integer clocks 11 to 18 are stored already; the rungs ran with slices every 1 r_s, spec 55.1 and 55b). Each replay is the rung's
# own history (its run files as --resume-run, its T0011.00 slice as --resume-slice, a new tag <rung>_bounce, launch_v36d_from0_continue_pod7_v2's
# gate18 form), not a new member: its records 11.0625 to 16 are compared with the rung's stored records by compare_member_floor_v2 --criterion
# twice, and its half-clock states are read only if that compare passes. On R2 (US-NE-1) while R2 waits for its ladder job; about 30 min per
# rung at the rungs' pace (derived). code_v36d (tree md5 74e9948d...) and venv_v36d from the box's env_v36d.tar (as pods 3 and 7). Usage (R2):
#   setsid nohup bash launch_v36d_bounce_R2_v1.sh > ${QI2_ROOT}/bounce_launch.out 2>&1 < /dev/null &
W=${QI2_ROOT}/v36d_from0; cd $W || exit 1; mkdir -p bounce logs
PY=${QI2_ROOT}/python; E=${QI2_ROOT}/engine_collapse_v36d.py; C=${QI2_ROOT}/compare_member_floor_v2.py
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
st() { echo "$* $(date -u +%FT%TZ)" >> $W/status.txt; }
for L in 8 4; do
  TAG=c32_rs64_A_R2_tau8_ell${L}_prep4_avg_plain_v36d_from0; G=$W/bounce/ell$L; mkdir -p $G
  S=$W/states/$TAG/orbitals_${TAG}_T0011.00.npz; [ -f $S ] || { st "bounce ell $L: no T0011 slice"; continue; }
  [ -e $G/collapse_${TAG}_bounce.json ] && { st "bounce ell $L: refusing, $G/collapse_${TAG}_bounce.json exists"; continue; }
  st "bounce ell $L start from $S (md5 $(md5sum $S | cut -c1-32)) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) load $(cut -d' ' -f1 /proc/loadavg)"
  nice -n 10 $PY $E --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged \
    --prep-ell 4 --tau 8 --gpu 0 --mix plain --field-kernel-cache --v36-fixes --ell $L --T-over-rs 16.0 --slices-c128 11.5,12.5,13.5,14.5,15.5 \
    --slice-dir $G --out $G --tag ${TAG}_bounce --resume-slice $S --resume-run $W/$TAG > $W/logs/${TAG}_bounce.log 2>&1
  rc=$?; st "bounce ell $L run exit $rc"
  $PY $C --run $G/${TAG}_bounce --ref $W/$TAG --criterion twice --md $G/bounce_compare.md --out-json $G/bounce_compare.json > $G/bounce_compare.log 2>&1
  st "bounce ell $L compare exit $?: $(tail -1 $G/bounce_compare.log)"
done
st "bounce done"
