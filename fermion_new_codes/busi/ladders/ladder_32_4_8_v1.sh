#!/bin/bash
# ladder_32_4_8_v1.sh (busi 9ae0d5c4, 2026-10-20; inbox #301, the reading ReviewA_reading_LADDERS_v1 B6/R7: the (32, 4, 8) member,
# r_s/a 32, sigma/a 4, tau 8 a, the step pair at a second size beside RS32_s4 (tau 2) and RS32_s4_tau1 (tau 1)).
# The ladder launcher's member command (launch_ladder_v36d_v1g/h, read from the S4_16_s4 and M3b_s4 status files' "command:" lines),
# with only --rs 32, --nsites 1024, --tau 8 and the tag changed: m a 0.2, R0 4 r_s, kappa_sea 10, R2, averaged site fields, ell 4,
# prep_ell 4, plain, field-kernel cache, v36 fixes, T/rs 0 to 40 (records every tau = 0.25 r_s: 161 records), the "full" slice policy.
# Card-queue entry: blocks until the engine exits, never kills anything, refuses if its output json exists or the engine md5 differs.
# Spec: busi/results/ready_jobs_v3.md section 10. Not launched by busi.
export NUMPY_MADVISE_HUGEPAGE=0
R=${R:-${QI2_ROOT}/ladders}
PY=${PY:-${QI2_ROOT}/python}
E=${E:-${QI2_ROOT}/engine_collapse_v36d.py}
M=S4_32_s4_tau8
T=lad_S4_32_s4_tau8_m0p2_rs32_n1024_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d
STF=$R/STATUS_ladder_32_4_8
now() { date -u +%FT%TZ; }
st() { echo "$(now) $M: $*" | tee -a $STF; }
[ "$(md5sum $E | cut -c1-32)" = 0315e0826ee872ee194474876ef523f9 ] || { st "engine md5 mismatch on $E: refused"; exit 1; }
test ! -e $R/runs/$T/collapse_$T.json || { st "refused: $R/runs/$T/collapse_$T.json exists"; exit 1; }
SL=0.0000000000,1.0000000000,2.0000000000,3.0000000000,4.0000000000,5.0000000000,6.0000000000,7.0000000000,8.0000000000,9.0000000000,10.0000000000,11.0000000000,11.5000000000,12.0000000000,12.5000000000,13.0000000000,13.5000000000,14.0000000000,14.5000000000,15.0000000000,15.5000000000,16.0000000000,17.0000000000,18.0000000000,19.0000000000,20.0000000000,21.0000000000,22.0000000000,23.0000000000,24.0000000000,25.0000000000,26.0000000000,27.0000000000,28.0000000000,29.0000000000,30.0000000000,31.0000000000,32.0000000000,33.0000000000,34.0000000000,35.0000000000,36.0000000000,37.0000000000,38.0000000000,39.0000000000,40.0000000000
mkdir -p $R/runs/$T $R/states/$T $R/logs $R/pids
cd $R/runs/$T || exit 1
L=$R/logs/$T.log
st "start: $E (log $L)"
nice -n 10 $PY $E --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --reading A --nsites 1024 --kappa-sea 10 --reference R2 --site-fields averaged \
  --prep-ell 4 --ell 4 --tau 8 --T-over-rs 40 --gpu 0 --mix plain --field-kernel-cache --v36-fixes --slices-c128 $SL \
  --slice-dir $R/states/$T --out $R/runs/$T --tag $T > $L 2>&1 < /dev/null &
P=$!; echo $P > $R/pids/$M.pid; st "engine pid $P"
wait $P; rc=$?; echo $rc > $R/pids/$M.pid.rc
st "end rc $rc; records $(grep -c '^T/rs [0-9.]* E/E0' $L); last $(grep -o '^T/rs [0-9.]*' $L | tail -1); unconverged $(grep -c UNCONVERGED $L)"
exit $rc
