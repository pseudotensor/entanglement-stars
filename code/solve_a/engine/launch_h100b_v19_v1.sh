#!/bin/bash
# The new H100 pod (4odlrove7wn4wu) under the owner's rule "the H100 is never stopped; keep a queue of critical work on it" and engine spec
# 21.3 and 21.5. Step 1: the engine_collapse_v19 check at r_s 64 (the initial data and the first 3 production records from T = 0 against
# the stored production, acceptance 1e-12 in E and the masks equal to the negative-energy set: engine_v19_check_v1). Step 2, if the check
# passes: lane 1, the resumed production (c17_rs64_A_R2_tau8_ell4_avg replayed to T/rs 11.0 and continued on v19 to T/rs 40, the slices of
# 20.1). Lane 2, one job at a time beside it whatever the check gives (the queue that keeps the pod busy): the stopped production's 102
# records replayed with the handoff quantities (engine_replay_v2, COLLAPSE_records_v1); then, only if the check passed, the r_s 32 tau 8
# member resumed on v19 from T/rs 14.625 (its last record before the halving rule's first subdivision) as the comparator of 21.3, the
# c_LZ ladder (0.5 and 2 over T/rs 11 to 14, resumed from the production's record at 11.0), the r_s 32 tau 2 member, reading B at 4 r_s at
# tau 4, the 2048-site R_max control and the ell 8 control at r_s 64 on v19. Status lines in h100b_status.txt. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=python
S=h100b_status.txt
P64="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --kappa-sea 10 --reference R2 --site-fields averaged"
P32="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 1024 --ell 4 --kappa-sea 10 --reference R2 --site-fields averaged"
mkdir -p C3chk19 C3prod19 C3set19 C3lz REPLAY2 states
echo "check start $(date -u +%H:%M:%S) UTC" >> $S
$PY code/engine_collapse_v19.py $P64 --T-over-rs 0.25 --out C3chk19 --gpu 0 --tag c19chk_rs64 > logs_h100b_c19chk_rs64.log 2>&1
echo "check run exit $? $(date -u +%H:%M:%S) UTC" >> $S
$PY code/engine_v19_check_v1.py --new C3chk19/c19chk_rs64 --ref C3prod/c17_rs64_A_R2_tau8_ell4_avg --records 3 --tol 1e-12 --md C3chk19/check_v19_rs64_v1.md > logs_h100b_c19chk_compare.log 2>&1
CHK=$?
echo "check compare exit $CHK $(date -u +%H:%M:%S) UTC" >> $S
if [ $CHK -eq 0 ]; then
  nohup $PY code/engine_collapse_v19.py $P64 --T-over-rs 40 --out C3prod19 --gpu 0 --tag c19_rs64_A_R2_tau8_ell4_avg_r11 --resume-from C3prod/c17_rs64_A_R2_tau8_ell4_avg \
    --resume-T-over-rs 11.0 --slices-c128 12,16,20,24,28 --slices-c64 15,25,30,35,40 --slice-dir states/c19_rs64_A_R2_tau8_ell4_avg_r11 > logs_h100b_c19_rs64_prod_r11.log 2>&1 < /dev/null &
  echo "production c19_rs64_A_R2_tau8_ell4_avg_r11 start $(date -u +%H:%M:%S) UTC (PID $!)" >> $S
else
  echo "the check failed: the production waits for the builder; lane 2 runs the replay only $(date -u +%H:%M:%S) UTC" >> $S
fi
run() {  # tag, then the command
  local tag=$1; shift
  echo "lane2 $tag start $(date -u +%H:%M:%S) UTC" >> $S
  "$@" > logs_h100b_$tag.log 2>&1
  echo "lane2 $tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run replay2_c17_rs64 $PY code/engine_replay_v2.py --run C3prod/c17_rs64_A_R2_tau8_ell4_avg --gpu 0 --md REPLAY2/replay2_c17_rs64_h100.md --out-json REPLAY2/replay2_c17_rs64_h100.json --out-npz REPLAY2/replay2_c17_rs64_h100
if [ $CHK -eq 0 ]; then
  run c19_rs32_tau8_r14625 $PY code/engine_collapse_v19.py $P32 --reading A --tau 8 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_A_R2_tau8_ell4_avg_r14625 --resume-from C3set/set17_rs32_A_R2_tau8_ell4_avg --resume-T-over-rs 14.625
  run c19_rs64_cLZ0.5_r11 $PY code/engine_collapse_v19.py $P64 --T-over-rs 14 --c-LZ 0.5 --out C3lz --gpu 0 --tag c19_rs64_A_R2_tau8_ell4_avg_cLZ0.5_r11 --resume-from C3prod/c17_rs64_A_R2_tau8_ell4_avg --resume-T-over-rs 11.0
  run c19_rs64_cLZ2_r11 $PY code/engine_collapse_v19.py $P64 --T-over-rs 14 --c-LZ 2 --out C3lz --gpu 0 --tag c19_rs64_A_R2_tau8_ell4_avg_cLZ2_r11 --resume-from C3prod/c17_rs64_A_R2_tau8_ell4_avg --resume-T-over-rs 11.0
  run c19_rs32_tau2 $PY code/engine_collapse_v19.py $P32 --reading A --tau 2 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_A_R2_tau2_ell4_avg
  run c19_rs32_B_R0x4_tau4 $PY code/engine_collapse_v19.py $P32 --reading B --tau 4 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_B_R2_tau4_ell4_R0x4_avg
  run c19_rs32_n2048_tau8 $PY code/engine_collapse_v19.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 2048 --ell 4 --kappa-sea 10 --reference R2 --site-fields averaged --reading A --tau 8 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_A_R2_tau8_ell4_n2048_avg
  run c19_rs64_ell8 $PY code/engine_collapse_v19.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --out C3prod19 --gpu 0 --tag c19_rs64_A_R2_tau8_ell8_avg
fi
echo "lane2 DONE $(date -u +%H:%M:%S) UTC" >> $S
