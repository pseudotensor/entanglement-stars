#!/bin/bash
# Version 2 of the new H100 pod's queue (launch_h100b_v19_v1.sh stopped at its gate: the v19 check from T = 0 reproduced the stored
# production's first 3 records with the masks equal to the negative-energy set and E0 to 7.2e-16, but E to 5.8e-13, 2.1e-12 and 4.8e-12
# against the gate's 1e-12; the builder attributes this to engine_fields_v14's series start (its fields test: 4.4e-14 to 3.7e-13 per solve
# at the same sources) and starts the production on the owner's order, the lead ruling on the gate in turn 30). Lane 1 now: the resumed
# production (from T/rs 11.0 on v19, the slices of 20.1). Lane 2: waits for version 1's lane (the replay of the 102 records, PID given),
# then the queue of version 1 behind its gate. Status lines in h100b_status.txt. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=python
S=h100b_status.txt
WAIT_PID=$1
P64="--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --kappa-sea 10 --reference R2 --site-fields averaged"
P32="--m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 1024 --ell 4 --kappa-sea 10 --reference R2 --site-fields averaged"
mkdir -p C3prod19 C3set19 C3lz states
nohup $PY code/engine_collapse_v19.py $P64 --T-over-rs 40 --out C3prod19 --gpu 0 --tag c19_rs64_A_R2_tau8_ell4_avg_r11 --resume-from C3prod/c17_rs64_A_R2_tau8_ell4_avg \
  --resume-T-over-rs 11.0 --slices-c128 12,16,20,24,28 --slices-c64 15,25,30,35,40 --slice-dir states/c19_rs64_A_R2_tau8_ell4_avg_r11 > logs_h100b_c19_rs64_prod_r11.log 2>&1 < /dev/null &
echo "production c19_rs64_A_R2_tau8_ell4_avg_r11 start $(date -u +%H:%M:%S) UTC (PID $!)" >> $S
while kill -0 $WAIT_PID 2>/dev/null; do sleep 15; done
run() {  # tag, then the command
  local tag=$1; shift
  echo "lane2 $tag start $(date -u +%H:%M:%S) UTC" >> $S
  "$@" > logs_h100b_$tag.log 2>&1
  echo "lane2 $tag exit $? $(date -u +%H:%M:%S) UTC" >> $S
}
run c19_rs32_tau8_r14625 $PY code/engine_collapse_v19.py $P32 --reading A --tau 8 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_A_R2_tau8_ell4_avg_r14625 --resume-from C3set/set17_rs32_A_R2_tau8_ell4_avg --resume-T-over-rs 14.625
run c19_rs64_cLZ0.5_r11 $PY code/engine_collapse_v19.py $P64 --T-over-rs 14 --c-LZ 0.5 --out C3lz --gpu 0 --tag c19_rs64_A_R2_tau8_ell4_avg_cLZ0.5_r11 --resume-from C3prod/c17_rs64_A_R2_tau8_ell4_avg --resume-T-over-rs 11.0
run c19_rs64_cLZ2_r11 $PY code/engine_collapse_v19.py $P64 --T-over-rs 14 --c-LZ 2 --out C3lz --gpu 0 --tag c19_rs64_A_R2_tau8_ell4_avg_cLZ2_r11 --resume-from C3prod/c17_rs64_A_R2_tau8_ell4_avg --resume-T-over-rs 11.0
run c19_rs32_tau2 $PY code/engine_collapse_v19.py $P32 --reading A --tau 2 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_A_R2_tau2_ell4_avg
run c19_rs32_B_R0x4_tau4 $PY code/engine_collapse_v19.py $P32 --reading B --tau 4 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_B_R2_tau4_ell4_R0x4_avg
run c19_rs32_n2048_tau8 $PY code/engine_collapse_v19.py --m 0.2 --g 0.25 --rs 32 --R0-over-rs 4 --nsites 2048 --ell 4 --kappa-sea 10 --reference R2 --site-fields averaged --reading A --tau 8 --T-over-rs 40 --out C3set19 --gpu 0 --tag c19_rs32_A_R2_tau8_ell4_n2048_avg
run c19_rs64_ell8 $PY code/engine_collapse_v19.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 8 --tau 8 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --out C3prod19 --gpu 0 --tag c19_rs64_A_R2_tau8_ell8_avg
echo "lane2 (version 2) DONE $(date -u +%H:%M:%S) UTC" >> $S
