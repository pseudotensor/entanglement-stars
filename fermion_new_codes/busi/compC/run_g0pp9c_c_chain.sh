#!/bin/bash
# REVIEWA_C64_toy_readout_v1 condition 5 on the box: run_g0pp9c1c.sh (CPU), then run_g0pp9c2c.sh and run_g0pp9c3c.sh side by side (CPU); the GPU half
# (run_g0pp9c1c_gpu.sh) by gpu_wait_g0pp9c1c.sh, started beside this chain. Done markers and logs in ${QI2_ROOT}/toy_late_step_v1.
O=${QI2_ROOT}/toy_late_step_v1
C=${QI2_ROOT}/compC
bash $C/v9c1c/run_g0pp9c1c.sh > $O/g0pp9c1c.log 2>&1; echo "rc $? $(date -u +%FT%TZ)" > $O/g0pp9c1c.done
( bash $C/v9c2c/run_g0pp9c2c.sh > $O/g0pp9c2c.log 2>&1; echo "rc $? $(date -u +%FT%TZ)" > $O/g0pp9c2c.done ) &
( bash $C/v9c3c/run_g0pp9c3c.sh > $O/g0pp9c3c.log 2>&1; echo "rc $? $(date -u +%FT%TZ)" > $O/g0pp9c3c.done ) &
wait
echo "chain done $(date -u +%FT%TZ)" > $O/g0pp9c_c_chain.done
