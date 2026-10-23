#!/bin/bash
# vf7's R2 column (r_s 32, kappa <= 16, the two-step current, the aware R2) reported when the GPU 2 run ends (the input of red team R26c,
# spec 18.2: "R26c, 18.2 with the decisive rerun's record and vf7's R2 column"): engine_vacflow_report_v6 with the keys R2 and the per-channel
# table, on the CPU. Usage (on the box, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_vf7_report_v1.sh > VF/vf7_report_launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
until grep -q "^GPU2 vf7_rs32_exact_m0_R2aware exit" status.txt; do sleep 60; done
J=VF/vacflow_vf7_rs32_exact_m0_R2aware.json
[ -f $J ] || { echo "no $J"; exit 1; }
OMP_NUM_THREADS=2 nice -n 10 ${QI2_ROOT}/python code/engine_vacflow_report_v6.py $J --keys raw,NO,R2 --md VF/vacflow_report_v6_vf7_rs32_R2.md > VF/vacflow_report_v6_vf7_rs32_R2.log 2>&1
echo "vf7 report exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
