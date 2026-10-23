#!/bin/bash
# launch_v6n_s8_R1_v1.sh (the busi lane as solve builder; inbox busi #168 and #162 item 6, "then page_history_v6n on the sigma/a 8 history";
# RUN_page_d13_v6n_v1.md section 4): page_history_v6n (collector-free, D13 readouts v1b inline) on the sigma/a 8 history to its end at T/r_s 40,
# kappa 1 to 16, on R1 (US-MO-1; 500 GB /workspace) while R1 waits for its ladder job (item 3 (d)); resumable with the same command (--resume).
# The history: collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg (engine_collapse_v27; 321 records 0 to 40; 2048 sites; the record HORIZON version 3
# read as hist_s8; json md5 ef0ae520..., fields a9d6472d...). Arguments as launch_v6n_B2_pod7_v2.sh (B2) with --T-end-rs 40, --out runs/v6n_s8,
# --tag v6n_s8; --nsites 8192 (the history's domain 2048 sites is below 4096, RUN section 4). Code: pod 7's code_v6 and code_d13 (tree md5s
# 09c2e665... and ad4a287b..., equal); python 3.11.10, numpy 1.26.3, scipy 1.14.1, torch 2.4.1+cu124 as pod 7's.
cd ${QI2_ROOT}/page; export PYTHONPATH=${QI2_ROOT}/code_d13 PAGE_CODE_V6=${QI2_ROOT}/code_v6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=${QI2_ROOT}/s8; T=c27_rs64_A_R2_tau8_ell8_prep4_avg
echo "v6n_s8 launch $(date -u +%FT%TZ) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) load $(cut -d' ' -f1 /proc/loadavg)" >> ${QI2_ROOT}/v6n_status.txt
nice -n 10 python code_d13/page_history_v6n.py --record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --kappas 1-16 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs 40 --save-every 4 --device cuda:0 --threads 4 --out runs/v6n_s8 --tag v6n_s8 --resume --max-wall-hours 60 --columns-at-rs 12,20,40 --columns-every-rs 10 >> logs/v6n_s8.log 2>&1
echo "v6n_s8 exit $? $(date -u +%FT%TZ)" >> ${QI2_ROOT}/v6n_status.txt
