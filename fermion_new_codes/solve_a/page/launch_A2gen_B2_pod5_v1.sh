#!/bin/bash
# launch_A2gen_B2_pod5_v1.sh (the solve builder, carried by the busi lane per inbox #128 and #129; turn 61b 53a.2: the 2 B2 collector runs on
# "whichever card offers room first"; turn 62 54.2 (6)). A2gen_B2 (A2's corrected-collector rerun on B2's records, page_history_v6m with
# --collector-select gen --collector-admit split --col-diag, the run PAGE version 3c needs) on pod 5's empty H100, the command of
# pod4_A2gen_job_v1.txt with pod 5's paths: the code (page/code_v6, md5 tree 5650be1bd05aa5b16ce9242f86890453, as on pod 4) and B2's records
# (json md5 417a4018..., fields cb26fcfa..., as pod 4's and /data B2_final) on pod 5's root disk, not its network volume. --resume and
# --save-every 4 make it resumable: rerun this script. Outputs pulled to /data by the box's sync_pod5_A2gen_v1.sh.
W=${QI2_ROOT}/page; ST=${QI2_ROOT}/A2gen_status.txt
cd $W || exit 1; mkdir -p runs logs
echo "A2gen_B2 launch $(date -u +%FT%TZ) gpu_used_MiB $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) load $(cut -d' ' -f1 /proc/loadavg)" >> $ST
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
D=${QI2_ROOT}/B2; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
nice -n 10 python code_v6/page_history_v6m.py --record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --device cuda:0 --out runs/v6mA2_B2_gensplit --tag v6mA2_B2_gensplit --resume --T-end-rs 100 --ledger-anchor-rs 12.125 --save-at-rs 100 --max-wall-hours 60 --record-ext $D/collapse_$T.json --fields-ext $D/collapse_${T}_fields.npz --ext-after-rs 2 --collector-select gen --collector-admit split --col-diag >> logs/v6mA2_B2_gensplit.log 2>&1
echo "A2gen_B2 exit $? $(date -u +%FT%TZ)" >> $ST
