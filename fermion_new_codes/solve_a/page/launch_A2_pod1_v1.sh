#!/bin/bash
# launch_A2_pod1_v1.sh (the lane's analyst builder; engine spec 42.2 (1)): A2, Computation A (page_history_v6i, A's settings) from T = 0 on B2's
# records (the doubled chain, 4096 engine sites, synced from pod 4 by sync_B2_to_pod1_v1.sh on the box), beside the jvp member on pod 1's H100;
# the chain 8192 sites (128 r_s), the join check over 32 to 64 r_s (the engine's domain ends at 64 r_s), --follow on B2's json to T/r_s 100;
# nice 10, 4 CPU threads while the capture minimum runs. Read check passed first (b2_read_check_v1.py). Usage (pod 1): setsid nohup bash launch_A2_pod1_v1.sh &
cd ${QI2_ROOT}/page || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
D=${QI2_ROOT}/B2; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
mkdir -p ${QI2_ROOT}/v6iA2_B2 logs
echo "A2 start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> logs/v6iA2_B2.status
nice -n 10 python code_v6/page_history_v6i.py --record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --channels 1,2 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --eps-loc 1e-2 --eps-form 1e-7 --eps-orth 1e-12 --eps-coll 1e-5 --pair-ledger sharp --collector-mode velocity --extra-ck-rs 15,30 --transfer-at-peak --ext-every-rs 10 --save-every 4 --device cuda:0 --out ${QI2_ROOT}/v6iA2_B2 --tag v6iA2_B2 --resume --T-end-rs 100 --ledger-anchor-rs 12.125 --save-at-rs 100 --max-wall-hours 60 --record-ext $D/collapse_$T.json --fields-ext $D/collapse_${T}_fields.npz --ext-after-rs 2 --follow --follow-poll-s 60 --follow-timeout-h 6 --follow-done-marker $D/EXT_DONE_$T.txt >> logs/v6iA2_B2.log 2>&1
echo "A2 exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> logs/v6iA2_B2.status
