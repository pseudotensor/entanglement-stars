#!/bin/bash
# gate_v6r_box.sh (busi lane PERF PG-A1..A3): one arm of the v6r self-test on the box: the B2 production line (launch_v6q_B2_card_v1.sh args)
# at a small kappa and a short T range, with the reference at 1 r_s, probes at 1, 1.5, 2 r_s and columns at 1 and 2 r_s, so that every PG-A3
# consumer (probes, saved columns) runs. One process on one GPU (GPU index $GPU, checked free of other compute first), nice 10, 4 threads.
# Usage: bash gate_v6r_box.sh ARM ENGINE GPU KAPPA TEND [extra engine args...]
ARM=$1; ENG=$2; GPU=$3; KA=$4; TEND=$5; shift 5
W=${QI2_ROOT}/gate_v6r_box; C=${QI2_ROOT}/code_pod7; D=${QI2_ROOT}/B2_history; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
PY=${QI2_ROOT}/python
mkdir -p $W/runs $W/logs; cd $W
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep "$UUID" | grep -v litellm | grep -v python3.11)
if [ -n "$BUSY" ]; then echo "$ARM REFUSED $(date -u +%FT%TZ): GPU $GPU busy: $BUSY" >> status.txt; exit 3; fi
if [ -e runs/$ARM ]; then echo "$ARM REFUSED: runs/$ARM exists" >> status.txt; exit 4; fi
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$C/code_v6:$C/code_d13 PAGE_CODE_V6=$C/code_v6 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NUMPY_MADVISE_HUGEPAGE=0
echo "$ARM launch $(date -u +%FT%TZ) engine $ENG md5 $(md5sum $C/code_d13/$ENG | cut -c1-8) gpu $GPU kappa $KA T_end $TEND extra '$*'" >> status.txt
nice -n 10 $PY $C/code_d13/$ENG --record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --kappas $KA-$KA --kappa-hard-max $KA --no-kappa-extend --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --sector-check-every-rs 10 --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs $TEND --save-every 4 --device cuda:0 --threads 4 --out runs/$ARM --tag gate --resume --max-wall-hours 4 --columns-at-rs 2 --columns-every-rs 0 --ref-clocks-rs 1 --probe-from-rs 1 --probe-every-rs 0.5 "$@" > logs/$ARM.log 2>&1
echo "$ARM exit $? $(date -u +%FT%TZ)" >> status.txt
