#!/bin/bash
# run.sh, kit b_page_v6q_k19 (solve bench_gputypes v1): one Page channel, kappa 19, on page_history_v6q.py (b4f4ea69) with the B2 production line of
# launch_v6q_B2_card_v1.sh (677fd00a) from T = 0 to T/r_s END (default 10 = 80 records of tau 8 at r_s 64; the T 10 checkpoint is a dense one),
# token for token except: --kappas 19-19 --kappa-hard-max 19, --T-end-rs END, --threads NT (the line's 4), --max-wall-hours 3, the out and tag.
# Inputs: the B2 record json 417a4018 and fields cb26fcfa (box ${QI2_ROOT}/B2_history), code_d13 and code_v6 (R2's card copies, md5-equal to the box's).
# Usage: bash run.sh GPU [NT] [END]
KIT=$(cd $(dirname $0) && pwd); CLASS=b_page_v6q_k19; ENVNAME=np1263; STACK="1.26.3 1.14.1 2.4.1+cu124"
. $KIT/bench_lib.sh
END=${3:-10}
T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096; TAG=bench_k19
export PYTHONPATH=$KIT/code_v6:$KIT/code_d13 PAGE_CODE_V6=$KIT/code_v6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$CLASS END $END NT $NT" > $D/kit.txt; md5sum $KIT/code_d13/page_history_v6q.py >> $D/kit.txt
mkdir -p $D/runs; cd $D && nice -n 10 $PY $KIT/bench_exec.py --gpu $GPU --out $D --log $D/log.txt -- $PY $KIT/code_d13/page_history_v6q.py --record $KIT/input/collapse_$T.json --fields $KIT/input/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --kappas 19-19 --kappa-hard-max 19 --no-kappa-extend --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --sector-check-every-rs 10 --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs $END --save-every 4 --device cuda:0 --threads $NT --out runs/$TAG --tag $TAG --resume --max-wall-hours 3 --columns-at-rs 12,20,40 --columns-every-rs 0
rc=$?; $PY $KIT/bench_cmp.py summary b $D; echo "run dir $D rc $rc"; exit $rc
