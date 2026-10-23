#!/bin/bash
# gate_v6r_cpu.sh (busi PERF PG-A1..A3): one arm of the v6r self-test on the box CPU (no GPU free): gate_v6r_box.sh's production line at a small
# kappa, --device cpu, TH threads, nice 10, with the reference and the first probe at RREF r_s, probes every 0.5 r_s and the columns at T_end.
# Usage: bash gate_v6r_cpu.sh ARM ENGINE TH KAPPA TEND RREF [extra engine args...]
ARM=$1; ENG=$2; TH=$3; KA=$4; TEND=$5; RREF=$6; shift 6
W=${QI2_ROOT}/gate_v6r_box; C=${QI2_ROOT}/code_pod7; D=${QI2_ROOT}/B2_history; T=c19_rs64_A_R2_tau8_ell4_avg_B2n4096
PY=${QI2_ROOT}/python
mkdir -p $W/runs $W/logs; cd $W
if [ -e runs/$ARM ]; then echo "$ARM REFUSED: runs/$ARM exists" >> status.txt; exit 4; fi
export CUDA_VISIBLE_DEVICES= PYTHONPATH=$C/code_v6:$C/code_d13 PAGE_CODE_V6=$C/code_v6 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMPY_MADVISE_HUGEPAGE=0
echo "$ARM launch $(date -u +%FT%TZ) engine $ENG md5 $(md5sum $C/code_d13/$ENG | cut -c1-8) cpu threads $TH kappa $KA T_end $TEND ref $RREF extra '$*'" >> status.txt
nice -n 10 $PY $C/code_d13/$ENG --record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --kappas $KA-$KA --kappa-hard-max $KA --no-kappa-extend --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --sector-check-every-rs 10 --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs $TEND --save-every 4 --device cpu --threads $TH --out runs/$ARM --tag gate --resume --max-wall-hours 6 --columns-at-rs $TEND --columns-every-rs 0 --ref-clocks-rs $RREF --probe-from-rs $RREF --probe-every-rs 0.5 "$@" > logs/$ARM.log 2>&1
echo "$ARM exit $? $(date -u +%FT%TZ)" >> status.txt
