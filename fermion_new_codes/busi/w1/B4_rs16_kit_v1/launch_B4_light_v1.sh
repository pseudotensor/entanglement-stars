#!/bin/bash
# launch_B4_light_v1.sh (busi f52b1152, 20 Oct; REVIEWA_B4_rs16_v1.md): B4's light kappa 1 run on a B4 history, ready_jobs_v3 sec. 16 (a)'s line
# without --kappa-zero (launch_v6s_B2_card_v2.sh's form): page_history_v6s 611d190f with page_interior_v6s 15e08e28, kappa 1, light sites
# = 2 x the history's (2048 at r_s/a 16, 3072 at 24: the chain to 128 r_s, as B2's 8192 at 64), the cuts 2,16,24,30,48 r_s and the joins
# 32,64 r_s (in r_s, so scaled to the object: 32 ... 768 sites at r_s/a 16), the initial vacuum on the device (--init-eig-device device),
# to the history's last record. Extra checkpoints: the production's 15, 30 plus the matched clocks of the r_s/a 64 and 32 readings
# (b4_light_clocks_v1.py from the history's own first root; table in runs/<tag>/B4_matched_clocks.md).
#   MODE h16 (tag v6s_B4rs16), twin8 (v6s_B4rs16t8), f24 (v6s_B4rs24); outputs runs/<tag>/<tag>_k1_{d13.json,d13.npz,d13_tables.md,summary.json,tables.md}.
# H1 thread limits (b4_threads_v1.sh; --threads the same number). Resumes (--resume) with the same line.
# Usage: W1_PY=<python> [W1_CODE=<dir holding code_v6/ code_d13/, default the kit's page/>] [W1_RUNDIR=dir] [B4_HIST=<dir of the B4 history>]
#        [W1_MAX_WALL_HOURS=h] [B4_XCK=<override list>] [B4_DEVICE=cuda:0] bash launch_B4_light_v1.sh <card> h16|twin8|f24
export NUMPY_MADVISE_HUGEPAGE=0; CARD=$1; MODE=$2
KIT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
case $MODE in
  h16)   T=c19_rs16_A_R2_tau2_ell4_avg_B4n1024; NS=2048; TAG=v6s_B4rs16 ;;
  twin8) T=c19_rs16_A_R2_tau8_ell4_avg_B4n1024; NS=2048; TAG=v6s_B4rs16t8 ;;
  f24)   T=c19_rs24_A_R2_tau3_ell4_avg_B4n1536; NS=3072; TAG=v6s_B4rs24 ;;
  *) echo "MODE h16, twin8 or f24"; exit 2 ;;
esac
CODE=${W1_CODE:-$KIT/page}; RUNDIR=${W1_RUNDIR:-${QI2_ROOT}/page}; D=${B4_HIST:-${QI2_ROOT}/B4}; PY=${W1_PY:-python}
cd "$RUNDIR" || exit 2
mkdir -p runs/$TAG logs w1
MD5_V6S=611d190f5047d0a65dfc1c72866cc69e; MD5_IR=15e08e2807a3bce6e565306de023e80f
m1=$(md5sum $CODE/code_d13/page_history_v6s.py | cut -d' ' -f1); m2=$(md5sum $CODE/code_d13/page_interior_v6s.py | cut -d' ' -f1)
if [ "$m1" != "$MD5_V6S" ] || [ "$m2" != "$MD5_IR" ]; then echo "$TAG REFUSED on $CARD $(date -u +%FT%TZ): md5 v6s $m1 (want $MD5_V6S), interior $m2 (want $MD5_IR)" >> v6o_status.txt; exit 1; fi
J=$D/collapse_$T.json; F=$D/collapse_${T}_fields.npz
if [ ! -s "$J" ] || [ ! -s "$F" ]; then echo "$TAG REFUSED on $CARD $(date -u +%FT%TZ): the history $J or its fields are missing" >> v6o_status.txt; exit 1; fi
if [ -n "$B4_XCK" ]; then XCK=$B4_XCK; else
  XCK=$($PY -I "$KIT/b4_light_clocks_v1.py" "$J" --table runs/$TAG/B4_matched_clocks.md) || { echo "$TAG REFUSED on $CARD $(date -u +%FT%TZ): no clock origin (no root) in $J" >> v6o_status.txt; exit 1; }
fi
TEND=$($PY -I -c "import json,sys; print('%g' % json.load(open(sys.argv[1]))['records'][-1]['T_over_rs'])" "$J")
. "$KIT/b4_threads_v1.sh"; b4_threads
export PYTHONPATH=$CODE/code_v6:$CODE/code_d13 PAGE_CODE_V6=$CODE/code_v6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAXH=${W1_MAX_WALL_HOURS:-8}
ARGS="--record $J --fields $F --fields-key applied --kappas 1-1 --kappa-hard-max 1 --no-kappa-extend --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --device ${B4_DEVICE:-cuda:0} --resume --nsites $NS --init-eig-device device --sector-check-every-rs 10 --collect-every-rs 2 --extra-ck-rs 15,30,$XCK --ext-every-rs 10 --vac-every-rs 10 --T-end-rs $TEND --save-every 4 --threads $B4_NT --columns-at-rs 12,20,40 --columns-every-rs 0"
GPUMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
echo "$TAG launch on $CARD $(date -u +%FT%TZ) cap_h $MAXH threads $B4_NT ($B4_QUOTA_LINE) T_end $TEND gpu_used_MiB ${GPUMEM:-none}" >> v6o_status.txt
echo "args: $ARGS" >> logs/$TAG.log
nice -n 10 $PY $CODE/code_d13/page_history_v6s.py $ARGS --out runs/$TAG --tag $TAG --max-wall-hours $MAXH >> logs/$TAG.log 2>&1 &
echo $! > w1/$TAG.pid; wait $!; rc=$?
echo "$TAG exit $rc $(date -u +%FT%TZ)" >> v6o_status.txt
exit $rc
