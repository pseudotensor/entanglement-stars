#!/bin/bash
# launch_B4_history_v1.sh (busi f52b1152, 20 Oct; REVIEWA_B4_rs16_v1.md): B4's collapse histories on engine_collapse_v19 (9e4e6abf), the kit's
# own engine and imports only (PYTHONPATH = the kit's engine/), B2's argument list (417a4018) with the ruled substitutions (ARGS_<mode>.txt,
# made by b4_args_v1.py; diffs in ARGS_diff.txt):
#   h16   r_s/a 16, 1024 sites, tau 2 (0.125 r_s), to 100 r_s, tag c19_rs16_A_R2_tau2_ell4_avg_B4n1024; GATED at 32 r_s
#   twin8 the same with tau 8 (0.5 r_s), tag c19_rs16_A_R2_tau8_ell4_avg_B4n1024; not gated (run after h16 passes)
#   f24   r_s/a 24, 1536 sites, tau 3 (0.125 r_s), to 100 r_s, tag c19_rs24_A_R2_tau3_ell4_avg_B4n1536; GATED at 32 r_s (only after h16's gate FAILS)
# The gate (b4_gate_v1.py beside the engine): R_t non-null at any record to T/r_s 32 -> continue untouched; null throughout -> the engine is
# stopped by its own pid, <tag>_gate.md/json carry the min(N - |v|) trace with its floor and flat clock, and this script exits 3.
# Exit: the engine's rc (0 at the end), 3 = gate FAIL, 1 = refused (md5 or disk), 2 = usage.
# Outputs under $B4_RUNDIR (default ${QI2_ROOT}/engine): B4/collapse_<tag>.json, B4/collapse_<tag>_fields.npz (N_applied, v_applied in
# B2's layout), B4/B4_<tag>_gate.{json,md}, states/<tag>/ (B2's slice clocks, complex128), logs/B4_<tag>.log, B4_status.txt.
# H1 thread limits: b4_threads_v1.sh (the quota minus 2, read now). Usage:
#   CUDA_VISIBLE_DEVICES=<i> B4_PY=<python> [B4_RUNDIR=dir] [B4_MIN_FREE_GB=n] bash launch_B4_history_v1.sh <card> h16|twin8|f24
#   dry check only: B4_CPU=1 B4_DRY_T=<T/r_s> (the CPU route, --gpu dropped, the end clock replaced)
CARD=$1; MODE=$2
KIT=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
case $MODE in
  h16)   TAG=c19_rs16_A_R2_tau2_ell4_avg_B4n1024; GATE=1; NEED=${B4_MIN_FREE_GB:-30} ;;
  twin8) TAG=c19_rs16_A_R2_tau8_ell4_avg_B4n1024; GATE=0; NEED=${B4_MIN_FREE_GB:-30} ;;
  f24)   TAG=c19_rs24_A_R2_tau3_ell4_avg_B4n1536; GATE=1; NEED=${B4_MIN_FREE_GB:-60} ;;
  *) echo "usage: launch_B4_history_v1.sh <card> h16|twin8|f24"; exit 2 ;;
esac
RUNDIR=${B4_RUNDIR:-${QI2_ROOT}/engine}; PY=${B4_PY:-python}
mkdir -p "$RUNDIR" && cd "$RUNDIR" || exit 2
mkdir -p B4 logs w1
ST=$RUNDIR/B4_status.txt
if ! (cd "$KIT" && md5sum -c --quiet MD5SUMS > /dev/null 2>&1); then echo "B4_$MODE REFUSED on $CARD $(date -u +%FT%TZ): the kit's MD5SUMS do not verify in $KIT" >> "$ST"; exit 1; fi
FREE=$(df -BG --output=avail "$RUNDIR" | tail -1 | tr -dc 0-9)
if [ -z "$B4_DRY_T" ] && [ "${FREE:-0}" -lt "$NEED" ]; then echo "B4_$MODE REFUSED on $CARD $(date -u +%FT%TZ): ${FREE} GB free in $RUNDIR, the slices need about $NEED GB" >> "$ST"; exit 1; fi
ARGS=$(cat "$KIT/ARGS_$MODE.txt")
if [ -n "$B4_CPU" ]; then ARGS=$(echo "$ARGS" | sed 's/ --gpu 0 / /'); fi
if [ -n "$B4_DRY_T" ]; then ARGS=$(echo "$ARGS" | sed "s/ --T-over-rs 100 / --T-over-rs $B4_DRY_T /"); fi
. "$KIT/b4_threads_v1.sh"; b4_threads
export NUMPY_MADVISE_HUGEPAGE=0 PYTHONNOUSERSITE=1 PYTHONPATH=$KIT/engine PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')
echo "B4_$MODE launch on $CARD $(date -u +%FT%TZ) tag $TAG threads $B4_NT ($B4_QUOTA_LINE) gpu_used_MiB ${GPUMEM:-none} free_GB $FREE${B4_DRY_T:+ DRY T/r_s $B4_DRY_T}${B4_CPU:+ CPU}" >> "$ST"
echo "args: $ARGS" >> logs/B4_$TAG.log
nice -n 10 $PY -s "$KIT/engine/engine_collapse_v19.py" $ARGS >> logs/B4_$TAG.log 2>&1 &
PID=$!; echo $PID > w1/B4_$TAG.pid
GPID=""
if [ "$GATE" = 1 ]; then
  nice -n 10 $PY -I "$KIT/b4_gate_v1.py" --json "$RUNDIR/B4/collapse_$TAG.json" --pid $PID --tag $TAG --prefix "$RUNDIR/B4/B4_$TAG" --gate-rs 32 >> logs/B4_${TAG}_gate.log 2>&1 &
  GPID=$!; echo $GPID > w1/B4_${TAG}_gate.pid
fi
wait $PID; rc=$?
grc=""
if [ -n "$GPID" ]; then wait $GPID; grc=$?; fi
if [ "$grc" = 3 ]; then
  echo "B4_$MODE GATE FAIL on $CARD $(date -u +%FT%TZ): no root to T/r_s 32; engine pid $PID stopped (rc $rc); trace B4/B4_${TAG}_gate.md" >> "$ST"; exit 3
fi
echo "B4_$MODE exit $rc on $CARD $(date -u +%FT%TZ) gate ${grc:-none} ($( [ "$grc" = 0 ] && echo PASS || echo 'no decision' ))" >> "$ST"
exit $rc
