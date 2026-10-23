#!/bin/bash
# launch_page_v6_regression_v2.sh (builder analyst, 2026-10-16): v1 on page_recursion_v6b.py (write_md prints a None check as "-": v6
# raised TypeError there in v5 mode, so R0, R0s and R1 of v1 exited at their first output write at block 10, t 40 r_s, with their state
# saved; inbox #41 D15: no mutual information computed or printed, the turnover quantity S(M) - S(M')). The same run directories and
# tags (v6reg_<name>): each variant resumes from its own block-10 checkpoint (--resume; the state code of v6b is v6's); a final run is
# not re-run (R3, exit 0 under v1). The gate is the self-test v6b (runs/selftest_v6b/selftest_results.json, code page_selftest_v6b.py).
# v1's text: launch_page_v6_regression_v1.sh (builder analyst, 2026-10-16; SPEC_final4b_engine.md 37.1 "Ruled" (3), the regression of page_*_v6
# against the uncapped v5 baseline control, every change attributed to a named repair). VALIDATION, NOT THE PAPER'S PAGE COMPUTATION
# (inbox #32: reviewer A's ruling pending; the S1 runs are launch_page_v6_v1.sh's, gated on the ruling). One variant per queue line on pod 1 lane 1:
#   R0   v6 with every repair off (--hull v5 --pair-ledger v5 --collector v5 --direct-every 5 --end-rule v5), the baseline control's line
#        (the pilot's, --dE-frac 1.0 --max-stages 1 --t-stage-max 400 --m-cap 16384), from the formation; it must reproduce the baseline;
#        it keeps its block-0 checkpoint (--keep-start-ckpt) for the others;
#   R0s  the same settings from R0's block-0 checkpoint (the shared formation): the resume's own effect, so the others compare with it;
#   R1   repair 1 alone (--hull exact), from R0's block-0 checkpoint;
#   R3   repairs 1 to 4 (--hull exact --pair-ledger sharp --collector active --eps-loc 1e-2 --direct-every 1), from R0's block-0 checkpoint,
#        still with v5's stall rule and t_stage_max 400 so that its curve meets the baseline's clocks (S1's end rule is the S1 lines').
# Repair 2 (the ledger) changes only the band; R0's formation record prints both ledgers from the stored sines. Run directories on the
# container disk ${QI2_ROOT}/page_v6 (R0 holds GB like a control; the active collector's checkpoints are small), the small outputs mirrored to
# ${QI2_ROOT}/v6reg_<name>/, the python log ${QI2_ROOT}/v6reg_<name>.log. Gates: the self-test v6 (runs/selftest_v6/
# selftest_results.json, all_pass and code page_selftest_v6.py), du -sm /workspace below 40000 MB, NEED_GB free on the container disk (made
# by removing the checkpoint and stored modes of finished, mirrored v5 controls and v6 runs, oldest first), R0's start checkpoint for the
# others. A final run is not re-run.
# Usage: bash code_v6/launch_page_v6_regression_v2.sh R0 | R0s | R1 | R3 | check <name>
set -euo pipefail
cd ${QI2_ROOT}/page
mkdir -p runs logs
PY=${PY:-python}
C=code_v6
BIG=${QI2_ROOT}/page_v6
BIGC=${QI2_ROOT}/page_ctrl_v5
NEED_GB=8
WS_STOP_MB=40000
LIBS="--engine-path ${QI2_ROOT}/code --obs-path ${QI2_ROOT}/code"
BASE="--rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 1.0 --interior selfsimilar --tau-tr 4 --m-cap 16384 --max-stages 1 --E-stop-frac 0.5 --t-stage-max 400 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 10 --max-wall-hours 6 --store-cap 6000 --device cuda:0"
V5MODE="--hull v5 --pair-ledger v5 --collector v5 --direct-every 5 --end-rule v5"
START=$BIG/v6reg_R0/v6reg_R0_ckpt_start.pt

args_of() {
  case "$1" in
    R0) echo "$V5MODE --keep-start-ckpt";;
    R0s) echo "$V5MODE --start-ckpt $START";;
    R1) echo "--hull exact --pair-ledger v5 --collector v5 --direct-every 5 --end-rule v5 --start-ckpt $START";;
    R3) echo "--hull exact --pair-ledger sharp --collector active --eps-loc 1e-2 --direct-every 1 --end-rule v5 --store-cap 100000 --start-ckpt $START";;
    *) echo "unknown variant '$1' (R0 R0s R1 R3)" >&2; exit 2;;
  esac
}

now() { date -u +%H:%M:%S; }
free_gb() { mkdir -p "$BIG"; df -BG --output=avail "$BIG" | tail -1 | tr -dc '0-9'; }
ws_mb() { (du -sm /workspace 2>/dev/null || true) | tail -1 | cut -f1 | tr -dc '0-9'; }
is_final() { [ -f "$1" ] && $PY -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('final') else 1)" "$1" 2>/dev/null; }

gate_selftest() {
  $PY -c "import json, sys; d = json.load(open('runs/selftest_v6b/selftest_results.json')); sys.exit(0 if (d.get('all_pass') and not d.get('skipped') and d.get('code') == 'page_selftest_v6b.py') else 3)" 2>/dev/null || {
    echo "the self-test v6b has not passed (runs/selftest_v6b/selftest_results.json): no launch"; exit 3; }
  echo "self-test v6b gate: all_pass true"
}

gate_workspace() {
  local ws; ws=$(ws_mb)
  if [ -z "$ws" ] || [ "$ws" -ge "$WS_STOP_MB" ]; then echo "$(now) UTC: du -sm /workspace reads '${ws:-unknown}' MB, at or above $WS_STOP_MB MB (or unreadable): no launch"; exit 4; fi
  echo "$(now) UTC: du -sm /workspace $ws MB (stop at $WS_STOP_MB MB)"
}

make_room() {
  local f d t
  for f in $(ls -1tr "$BIGC"/ctrl32_v5_*/ctrl32_v5_*_summary.json "$BIG"/v6reg_*/v6reg_*_summary.json 2>/dev/null || true); do
    [ "$(free_gb)" -ge "$NEED_GB" ] && break
    d=$(dirname "$f"); t=$(basename "$d")
    case "$d" in "$BIGC"/ctrl32_v5_*|"$BIG"/v6reg_*) ;; *) continue;; esac
    if is_final "$f" && is_final "runs/$t/${t}_summary.json"; then
      if [ -e "$d/${t}_ckpt.pt" ] || [ -e "$d/${t}_cQ" ]; then
        echo "$(now) UTC: container disk $(free_gb) GB free, below $NEED_GB: $t is final and mirrored in runs/$t; removing its checkpoint and stored modes ($(du -sm "$d" | cut -f1) MB; R0's start checkpoint is kept)"
        rm -f "$d/${t}_ckpt.pt" "$d/${t}_ckpt.pt.tmp"
        rm -rf "${d:?}/${t}_cQ"
      fi
    fi
  done
}

gate_container() {
  make_room
  local fg; fg=$(free_gb)
  if [ -z "$fg" ] || [ "$fg" -lt "$NEED_GB" ]; then echo "$(now) UTC: the container disk has ${fg:-unknown} GB free below $NEED_GB GB: no launch"; exit 5; fi
  echo "$(now) UTC: container disk $fg GB free (need $NEED_GB)"
}

mirror() {
  local out=$1 rec=$2 tag=$3 f
  for f in "${tag}_summary.json" "${tag}_tables.md" "${tag}_curve.npz"; do
    if [ -f "$out/$f" ]; then { cp "$out/$f" "$rec/$f.tmp" && mv -f "$rec/$f.tmp" "$rec/$f"; } || true; fi
  done
}

mirror_loop() {
  local out=$1 rec=$2 tag=$3 parent=$4 i=0
  while kill -0 "$parent" 2>/dev/null && [ ! -f "$out/.mirror_stop" ]; do
    sleep 15; i=$((i + 1))
    if [ $((i % 8)) -eq 0 ]; then mirror "$out" "$rec" "$tag"; fi
  done
}

line_of() { echo "$PY $C/page_recursion_v6b.py $BASE $LIBS --out $BIG/v6reg_$1 --tag v6reg_$1 --resume $(args_of "$1")"; }

run_one() {
  local name=$1 tag=v6reg_$1 out rec rc mpid
  args_of "$name" > /dev/null
  out=$BIG/$tag; rec=runs/$tag
  mkdir -p "$out" "$rec"
  if is_final "$out/${tag}_summary.json" || is_final "$rec/${tag}_summary.json"; then echo "$(now) UTC: $tag already final: not re-run"; return 0; fi
  gate_selftest; gate_workspace; gate_container
  if [ "$name" != "R0" ] && [ ! -f "$START" ] && [ ! -f "$out/${tag}_ckpt.pt" ]; then echo "$(now) UTC: R0's block-0 checkpoint $START is missing: no launch"; exit 6; fi
  echo "$(now) UTC: $tag start (VALIDATION: the regression of 37.1 (3) on page_recursion_v6b.py, resumed from its own checkpoint; not the paper's Page computation); line: $(line_of "$name")"
  rm -f "$out/.mirror_stop"
  mirror_loop "$out" "$rec" "$tag" $$ &
  mpid=$!
  set +e
  # shellcheck disable=SC2046
  $PY $C/page_recursion_v6b.py $BASE $LIBS --out "$out" --tag "$tag" --resume $(args_of "$name") >> "logs/$tag.log" 2>&1
  rc=$?
  touch "$out/.mirror_stop"; wait "$mpid" 2>/dev/null
  set -e
  rm -f "$out/.mirror_stop"
  mirror "$out" "$rec" "$tag"
  echo "$(now) UTC: $tag python exit $rc"; tail -n 2 "logs/$tag.log" || true
  return $rc
}

case "${1:-}" in
  check) n=${2:?usage: check <name>}; args_of "$n" > /dev/null; echo "line: $(line_of "$n")"; echo "du -sm /workspace: $(ws_mb) MB; container disk free: $(free_gb) GB"; gate_selftest;;
  ""|-h|--help) echo "usage: bash code_v6/launch_page_v6_regression_v2.sh R0 | R0s | R1 | R3 | check <name>"; exit 2;;
  *) run_one "$1";;
esac
