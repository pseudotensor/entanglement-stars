#!/bin/bash
# launch_page_v6_v1.sh (builder analyst, 2026-10-16; SPEC_final4b_engine.md 37.1 "Ruled" (2) and (4), 37.9 S1; inbox #31 S1). WRITTEN, NOT
# EXECUTED: the v6 runs launch only after reviewer A's ruling on inbox #32 is posted in the inbox and under the label it gives (37.1 (4)); this
# launcher refuses unless RULING_ENTRY (the ruling's inbox entry, e.g. "#36") and RULING_LABEL (the label it gives, printed on every run)
# are set in the queue line's environment. Never the paper's Page computation without the ruling.
# The run line (37.1 (2)): the pilot's (PILOT32 of launch_page_pilot_v5.sh: --R-cut 32 --R-collar 16 --tau-s 4, channels 1,2, nsites 8192,
# eps_rank 1e-9, eps_form 1e-7; eps_coll 1e-5) with --max-stages 1 --dE-frac 1.0 --t-stage-max 300, the cap lifted (--m-cap 16384, printed
# as m_cap_used), the four repairs (--hull exact --pair-ledger sharp --collector active --eps-loc 1e-2 --direct-every 1), S1's end rule
# (--end-rule v6: 10 consecutive blocks with |L| below 1e-3 of L_cont after S_esc below 1 percent of its peak, else 300 r_s), --store-cap
# 100000. The runs: ell4 (pod 1 lane 1), ell2 (--ell 2; pod 1 lane 5), then guard (--channels 1,2,3 at ell 4) and epsloc3 (--eps-loc 1e-3
# at ell 4). Every output carries S1's label (the code prints it). Run directories on the container disk ${QI2_ROOT}/page_v6, the small outputs
# mirrored to ${QI2_ROOT}/v6s1_<name>/, the python log ${QI2_ROOT}/v6s1_<name>.log. Gates: the ruling's variables, the
# self-test v6, du -sm /workspace below 40000 MB, NEED_GB free on the container disk (made by removing finished, mirrored v5 controls' and
# v6 runs' checkpoints and stored modes). A final run is not re-run; each run is resumable (--resume).
# Usage: RULING_ENTRY="#nn" RULING_LABEL="..." bash code_v6/launch_page_v6_v1.sh ell4 | ell2 | guard | epsloc3 | check <name>
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
S1LINE="--rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 1.0 --interior selfsimilar --tau-tr 4 --eps-coll 1e-5 --eps-rank 1e-9 --eps-form 1e-7 --eps-orth 1e-12 --m-cap 16384 --max-stages 1 --t-stage-max 300 --hull exact --pair-ledger sharp --collector active --eps-loc 1e-2 --direct-every 1 --end-rule v6 --end-flux-frac 1e-3 --end-window 10 --end-entropy-frac 0.01 --store-cap 100000 --ckpt-every 10 --max-wall-hours 6 --device cuda:0"

args_of() {
  case "$1" in
    ell4) echo "";;
    ell2) echo "--ell 2";;
    guard) echo "--channels 1,2,3";;
    epsloc3) echo "--eps-loc 1e-3";;
    *) echo "unknown run '$1' (ell4 ell2 guard epsloc3)" >&2; exit 2;;
  esac
}

now() { date -u +%H:%M:%S; }
free_gb() { mkdir -p "$BIG"; df -BG --output=avail "$BIG" | tail -1 | tr -dc '0-9'; }
ws_mb() { (du -sm /workspace 2>/dev/null || true) | tail -1 | cut -f1 | tr -dc '0-9'; }
is_final() { [ -f "$1" ] && $PY -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get('final') else 1)" "$1" 2>/dev/null; }

gate_ruling() {
  if [ -z "${RULING_ENTRY:-}" ] || [ -z "${RULING_LABEL:-}" ]; then
    echo "$(now) UTC: no ruling given (RULING_ENTRY and RULING_LABEL unset): the v6 S1 runs wait for reviewer A's ruling on inbox #32 (37.1 (4)); no launch"; exit 7
  fi
  echo "$(now) UTC: the ruling: inbox ${RULING_ENTRY}; the label it gives: ${RULING_LABEL}"
}

gate_selftest() {
  $PY -c "import json, sys; d = json.load(open('runs/selftest_v6/selftest_results.json')); sys.exit(0 if (d.get('all_pass') and not d.get('skipped') and d.get('code') == 'page_selftest_v6.py') else 3)" 2>/dev/null || {
    echo "the self-test v6 has not passed (runs/selftest_v6/selftest_results.json): no launch"; exit 3; }
  echo "self-test v6 gate: all_pass true"
}

gate_workspace() {
  local ws; ws=$(ws_mb)
  if [ -z "$ws" ] || [ "$ws" -ge "$WS_STOP_MB" ]; then echo "$(now) UTC: du -sm /workspace reads '${ws:-unknown}' MB, at or above $WS_STOP_MB MB (or unreadable): no launch"; exit 4; fi
  echo "$(now) UTC: du -sm /workspace $ws MB (stop at $WS_STOP_MB MB)"
}

make_room() {
  local f d t
  for f in $(ls -1tr "$BIGC"/ctrl32_v5_*/ctrl32_v5_*_summary.json "$BIG"/v6*_*/v6*_*_summary.json 2>/dev/null || true); do
    [ "$(free_gb)" -ge "$NEED_GB" ] && break
    d=$(dirname "$f"); t=$(basename "$d")
    case "$d" in "$BIGC"/ctrl32_v5_*|"$BIG"/v6reg_*|"$BIG"/v6s1_*) ;; *) continue;; esac
    if is_final "$f" && is_final "runs/$t/${t}_summary.json"; then
      if [ -e "$d/${t}_ckpt.pt" ] || [ -e "$d/${t}_cQ" ]; then
        echo "$(now) UTC: container disk $(free_gb) GB free, below $NEED_GB: $t is final and mirrored in runs/$t; removing its checkpoint and stored modes ($(du -sm "$d" | cut -f1) MB)"
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

line_of() { echo "$PY $C/page_recursion_v6.py $S1LINE $LIBS --out $BIG/v6s1_$1 --tag v6s1_$1 --resume $(args_of "$1")"; }

run_one() {
  local name=$1 tag=v6s1_$1 out rec rc mpid
  args_of "$name" > /dev/null
  out=$BIG/$tag; rec=runs/$tag
  mkdir -p "$out" "$rec"
  if is_final "$out/${tag}_summary.json" || is_final "$rec/${tag}_summary.json"; then echo "$(now) UTC: $tag already final: not re-run"; return 0; fi
  gate_ruling; gate_selftest; gate_workspace; gate_container
  echo "$(now) UTC: $tag start under the ruling ${RULING_ENTRY} (${RULING_LABEL}); line: $(line_of "$name")"
  echo "$(now) UTC: the ruling: inbox ${RULING_ENTRY}; the label it gives: ${RULING_LABEL}" >> "logs/$tag.log"
  rm -f "$out/.mirror_stop"
  mirror_loop "$out" "$rec" "$tag" $$ &
  mpid=$!
  set +e
  # shellcheck disable=SC2046
  $PY $C/page_recursion_v6.py $S1LINE $LIBS --out "$out" --tag "$tag" --resume $(args_of "$name") >> "logs/$tag.log" 2>&1
  rc=$?
  touch "$out/.mirror_stop"; wait "$mpid" 2>/dev/null
  set -e
  rm -f "$out/.mirror_stop"
  mirror "$out" "$rec" "$tag"
  echo "$(now) UTC: $tag python exit $rc"; tail -n 2 "logs/$tag.log" || true
  return $rc
}

case "${1:-}" in
  check) n=${2:?usage: check <name>}; args_of "$n" > /dev/null; echo "line: $(line_of "$n")"; echo "du -sm /workspace: $(ws_mb) MB; container disk free: $(free_gb) GB"; gate_selftest
         if [ -z "${RULING_ENTRY:-}" ]; then echo "the ruling: not given (a launch would refuse)"; else echo "the ruling: ${RULING_ENTRY} (${RULING_LABEL:-no label})"; fi;;
  ""|-h|--help) echo "usage: RULING_ENTRY=\"#nn\" RULING_LABEL=\"...\" bash code_v6/launch_page_v6_v1.sh ell4 | ell2 | guard | epsloc3 | check <name>"; exit 2;;
  *) run_one "$1";;
esac
