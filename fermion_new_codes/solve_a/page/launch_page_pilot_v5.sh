#!/bin/bash
# launch_page_pilot_v5.sh. Changes against launch_page_pilot_v4.sh (builder analyst; SPEC_final4b_engine.md 33.1 "Ruled" (6) and (7), 33.6
# BUILDER item 1): every command runs code_v5/page_recursion_v5.py (the v5 code in ${QI2_ROOT}/code_v5, the outputs in
# ${QI2_ROOT}/runs, the logs in ${QI2_ROOT}/logs); every folder and tag carries _v5, so nothing overwrites or resumes a v1 to v4 run;
# the gate reads runs/selftest_v5/selftest_results.json (the self-test v5 runs on the box CPU and its results file is copied there); (33.1 (7))
# the r_s/a 32 pilot's line is the v4 report's with --max-wall-hours 40 and nothing else changed (RUN32), labelled "on the exact exterior
# sequence" by the code; (33.1 (6)) the r_s/a 64 production's line is the same with --rs 64 --nsites 16384 --ckpt-every 5 --max-wall-hours 60
# (RUN64; branches rs64 and rs64-fg, for the second H100 when 33.1 (6)'s condition holds: 4 times the pilot's first-transition peak below
# 64 GB; this launcher does not check it, launch_page_rs64_p2_v5.sh does). This is a new file: a launcher is never edited while a process
# executes it (the v4 report's defect); a changed command is a new launcher file.
# launch_page_pilot_v4.sh (builder analyst; 31.4 (9)): the gate (R11), pilot-fg and rs64-fg for the pods' queue runners, timing with
# --timing-transition 1, SCAT_TAG. launch_page_pilot_v3.sh: code_v3, "timing". launch_page_pilot_v1.sh (lead reviewer A): GAP03 section 12.
# Usage: bash code_v5/launch_page_pilot_v5.sh selftest | scattering | timing | pilot | pilot-fg | controls | rs64 | rs64-fg | status TAG
set -euo pipefail
cd ${QI2_ROOT}/page
mkdir -p runs logs
PY=${PY:-python}
C=code_v5
LIBS="--engine-path ${QI2_ROOT}/code --obs-path ${QI2_ROOT}/code"
PILOT32="--rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4"
PILOT64="--rs 64 --g 0.25 --m 0 --nsites 16384 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4"
RUN32="--max-stages 200 --E-stop-frac 0.5 --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 10 --max-wall-hours 40 --device cuda:0"
RUN64="--max-stages 200 --E-stop-frac 0.5 --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 5 --max-wall-hours 60 --device cuda:0"

gate_selftest() {
  # R11: a launch depends programmatically on a passing self-test (v5: T1 to T5 with T4f, T4f-roundoff and T4k)
  $PY -c "import json, sys; d = json.load(open('runs/selftest_v5/selftest_results.json')); sys.exit(0 if (d.get('all_pass') and not d.get('skipped') and d.get('code') == 'page_selftest_v5.py') else 3)" 2>/dev/null || {
    echo "the self-test v5 has not passed (runs/selftest_v5/selftest_results.json): no launch (R11)"; exit 3; }
  echo "self-test v5 gate: all_pass true ($($PY -c "import json; d = json.load(open('runs/selftest_v5/selftest_results.json')); print(d.get('device'), d.get('code'), round(d.get('seconds', 0)), 's')"))"
}

status() {
  local tag=$1
  $PY -c "import json, sys; s = json.load(open('runs/$tag/${tag}_summary.json')); v = s.get('validation') or {}; print('$tag validation status:', v.get('status')); print('not evaluated:', '; '.join(v.get('not_evaluated', []))); dv = s.get('device') or {}; print('device peaks per transition (GB, 33.1 (6)):', [t.get('device_peak_GB') for t in dv.get('transitions', [])], '; the run peak', dv.get('gpu_peak_run_GB'))" || echo "$tag: no summary yet"
}

case "${1:-}" in
  selftest)
    $PY $C/page_recursion_v5.py --selftest --device ${DEV:-cuda:0} $LIBS --out runs/selftest_v5_pod 2>&1 | tee logs/selftest_v5_pod.log
    ;;
  scattering)
    T=${SCAT_TAG:-scat32_v5}
    $PY $C/page_recursion_v5.py --validate-scattering --rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2,3 --ell 4 --device cuda:0 $LIBS --out runs/$T --tag $T 2>&1 | tee logs/$T.log
    ;;
  timing)
    $PY $C/page_recursion_v5.py $PILOT32 --timing-blocks 3 --timing-transition 1 --tau-tr-doublings 0 --device cuda:0 $LIBS --out runs/timing32_v5 --tag timing32_v5 2>&1 | tee logs/timing32_v5.log
    status timing32_v5
    ;;
  pilot)
    gate_selftest
    nohup $PY $C/page_recursion_v5.py $PILOT32 $RUN32 $LIBS --out runs/pilot32_v5 --tag pilot32_v5 --resume > logs/pilot32_v5.log 2>&1 &
    echo "pilot launched: tail -f logs/pilot32_v5.log; status: bash $C/launch_page_pilot_v5.sh status pilot32_v5"
    ;;
  pilot-fg)
    gate_selftest
    $PY $C/page_recursion_v5.py $PILOT32 $RUN32 $LIBS --out runs/pilot32_v5 --tag pilot32_v5 --resume 2>&1 | tee -a logs/pilot32_v5.log
    status pilot32_v5
    ;;
  controls)
    gate_selftest
    # the pilot's controls (run after the pilot's first stage is read; sequentially on the same GPU, each resumable)
    for spec in "ell8:--ell 8" "point:--fields point" "Rb08:--Rb 0.8" "Rb0_8:--Rb0 8" "ramp:--formation ramp" "box:--nsites 16384" "cut48:--R-cut 48" "taus2:--tau-s 2" "kappa3:--channels 1,2,3" "coll4:--eps-coll 1e-4" "coll6:--eps-coll 1e-6"; do
      tag=${spec%%:*}; args=${spec#*:}
      $PY $C/page_recursion_v5.py $PILOT32 --max-stages 1 --t-stage-max 1000 \
        --ckpt-every 10 --max-wall-hours 6 --device cuda:0 $LIBS --out runs/ctrl32_v5_$tag --tag ctrl32_v5_$tag --resume $args 2>&1 | tee logs/ctrl32_v5_$tag.log
      status ctrl32_v5_$tag
    done
    ;;
  rs64)
    gate_selftest
    nohup $PY $C/page_recursion_v5.py $PILOT64 $RUN64 $LIBS --out runs/fig64_v5 --tag fig64_v5 --resume > logs/fig64_v5.log 2>&1 &
    echo "r_s/a 64 launched: tail -f logs/fig64_v5.log"
    ;;
  rs64-fg)
    gate_selftest
    $PY $C/page_recursion_v5.py $PILOT64 $RUN64 $LIBS --out runs/fig64_v5 --tag fig64_v5 --resume 2>&1 | tee -a logs/fig64_v5.log
    status fig64_v5
    ;;
  status)
    status "${2:?usage: status TAG}"
    ;;
  *)
    echo "usage: bash code_v5/launch_page_pilot_v5.sh selftest | scattering | timing | pilot | pilot-fg | controls | rs64 | rs64-fg | status TAG"; exit 2;;
esac
