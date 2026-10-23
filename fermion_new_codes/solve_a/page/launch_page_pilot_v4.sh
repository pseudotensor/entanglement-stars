#!/bin/bash
# launch_page_pilot_v4.sh. Changes against launch_page_pilot_v3.sh (builder analyst; SPEC_final4b_engine.md 31.4 (9) and 31.6 BUILDER item 1;
# the GAP03 review review_GAP03_page_curve.tex, R11): every command runs code_v4/page_recursion_v4.py (the v4 code in ${QI2_ROOT}/code_v4
# on pod 1, the outputs in ${QI2_ROOT}/runs, the logs in ${QI2_ROOT}/logs); every folder and tag carries _v4, so nothing overwrites or
# resumes a v1 to v3 run; --engine-path ${QI2_ROOT}/code (T1a and the transition's engine_ops_v3) and --obs-path ${QI2_ROOT}/code
# (the stationary scattering's libraries, 31.4 (8)) are passed. (R11) the pilot, controls and rs64 branches launch only when the self-test's
# results file runs/selftest_v4/selftest_results.json says all_pass true (gate_selftest; the self-test runs on the box CPU, 31.4 (9), and its
# results file and log are copied there), and each branch prints the run's validation status from its summary when it ends (the status
# branch prints it for any tag). New branches: pilot-fg and rs64-fg (in the foreground, for the pod's queue runner: one queue line
# "tag<TAB>cd ${QI2_ROOT}/page && bash code_v4/launch_page_pilot_v4.sh pilot-fg"); timing adds --timing-transition 1 (31.4 (1): the full-sea
# transition timed first; --tau-tr-doublings 0 there, so the timing measures one transition); SCAT_TAG names the scattering run's folder
# and tag (default scat32_v4).
# launch_page_pilot_v3.sh. Changes against launch_page_pilot_v2.sh (builder analyst, BRIEF_page_v3_builder.md; SPEC_final4b_engine.md 30.1):
# every command runs code_v3/page_recursion_v3.py; every output folder and tag carries _v3; new target "timing" (labelled "timing, not the pilot").
# launch_page_pilot_v1.sh (lead reviewer A, GAP03 integration): the exact commands of GAP03_integrated_v1.md section 12.
# Usage: bash code_v4/launch_page_pilot_v4.sh selftest | scattering | timing | pilot | pilot-fg | controls | rs64 | rs64-fg | status TAG
set -euo pipefail
cd ${QI2_ROOT}/page
mkdir -p runs logs
PY=${PY:-python}
C=code_v4
LIBS="--engine-path ${QI2_ROOT}/code --obs-path ${QI2_ROOT}/code"
PILOT32="--rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4"
PILOT64="--rs 64 --g 0.25 --m 0 --nsites 16384 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4"
RUN32="--max-stages 200 --E-stop-frac 0.5 --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 10 --max-wall-hours 20 --device cuda:0"
RUN64="--max-stages 200 --E-stop-frac 0.5 --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 5 --max-wall-hours 60 --device cuda:0"

gate_selftest() {
  # R11: a launch depends programmatically on a passing self-test
  $PY -c "import json, sys; d = json.load(open('runs/selftest_v4/selftest_results.json')); sys.exit(0 if (d.get('all_pass') and not d.get('skipped')) else 3)" 2>/dev/null || {
    echo "the self-test v4 has not passed (runs/selftest_v4/selftest_results.json): no launch (R11)"; exit 3; }
  echo "self-test v4 gate: all_pass true ($(python -c "import json; d = json.load(open('runs/selftest_v4/selftest_results.json')); print(d.get('device'), d.get('code'), round(d.get('seconds', 0)), 's')"))"
}

status() {
  local tag=$1
  $PY -c "import json, sys; s = json.load(open('runs/$tag/${tag}_summary.json')); v = s.get('validation') or {}; print('$tag validation status:', v.get('status')); print('not evaluated:', '; '.join(v.get('not_evaluated', [])))" || echo "$tag: no summary yet"
}

case "${1:-}" in
  selftest)
    $PY $C/page_recursion_v4.py --selftest --device ${DEV:-cuda:0} $LIBS --out runs/selftest_v4_pod 2>&1 | tee logs/selftest_v4_pod.log
    ;;
  scattering)
    T=${SCAT_TAG:-scat32_v4}
    $PY $C/page_recursion_v4.py --validate-scattering --rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2,3 --ell 4 --device cuda:0 $LIBS --out runs/$T --tag $T 2>&1 | tee logs/$T.log
    ;;
  timing)
    $PY $C/page_recursion_v4.py $PILOT32 --timing-blocks 3 --timing-transition 1 --tau-tr-doublings 0 --device cuda:0 $LIBS --out runs/timing32_v4 --tag timing32_v4 2>&1 | tee logs/timing32_v4.log
    status timing32_v4
    ;;
  pilot)
    gate_selftest
    nohup $PY $C/page_recursion_v4.py $PILOT32 $RUN32 $LIBS --out runs/pilot32_v4 --tag pilot32_v4 --resume > logs/pilot32_v4.log 2>&1 &
    echo "pilot launched: tail -f logs/pilot32_v4.log; status: bash $C/launch_page_pilot_v4.sh status pilot32_v4"
    ;;
  pilot-fg)
    gate_selftest
    $PY $C/page_recursion_v4.py $PILOT32 $RUN32 $LIBS --out runs/pilot32_v4 --tag pilot32_v4 --resume 2>&1 | tee -a logs/pilot32_v4.log
    status pilot32_v4
    ;;
  controls)
    gate_selftest
    # the pilot's controls (run after the pilot's first stage is read; sequentially on the same GPU, each resumable)
    for spec in "ell8:--ell 8" "point:--fields point" "Rb08:--Rb 0.8" "Rb0_8:--Rb0 8" "ramp:--formation ramp" "box:--nsites 16384" "cut48:--R-cut 48" "taus2:--tau-s 2" "kappa3:--channels 1,2,3" "coll4:--eps-coll 1e-4" "coll6:--eps-coll 1e-6"; do
      tag=${spec%%:*}; args=${spec#*:}
      $PY $C/page_recursion_v4.py $PILOT32 --max-stages 1 --t-stage-max 1000 \
        --ckpt-every 10 --max-wall-hours 6 --device cuda:0 $LIBS --out runs/ctrl32_v4_$tag --tag ctrl32_v4_$tag --resume $args 2>&1 | tee logs/ctrl32_v4_$tag.log
      status ctrl32_v4_$tag
    done
    ;;
  rs64)
    gate_selftest
    nohup $PY $C/page_recursion_v4.py $PILOT64 $RUN64 $LIBS --out runs/fig64_v4 --tag fig64_v4 --resume > logs/fig64_v4.log 2>&1 &
    echo "r_s/a 64 launched: tail -f logs/fig64_v4.log"
    ;;
  rs64-fg)
    gate_selftest
    $PY $C/page_recursion_v4.py $PILOT64 $RUN64 $LIBS --out runs/fig64_v4 --tag fig64_v4 --resume 2>&1 | tee -a logs/fig64_v4.log
    status fig64_v4
    ;;
  status)
    status "${2:?usage: status TAG}"
    ;;
  *)
    echo "usage: bash code_v4/launch_page_pilot_v4.sh selftest | scattering | timing | pilot | pilot-fg | controls | rs64 | rs64-fg | status TAG"; exit 2;;
esac
