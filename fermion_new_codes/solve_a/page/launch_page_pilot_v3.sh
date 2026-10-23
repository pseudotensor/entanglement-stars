#!/bin/bash
# launch_page_pilot_v3.sh. Changes against launch_page_pilot_v2.sh (builder analyst, BRIEF_page_v3_builder.md; SPEC_final4b_engine.md 30.1):
# every command runs code_v3/page_recursion_v3.py (the v3 code lives in ${QI2_ROOT}/code_v3 on the second H100, the outputs in
# ${QI2_ROOT}/runs); every output folder and tag carries _v3, so nothing overwrites or resumes a v1 or v2 run (30.1 (6)); new target
# "timing" (the formation, the stage-1 eigendecomposition and 3 checkpoints at the pilot's settings, labelled "timing, not the pilot",
# 30.1 (6) and (7)). The pilot, controls and rs64 targets are the lane's to queue after 30.1 (7)'s launch condition holds.
# launch_page_pilot_v1.sh (lead reviewer A, GAP03 integration): the exact commands of GAP03_integrated_v1.md section 12 for the second H100 pod
# (uau1qz03fco10u). Paths relative to ${QI2_ROOT}/page on the pod. Run the steps in order; each writes its own folder under ${QI2_ROOT}/runs.
# Usage: bash code_v3/launch_page_pilot_v3.sh selftest | scattering | timing | pilot | controls | rs64
set -euo pipefail
cd ${QI2_ROOT}/page
mkdir -p runs logs
PY=${PY:-python}
C=code_v3
PILOT32="--rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4"
case "${1:-}" in
  selftest)
    $PY $C/page_recursion_v3.py --selftest --device cuda:0 --engine-path ${QI2_ROOT}/code --out runs/selftest_v3 2>&1 | tee logs/selftest_v3.log
    ;;
  scattering)
    $PY $C/page_recursion_v3.py --validate-scattering --rs 32 --g 0.25 --nsites 8192 --channels 1,2,3 --ell 4 --device cuda:0 --out runs/scat32_v3 --tag scat32_v3 2>&1 | tee logs/scat32_v3.log
    ;;
  timing)
    $PY $C/page_recursion_v3.py $PILOT32 --timing-blocks 3 --device cuda:0 --out runs/timing32_v3 --tag timing32_v3 2>&1 | tee logs/timing32_v3.log
    ;;
  pilot)
    nohup $PY $C/page_recursion_v3.py $PILOT32 --max-stages 200 --E-stop-frac 0.5 \
      --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 10 --max-wall-hours 20 --device cuda:0 \
      --out runs/pilot32_v3 --tag pilot32_v3 --resume > logs/pilot32_v3.log 2>&1 &
    echo "pilot launched: tail -f logs/pilot32_v3.log"
    ;;
  controls)
    # the pilot's controls (run after the pilot's first stage is read; sequentially on the same GPU, each resumable)
    for spec in "ell8:--ell 8" "point:--fields point" "Rb08:--Rb 0.8" "Rb0_8:--Rb0 8" "ramp:--formation ramp" "box:--nsites 16384" "cut48:--R-cut 48" "taus2:--tau-s 2" "kappa3:--channels 1,2,3" "coll4:--eps-coll 1e-4" "coll6:--eps-coll 1e-6"; do
      tag=${spec%%:*}; args=${spec#*:}
      $PY $C/page_recursion_v3.py $PILOT32 --max-stages 1 --t-stage-max 1000 \
        --ckpt-every 10 --max-wall-hours 6 --device cuda:0 --out runs/ctrl32_v3_$tag --tag ctrl32_v3_$tag --resume $args 2>&1 | tee logs/ctrl32_v3_$tag.log
    done
    ;;
  rs64)
    nohup $PY $C/page_recursion_v3.py --rs 64 --g 0.25 --m 0 --nsites 16384 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust \
      --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4 --max-stages 200 --E-stop-frac 0.5 \
      --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 5 --max-wall-hours 60 --device cuda:0 \
      --out runs/fig64_v3 --tag fig64_v3 --resume > logs/fig64_v3.log 2>&1 &
    echo "r_s/a 64 launched: tail -f logs/fig64_v3.log"
    ;;
  *)
    echo "usage: bash code_v3/launch_page_pilot_v3.sh selftest | scattering | timing | pilot | controls | rs64"; exit 2;;
esac
