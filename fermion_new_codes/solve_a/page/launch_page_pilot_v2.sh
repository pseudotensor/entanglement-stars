#!/bin/bash
# launch_page_pilot_v2.sh: launch_page_pilot_v1.sh with page_recursion_v2.py in every command (the v2 fixes, BRIEF_page_debug_v1.md) and
# --engine-path ${QI2_ROOT}/code on the self-test (T1a); every output folder and tag carries _v2 so that no v2 run overwrites or resumes
# a v1 result (runs/scat32 from v1 exists on the pod); nothing else changes (builder analyst).
# launch_page_pilot_v1.sh (lead reviewer A, GAP03 integration): the exact commands of GAP03_integrated_v1.md section 12 for the second H100 pod
# (uau1qz03fco10u). Paths relative to ${QI2_ROOT}/page on the pod. Run the steps in order; each writes its own folder under ${QI2_ROOT}/runs.
# Usage: bash launch_page_pilot_v2.sh selftest | scattering | pilot | controls | rs64
set -euo pipefail
cd ${QI2_ROOT}/page
mkdir -p runs logs
PY=${PY:-python}
case "${1:-}" in
  selftest)
    $PY page_recursion_v2.py --selftest --device cuda:0 --engine-path ${QI2_ROOT}/code --out runs/selftest_v2 2>&1 | tee logs/selftest_v2.log
    ;;
  scattering)
    $PY page_recursion_v2.py --validate-scattering --rs 32 --g 0.25 --nsites 8192 --channels 1,2,3 --ell 4 --device cuda:0 --out runs/scat32_v2 --tag scat32_v2 2>&1 | tee logs/scat32_v2.log
    ;;
  pilot)
    nohup $PY page_recursion_v2.py --rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust \
      --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4 --max-stages 200 --E-stop-frac 0.5 \
      --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 10 --max-wall-hours 20 --device cuda:0 \
      --out runs/pilot32_v2 --tag pilot32_v2 --resume > logs/pilot32_v2.log 2>&1 &
    echo "pilot launched: tail -f logs/pilot32_v2.log"
    ;;
  controls)
    # the pilot's controls (run after the pilot's first stage is read; sequentially on the same GPU, each resumable)
    for spec in "ell8:--ell 8" "point:--fields point" "Rb08:--Rb 0.8" "Rb0_8:--Rb0 8" "ramp:--formation ramp" "box:--nsites 16384" "cut48:--R-cut 48" "taus2:--tau-s 2" "kappa3:--channels 1,2,3" "coll4:--eps-coll 1e-4" "coll6:--eps-coll 1e-6"; do
      tag=${spec%%:*}; args=${spec#*:}
      $PY page_recursion_v2.py --rs 32 --g 0.25 --m 0 --nsites 8192 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust \
        --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4 --max-stages 1 --t-stage-max 1000 \
        --ckpt-every 10 --max-wall-hours 6 --device cuda:0 --out runs/ctrl32_v2_$tag --tag ctrl32_v2_$tag --resume $args 2>&1 | tee logs/ctrl32_v2_$tag.log
    done
    ;;
  rs64)
    nohup $PY page_recursion_v2.py --rs 64 --g 0.25 --m 0 --nsites 16384 --channels 1,2 --ell 4 --fields averaged --Rb 0.5 --Rb0 4 --formation dust \
      --R-cut 32 --R-collar 16 --tau-s 4 --dE-frac 0.005 --interior selfsimilar --tau-tr 4 --max-stages 200 --E-stop-frac 0.5 \
      --t-stage-max 3000 --t-stall-min 200 --stall-frac 0.01 --ckpt-every 5 --max-wall-hours 60 --device cuda:0 \
      --out runs/fig64_v2 --tag fig64_v2 --resume > logs/fig64_v2.log 2>&1 &
    echo "r_s/a 64 launched: tail -f logs/fig64_v2.log"
    ;;
  *)
    echo "usage: bash launch_page_pilot_v2.sh selftest | scattering | pilot | controls | rs64"; exit 2;;
esac
