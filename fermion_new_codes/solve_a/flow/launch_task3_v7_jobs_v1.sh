#!/bin/bash
# Task 3, flow_chart_v7.py (the lead's flow_chart version 5, SPEC_task3_prescribed_flow.md sec. 11.4): the sec. 11.3 jobs, launched only
# after launch_task3_v7_validate_v1.sh's validation against flow_chart_v4.py was read.  (i) The A2 comparator on v4's ten stored A2 records
# (no evolution; one JSON per record); (ii) D1 planar (the two-component planar chain, m = 0.5 and 1, sigma_x = 16, released at rest at
# x = 4 r_s, r_s = 32, R_m = 8; m = 0 records the rejection); (iii) D2 with the exact rest condition at m = 0.5; (iv) the radial D1 re-emitted
# (m = 0.5, 0.1; m = 0 records the rejection) with the momentum-resolved norm inside R_m at every output, the symbol's ODE and ray ensemble;
# (v) compare_v4 on the radial D1 and D2 pairs; (vi) the convert mode's smoke test (not a measurement).
# One CPU thread, nice 10, sequential, gated (below 15 busy lane threads).  Label: prescribed flow (simpler version); v declared.
# Builder analyst.
F=${QI2_ROOT}/flow
cd $F || exit 1
PY="nice -n 10 ${QI2_ROOT}/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
O=task3v7
mkdir -p $O
gate() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -lt 15 ]; do sleep 60; done; }
run() { local tag=$1; shift; gate; echo "$tag start $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $O/status.txt
  $PY code/flow_chart_v7.py "$@" --output $O/$tag.json > $O/$tag.log 2>&1
  echo "$tag exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $O/status.txt; }
for t in A2_m0_kj-1 A2_m0_kj1 A2_m0.1_kj-1 A2_m0.1_kj1 A2r256_m0 A2r256_m0.1 A2r1024_m0 A2r1024_m0.1 A2r1024rm8_m0 A2r1024rm8_m0.1; do
  run A2cmp_$t a2compare --records task3v4/$t.json; done
for m in 0.5 1 0; do run D1planar_m$m planar_dust --mass $m --rs 32 --Rm 8 --sigma-x 16 --half-length 384 --tsample 2; done
run D2_m0.5 radial --launch D2 --rs 32 --Rm 8 --nsites 256 --mass 0.5 --kappa-j -1 --tsample 2
for m in 0.5 0.1 0; do run D1_m$m radial --launch D1 --rs 32 --Rm 8 --nsites 256 --mass $m --kappa-j -1 --tsample 2; done
run D_v7_vs_v4 compare_v4 --pairs task3v4/D1_m0.5.json:$O/D1_m0.5.json task3v4/D1_m0.1.json:$O/D1_m0.1.json task3v4/D2_m0.5.json:$O/D2_m0.5.json
run convert_smoke convert --inv-kappa 8 --delta 0.5 --nsites 256 --mass 0.03125 --ratios 0.25 0.5 1 2 --keep-rejected \
  --note "SMOKE TEST of the version-5 convert mode on a 256-site chain (not a measurement; part (b) runs on the record's package)"
echo "JOBS DONE $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $O/status.txt
