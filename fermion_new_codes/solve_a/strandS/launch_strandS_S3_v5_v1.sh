#!/bin/bash
# Strand S, S3 controls of the lead's judgment sec. 8.5(6) on strandS_sphere_v5.py (CPU, 10 workers, nice 10), queued behind S3v3:
# (1) the spacings 0.025 and 0.0125 for every normal (the base rule; (1,1,1) with the direct massive-channel check), merged with S3v3's base
# rows in the report; (2) at (1,1,1) the matching radius g_lim = 0.03 and 0.08 (Lcap 800, 300) over all six spacings; (3) at (1,1,1) the
# |p| finite-interval rule over all six spacings with the massive check.  Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S3v5
until grep -qx "DONE" S3v3/status.txt 2>/dev/null; do sleep 30; done
run() { local tag=$1; shift
  until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 6 ]; do sleep 30; done
  date -u "+S3v5 $tag start %H:%M:%S UTC (10 workers)" >> S3v5/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 ${QI2_ROOT}/python code/strandS_sphere_v5.py "$@" \
    --workers 10 --out-json S3v5/sphere_$tag.json > S3v5/run_$tag.log 2>&1
  echo "S3v5 $tag exit $? $(date -u +%H:%M:%S)" >> S3v5/status.txt; }
run 111_ext --normal 1,1,1 --aM 0.025,0.0125 --massive-check 12
for n in 1,0,0 1,1,0 2,1,0 2,1,1 2,2,1; do run $(echo $n | tr -d ,)_ext --normal $n --aM 0.025,0.0125; done
run 111_glim0.03 --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --Lcap 800
run 111_glim0.08 --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --Lcap 300
run 111_prule --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --pnorm-rule --massive-check 12
echo DONE >> S3v5/status.txt
