#!/bin/bash
# Strand S S3, the queue after the inner-form finding (builder analyst): (a) strandS_sphere_v6.py's per-ray inner form re-assembled on the six
# base runs (S3v3; the ladders on every quadrature ray, 10 workers); (b) the spacings 0.025 and 0.0125 per normal on strandS_sphere_v7.py (the
# assembly fix; (1,1,1) with the direct massive check); (c) version 6 on base plus extension per normal, reusing (a)'s ladders; (d) the |p|-rule
# control at (1,1,1) and its re-assembly; (e) the matching radius 0.03 and 0.08 at (1,1,1) and their re-assemblies.  CPU, nice 10.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p S3v5 S3v6
PY=${QI2_ROOT}/python
wait_threads() { until [ "$(bash ${QI2_ROOT}/lane_threads_v1.sh)" -le 6 ]; do sleep 30; done; }
run() { local tag=$1 script=$2 dir=$3; shift 3; wait_threads
  date -u "+S3 $tag start %H:%M:%S UTC (10 workers)" >> S3v6/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/$script "$@" > $dir/run_$tag.log 2>&1
  echo "S3 $tag exit $? $(date -u +%H:%M:%S)" >> S3v6/status.txt; }
for n in 100 110 111 210 211 221; do
  run v6base_$n strandS_sphere_v6.py S3v6 --from-json S3v3/sphere_$n.json --from-npz S3v3/sphere_${n}_points.npz --workers 10 --out-json S3v6/sphere_${n}_v6base.json
done
run ext_111 strandS_sphere_v7.py S3v5 --normal 1,1,1 --aM 0.025,0.0125 --massive-check 12 --workers 10 --out-json S3v5/sphere_111_ext.json
for nn in 1,0,0 1,1,0 2,1,0 2,1,1 2,2,1; do n=$(echo $nn | tr -d ,)
  run ext_$n strandS_sphere_v7.py S3v5 --normal $nn --aM 0.025,0.0125 --workers 10 --out-json S3v5/sphere_${n}_ext.json
done
for n in 100 110 111 210 211 221; do
  run v6all_$n strandS_sphere_v6.py S3v6 --from-json S3v3/sphere_$n.json,S3v5/sphere_${n}_ext.json --from-npz S3v3/sphere_${n}_points.npz,S3v5/sphere_${n}_ext_points.npz --reuse-ladder S3v6/sphere_${n}_v6base.json --workers 2 --out-json S3v6/sphere_${n}_v6all.json
done
run prule_111 strandS_sphere_v7.py S3v5 --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --pnorm-rule --massive-check 12 --workers 10 --out-json S3v5/sphere_111_prule.json
run v6prule_111 strandS_sphere_v6.py S3v6 --from-json S3v5/sphere_111_prule.json --from-npz S3v5/sphere_111_prule_points.npz --workers 10 --out-json S3v6/sphere_111_prule_v6.json
for lc in 800 300; do
  run glim${lc}_111 strandS_sphere_v7.py S3v5 --normal 1,1,1 --aM 0.4,0.2,0.1,0.05,0.025,0.0125 --Lcap $lc --workers 10 --out-json S3v5/sphere_111_Lcap$lc.json
  run v6glim${lc}_111 strandS_sphere_v6.py S3v6 --from-json S3v5/sphere_111_Lcap$lc.json --from-npz S3v5/sphere_111_Lcap${lc}_points.npz --workers 10 --out-json S3v6/sphere_111_Lcap${lc}_v6.json
done
echo DONE >> S3v6/status.txt
