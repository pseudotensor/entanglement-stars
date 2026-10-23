#!/bin/bash
# J7's companion: the three mesh-2 normals not in the mesh-3 vertex set, (2,1,0), (2,1,1), (2,2,1), on the same instrument (strandS_sphere_v7.py,
# --aM 0.05, massless read), so that eta_sphere at mesh 2 and mesh 3 come from one code; waits for J7's DONE line. 6 workers, nice 10.
# Usage (on the box, detached, from sS_entropy): nohup bash code/launch_J7_mesh2_v1.sh > ../t4b/J7/launcher_m2_v1.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
PY=${QI2_ROOT}/python
until grep -q "J7 DONE" ../t4b/J7/status.txt 2>/dev/null; do sleep 60; done
for nn in 2,1,0 2,1,1 2,2,1; do
  tag=$(echo $nn | tr -d ,)
  echo "J7 m2 $nn start $(date -u +%H:%M:%S) UTC (6 workers)" >> ../t4b/J7/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strandS_sphere_v7.py --normal $nn --aM 0.05 --workers 6 \
    --out-json ../t4b/J7/sphere_${tag}_m3.json > ../t4b/J7/run_${tag}.log 2>&1 < /dev/null
  echo "J7 m2 $nn exit $? $(date -u +%H:%M:%S) UTC" >> ../t4b/J7/status.txt
done
echo "J7 m2 DONE $(date -u +%H:%M:%S) UTC" >> ../t4b/J7/status.txt
