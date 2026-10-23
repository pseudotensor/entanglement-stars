#!/bin/bash
# J7 (SPEC_final4b_observables.md, the CPU queue; the line "J7 (last in the CPU queue): the mesh-3 normals ... massless only, corner-adapted
# (strandS_sphere_v7.py with --aM 0.05 so that the massive columns exist but are not read), with the mesh-3 weights from the paper's sphere mesh
# routine"): strandS_sphere_v7.py on the ten vertex normals of the paper's mesh-3 triangulation (the seven new ones and, for one instrument
# throughout, (1,0,0), (1,1,0), (1,1,1)), sequentially, 6 workers, nice 10; outputs in ${QI2_ROOT}/J7. The assembly with the
# mesh weights is obs_J7_assemble_v1.py. The timing (3,1,0 with --timing 8 on 4 workers) projected 1856 outer CPU seconds per normal.
# Usage (on the box, detached): nohup bash code/launch_J7_mesh3_v1.sh > ../t4b/J7/launcher_v1.log 2>&1 < /dev/null &   (from sS_entropy)
# Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p ../t4b/J7
PY=${QI2_ROOT}/python
for nn in 3,1,0 3,1,1 3,2,0 3,2,1 3,2,2 3,3,1 3,3,2 1,0,0 1,1,0 1,1,1; do
  tag=$(echo $nn | tr -d ,)
  echo "J7 $nn start $(date -u +%H:%M:%S) UTC (6 workers)" >> ../t4b/J7/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strandS_sphere_v7.py --normal $nn --aM 0.05 --workers 6 \
    --out-json ../t4b/J7/sphere_${tag}_m3.json > ../t4b/J7/run_${tag}.log 2>&1 < /dev/null
  echo "J7 $nn exit $? $(date -u +%H:%M:%S) UTC" >> ../t4b/J7/status.txt
done
echo "J7 DONE $(date -u +%H:%M:%S) UTC" >> ../t4b/J7/status.txt
