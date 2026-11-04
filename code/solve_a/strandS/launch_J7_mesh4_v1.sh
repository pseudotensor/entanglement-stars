#!/bin/bash
# J7 at mesh 4 (an addition to SPEC_final4b_observables.md's J7, which reads mesh 3): strandS_sphere_v7.py on the nine vertex normals of the
# paper's mesh-4 triangulation (sphere.py's triangles(4)) that meshes 2 and 3 do not already carry: (4,1,0), (4,1,1), (4,2,1), (4,3,0),
# (4,3,1), (4,3,2), (4,3,3), (4,4,1), (4,4,3); the other six mesh-4 normals are the mesh-2 ones, already run with the same instrument and
# arguments (launch_J7_mesh3_v1.sh and launch_J7_mesh2_v1.sh). Sequential, 6 workers, nice 10, massless only read (--aM 0.05 as before);
# outputs t4b/J7/sphere_<pqr>_m4.json. The assembly at meshes 2, 3 and 4 is obs_J7_assemble_v2.py. About 8 minutes per normal at mesh 2.
# Usage (on the box, detached): nohup bash code/launch_J7_mesh4_v1.sh > ../t4b/J7/launcher_m4_v1.log 2>&1 < /dev/null &   (from sS_entropy)
# Builder analyst.
cd ${QI2_ROOT}/sS_entropy || exit 1
mkdir -p ../t4b/J7
PY=${QI2_ROOT}/python
for nn in 4,1,0 4,1,1 4,2,1 4,3,0 4,3,1 4,3,2 4,3,3 4,4,1 4,4,3; do
  tag=$(echo $nn | tr -d ,)
  echo "J7 m4 $nn start $(date -u +%H:%M:%S) UTC (6 workers)" >> ../t4b/J7/status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strandS_sphere_v7.py --normal $nn --aM 0.05 --workers 6 \
    --out-json ../t4b/J7/sphere_${tag}_m4.json > ../t4b/J7/run_${tag}.log 2>&1 < /dev/null
  echo "J7 m4 $nn exit $? $(date -u +%H:%M:%S) UTC" >> ../t4b/J7/status.txt
done
echo "J7 m4 DONE $(date -u +%H:%M:%S) UTC" >> ../t4b/J7/status.txt
