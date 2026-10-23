#!/bin/bash
# Strand 3 addendum parts C and E, queued behind the 2b runs (ring4 pass 2 and the cube pass 1, which hold the lane's CPU budget):
# C: eta_100 at the masses 4 q_s on the (32,24) ladder (strand3_planar.py, 4 single-thread workers) and the mesh-2 sphere at the two 0.02
# masses (strand3_batches.py --what sphere, 2 workers each), then the assembly (strand3_addCD_v1.py C-assemble);
# E: the paper's balls.py measure at radius 8.5, box 96, mass 0, SLQ 32 probes, 96 steps, seed 781, centre (0,0,0), 4 BLAS threads.
set -u
D=${QI2_ROOT}/s3_entropy
cd $D || exit 1
PY=${QI2_ROOT}/python
date -u '+queued CE %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
until grep -q DONE ${QI2_ROOT}/status.txt 2>/dev/null && grep -q DONE ${QI2_ROOT}/status.txt 2>/dev/null; do sleep 60; done
date -u '+start CE %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
mkdir -p addendum1/E
(export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4; nice -n 10 $PY code/orient/balls.py measure --radius 8.5 --box 96 --method slq --probes 32 --steps 96 --seed 781 --mass 0 --center 0 0 0 --output addendum1/E/ball_q0_c0_0_0_box96.json > addendum1/E/ball_box96.log 2>&1; echo "E exit $?" >> addendum1/status.txt) &
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 10 $PY code/strand3_planar.py --masses addendum1/C/masses_addC.json --out addendum1/C --workers 4 > addendum1/C/planar.log 2>&1; echo "C planar exit $?" >> addendum1/status.txt
nice -n 10 $PY code/strand3_batches.py --masses addendum1/C/masses_addC.json --out addendum1/C --what sphere > addendum1/C/sphere_batch.log 2>&1; echo "C sphere exit $?" >> addendum1/status.txt
nice -n 10 $PY code/strand3_addCD_v1.py C-assemble --out addendum1/C --s3run run1 > addendum1/C/assemble.log 2>&1; echo "C assemble exit $?" >> addendum1/status.txt
wait
date -u '+end CE %Y-%m-%d %H:%M:%S UTC' >> addendum1/status.txt
echo "CE DONE" >> addendum1/status.txt
