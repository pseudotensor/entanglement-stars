#!/bin/bash
# Strand 2d control C7 (sec. 8(1), 8(3)) run now in a spare thread (the small chain reaches C7 only after the 2c kernel and the strand 4
# midpoint rerun); same code and inputs as the chain's call, so the chain's later C7 rewrites identical JSON.  Status in
# c7/status_early.txt (not c7/status.txt, whose DONE starts the jacobian-gap check in the chain's slot).  One thread.
set -u
B=${QI2_ROOT}/s2d_jet
cd $B || exit 1
mkdir -p c7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
date -u '+early start %Y-%m-%d %H:%M:%S UTC' >> c7/status_early.txt
nice -n 10 $PY code/strand2d_c7_v1.py --out c7 --lattice C8 --jet C8run2/jet_C8_gM0_gB0.json --prep31 prep31_C8/prep31_C8_gM0_gB0.json > c7/c7_C8_early.log 2>&1; echo "c7 C8 exit $?" >> c7/status_early.txt
nice -n 10 $PY code/strand2d_c7_v1.py --out c7 --lattice B4 --jet B4run1/jet_B4_gM0_gB0.json --prep31 prep31_B4/prep31_B4_gM0_gB0.json > c7/c7_B4_early.log 2>&1; echo "c7 B4 exit $?" >> c7/status_early.txt
date -u '+early end %Y-%m-%d %H:%M:%S UTC' >> c7/status_early.txt
