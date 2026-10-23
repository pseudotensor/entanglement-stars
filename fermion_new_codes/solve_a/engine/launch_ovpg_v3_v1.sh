#!/bin/bash
# engine_ovpg_v3 (engine spec 28.4 (d), 28.10): the continuum turning points with the native kinetic stress in place of the thermodynamic
# pressure at the six (m a, g) pairs of OVPG, the controls, and the lattice members near each maximum read against every closure.
# Box CPU only: 6 processes x 2 threads (12 threads), nice 10. Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p OVPG3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "ovpg3 v1 start $(date -u +%H:%M:%S) UTC" >> OVPG3/status.txt
nice -n 10 ${QI2_ROOT}/python code/engine_ovpg_v3.py --procs 6 \
  --lattice "0.2:0.25=Sfam6/star_m0.2_g0.25_ell8.json,Sfam6/star_m0.2_g0.25_ell8_ext.json,Sfam6/star_m0.2_g0.25_ell8_ext2.json" \
            "0.1:0.25=Sfam5/star_m0.1_g0.25_ell8.json" \
            "0.05:0.25=Sfam7/star_m0.05_g0.25_ell8.json" \
            "0.2:0.0625=Sfam6/star_m0.2_g0.0625_ell8.json,Sfam6/star_m0.2_g0.0625_ell8_ext.json" \
            "0.1:0.0625=Sfam6/star_m0.1_g0.0625_ell8.json,Sfam6/star_m0.1_g0.0625_ell8_ext.json" \
            "0.1:0.015625=Sfam8/star_star8box_m0.1_g0.015625_max.json" \
  --out OVPG3/ovpg3_v1.json --md OVPG3/ovpg3_v1.md > OVPG3/ovpg3_v1.log 2>&1
echo "ovpg3 v1 exit $? $(date -u +%H:%M:%S) UTC" >> OVPG3/status.txt
