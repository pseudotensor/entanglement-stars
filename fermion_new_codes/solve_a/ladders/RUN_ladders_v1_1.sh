#!/bin/bash
# RUN_ladders_v1_1.sh: MASS_ladder_v1.1 and SCALING_ladder_v1.1 on the box CPU (stored files only; S4_128_s8 from version 1's snapshot).
set -e
C=${QI2_ROOT}/qi2_ladders_code; O=${QI2_ROOT}/ladders_readout; D=$O/v1_1
ENG=${QI2_ROOT}/engine_collapse_v36d.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=${QI2_ROOT}/python
mkdir -p $D; cd $C
CATS="--catalog $O/catalog_v1.json --catalog $O/catalog_v1_A.json --catalog $O/catalog_v1_B.json"
nice -n 10 $PY ladders_readout_v1_1.py $CATS --out-dir $D --snapshot S4_128_s8=$O/v1/snap/collapse_lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d.json
nice -n 10 $PY ladders_sources_v1_1.py --readout-dir $D --code-dir $C $CATS --engine $ENG --out $D/SOURCES_ladders_v1_1.json
nice -n 10 $PY ladders_compose_v1_1.py --readout $D/ladders_readout_v1_1.json --sources $D/SOURCES_ladders_v1_1.json --out-dir $D
nice -n 10 $PY ladders_figs_v1_1.py --readout-dir $D --out-dir $D
cd $D && md5sum ladders_readout_v1_1.json ladders_readout_v1_1.npz SOURCES_ladders_v1_1.json MASS_ladder_v1.1.md SCALING_ladder_v1.1.md *.png *.pdf > MANIFEST_v1_1.md5
echo done $D
