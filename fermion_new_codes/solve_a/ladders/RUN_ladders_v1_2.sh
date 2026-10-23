#!/bin/bash
# RUN_ladders_v1_2.sh: MASS_ladder_v1.2 and SCALING_ladder_v1.2 on the box CPU (stored files only). Version 1.1's pipeline with one input change:
# S4_128_s8 is read from its final run json (finished to T/r_s 40 on R1; box ${QI2_ROOT}/..., md5 ff475918)
# instead of version 1's snapshot; ladders_diff_v1_2.py lists every producer value that changed from version 1.1.
set -e
C=${QI2_ROOT}/qi2_ladders_code; O=${QI2_ROOT}/ladders_readout; D=$O/v1_2
ENG=${QI2_ROOT}/engine_collapse_v36d.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=${QI2_ROOT}/python
mkdir -p $D; cd $C
CATS="--catalog $O/catalog_v1.json --catalog $O/catalog_v1_A.json --catalog $O/catalog_v1_B.json"
nice -n 10 $PY ladders_readout_v1_2.py $CATS --out-dir $D
nice -n 10 $PY ladders_diff_v1_2.py $O/v1_1/ladders_readout_v1_1.json $D/ladders_readout_v1_2.json $C $D/ladders_diff_v1_2.json
nice -n 10 $PY ladders_sources_v1_2.py --readout-dir $D --code-dir $C $CATS --engine $ENG --out $D/SOURCES_ladders_v1_2.json
nice -n 10 $PY ladders_compose_v1_2.py --readout $D/ladders_readout_v1_2.json --sources $D/SOURCES_ladders_v1_2.json --out-dir $D --diff $D/ladders_diff_v1_2.json
nice -n 10 $PY ladders_figs_v1_2.py --readout-dir $D --out-dir $D
cd $D && md5sum ladders_readout_v1_2.json ladders_readout_v1_2.npz ladders_diff_v1_2.json SOURCES_ladders_v1_2.json MASS_ladder_v1.2.md SCALING_ladder_v1.2.md *.png *.pdf > MANIFEST_v1_2.md5
echo done $D
