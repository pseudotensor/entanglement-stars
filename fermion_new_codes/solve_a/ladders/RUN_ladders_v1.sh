#!/bin/bash
# RUN_ladders_v1.sh: MASS_ladder_v1 and SCALING_ladder_v1 on the box CPU (stored files only). Usage: bash RUN_ladders_v1.sh OUTSUBDIR
set -e
C=${QI2_ROOT}/qi2_ladders_code
O=${QI2_ROOT}/ladders_readout
D=$O/${1:-v1}
ENG=${QI2_ROOT}/engine_collapse_v36d.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=${QI2_ROOT}/python
mkdir -p $D
cd $C
CATS=""; for c in $O/catalog_v1.json $O/catalog_v1_A.json $O/catalog_v1_B.json; do [ -f $c ] && CATS="$CATS --catalog $c"; done
nice -n 10 $PY ladders_readout_v1.py $CATS --out-dir $D --snap-dir $D/snap
nice -n 10 $PY ladders_sources_v1.py --readout-dir $D --code-dir $C $CATS --engine $ENG --out $D/SOURCES_ladders_v1.json
nice -n 10 $PY ladders_compose_v1.py --readout $D/ladders_readout_v1.json --sources $D/SOURCES_ladders_v1.json --out-dir $D
nice -n 10 $PY ladders_figs_v1.py --readout-dir $D --out-dir $D
cd $D && md5sum ladders_readout_v1.json ladders_readout_v1.npz SOURCES_ladders_v1.json MASS_ladder_v1.md SCALING_ladder_v1.md *.png *.pdf snap/* > MANIFEST_v1.md5
echo done $D
