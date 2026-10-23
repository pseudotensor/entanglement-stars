#!/bin/bash
# RUN_ladders_v1_3.sh: MASS_ladder_v1.3 and SCALING_ladder_v1.3 on the box CPU (stored files only). Version 1.2's pipeline with one change,
# the certification inputs (deliveries/lead_gaps/ReviewA_cert_rule_v1.md, the production rule): the producer counts a record when its kept
# and confirming residuals are both <= 1e-3 (plain records kept unconverged: the iteration_history entry after the kept iterate), marks the
# banded ones (1e-4 to 1e-3), and checks its extents against certext_production_v1.json and certext_followup_v1.json.
set -e
C=${QI2_ROOT}/qi2_ladders_code; O=${QI2_ROOT}/ladders_readout; D=$O/v1_3
ENG=${QI2_ROOT}/engine_collapse_v36d.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
PY=${QI2_ROOT}/python
mkdir -p $D; cd $C
cp $O/v1_2/certext_production_v1.json $D/
CATS="--catalog $O/catalog_v1.json --catalog $O/catalog_v1_A.json --catalog $O/catalog_v1_B.json"
nice -n 10 $PY certext_followup_v1.py --certext $D/certext_production_v1.json --out $D/certext_followup_v1.json
nice -n 10 $PY ladders_readout_v1_3.py $CATS --out-dir $D --certext $D/certext_production_v1.json --followup $D/certext_followup_v1.json
nice -n 10 $PY ladders_diff_v1_3.py $O/v1_2/ladders_readout_v1_2.json $D/ladders_readout_v1_3.json $C $D/ladders_diff_v1_3.json
nice -n 10 $PY ladders_sources_v1_3.py --readout-dir $D --code-dir $C $CATS --engine $ENG --certext $D/certext_production_v1.json --followup $D/certext_followup_v1.json --out $D/SOURCES_ladders_v1_3.json
nice -n 10 $PY ladders_compose_v1_3.py --readout $D/ladders_readout_v1_3.json --sources $D/SOURCES_ladders_v1_3.json --out-dir $D --diff $D/ladders_diff_v1_3.json --old $O/v1_2/ladders_readout_v1_2.json
nice -n 10 $PY ladders_figs_v1_3.py --readout-dir $D --out-dir $D
cd $D && md5sum certext_production_v1.json certext_followup_v1.json ladders_readout_v1_3.json ladders_readout_v1_3.npz ladders_diff_v1_3.json SOURCES_ladders_v1_3.json MASS_ladder_v1.3.md SCALING_ladder_v1.3.md *.png *.pdf > MANIFEST_v1_3.md5
echo done $D
