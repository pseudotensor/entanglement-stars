#!/bin/bash
# RUN_attrib_v1.sh: W5_W4_formation_attribution_v1 and W3_completions_v1 on the box CPU (stored files only; the W5 slice readout must have run).
set -e
C=${QI2_ROOT}/qi2_attrib_code; LC=${QI2_ROOT}/qi2_ladders_code; O=${QI2_ROOT}/attrib_readout
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${QI2_ROOT}/python
cd $C
nice -n 10 $PY attrib_readout_v1.py --ladders-code $LC --w4a-dir ${QI2_ROOT}/default --w5-json $O/w5/w5_readout_v1.json --out-dir $O/attrib
nice -n 10 $PY w3_readout_v1.py --ladders-code $LC --w3-dir $O/w3_from_gdrive --out-dir $O/w3
nice -n 10 $PY attrib_sources_v1.py --attrib-dir $O/attrib --w3-dir $O/w3 --code-dir $C --out $O/SOURCES_attrib_v1.json
nice -n 10 $PY attrib_compose_v1.py --ladders-code $LC --attrib $O/attrib/attrib_readout_v1.json --w3 $O/w3/w3_readout_v1.json --sources $O/SOURCES_attrib_v1.json --out-dir $O
nice -n 10 $PY attrib_figs_v1.py --attrib-dir $O/attrib --w3-dir $O/w3 --out-dir $O
cd $O && md5sum attrib/* w3/* w5/w5_readout_v1.* SOURCES_attrib_v1.json W5_W4_formation_attribution_v1.md W3_completions_v1.md *.png *.pdf > MANIFEST_attrib_v1.md5
echo done
