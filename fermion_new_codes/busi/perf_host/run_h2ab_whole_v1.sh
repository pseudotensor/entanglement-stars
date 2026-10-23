#!/bin/bash
# run_h2ab_whole_v1.sh (busi, analyst; PERF H2-A/H2-B item 3): code dirs from tourn331/code (v36z 924eefed), the swap, the six runs
set -u
B=${QI2_ROOT}/H2AB; W=$B/whole; PY=${QI2_ROOT}/python
mkdir -p $W
for d in code16 code16b; do [ -d $W/$d ] || { mkdir -p $W/$d; cp -p ${QI2_ROOT}/*.py $W/$d/; }; done
cmp -s $W/code16b/engine_fields_v16.py $B/src_orig/engine_fields_v16.py && { cp -p $W/code16b/engine_fields_v16.py $W/code16b/engine_fields_v16.py.v16_1ffcf5e8; cp $B/src_v16b/engine_fields_v16b.py $W/code16b/engine_fields_v16.py; cp $B/src_v16b/engine_fields_v16b.py $W/code16b/engine_fields_v16b.py; }
md5sum $W/code16/engine_fields_v16.py $W/code16b/engine_fields_v16.py $W/code16/engine_collapse_v36z.py $W/code16b/engine_collapse_v36z.py
$PY -c "import numpy, scipy, sys; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'python', sys.version.split()[0])"
$PY -I $B/code/h2ab_whole_v1.py --code16 $W/code16 --code16b $W/code16b --out $W/runs --selftest ${QI2_ROOT}/engine_collapse_v36z_selftest.py --ref-dir ${QI2_ROOT}/whole_b
echo done > $W/WHOLE_DONE
