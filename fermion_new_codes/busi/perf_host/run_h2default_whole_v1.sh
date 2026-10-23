#!/bin/bash
# run_h2default_whole_v1.sh (busi, analyst; inbox #343): code dirs from rc_v37/code (all seven lineage engines; read only), the kit swap in
# code16c, then h2default_whole_v1.py (pairs of 4 threads = 8). Box CPU, nice 10.
set -u
B=${QI2_ROOT}/H2default; W=$B/whole; PY=${QI2_ROOT}/python; RC=${QI2_ROOT}/code
mkdir -p $W
for d in code16 code16c; do [ -d $W/$d ] || { mkdir -p $W/$d; cp -p $RC/*.py $W/$d/; }; done
if cmp -s $W/code16c/engine_fields_v16.py $B/src_v16c/engine_fields_v16.py; then
  cp -p $W/code16c/engine_fields_v16.py $W/code16c/engine_fields_v16.py.v16_1ffcf5e8; cp $B/src_v16c/engine_fields_v16c.py $W/code16c/engine_fields_v16.py
fi
md5sum $W/code16/engine_fields_v16.py $W/code16c/engine_fields_v16.py $W/code16/engine_collapse_v3[67]*.py | grep -E "fields_v16|v36q_srf|v36r\.|v36nq|v36y\.|v36z\.|v37a|v37n"
diff <(cd $W/code16 && md5sum *.py) <(cd $W/code16c && md5sum *.py)
$PY -c "import numpy, scipy, sys; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'python', sys.version.split()[0])"
$PY -I $B/code/h2default_whole_v1.py --code16 $W/code16 --code16c $W/code16c --out $W/runs --selftest ${QI2_ROOT}/engine_collapse_v36z_selftest.py --threads 4 ${ONLY:+--only $ONLY}
echo done > $W/WHOLE_DONE
