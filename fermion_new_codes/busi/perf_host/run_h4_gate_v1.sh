#!/bin/bash
# run_h4_gate_v1.sh (busi, analyst; PERF H4(a), #348): h4_slice_async_gate_v1.py on the box CPU (nice 10, 8 threads) in ${QI2_ROOT}/H4.
set -u
H=${QI2_ROOT}/H4; PY=${QI2_ROOT}/python
cd $H && md5sum code/engine_collapse_v36nq.py code/engine_collapse_v36nq_aw.py build/build_engine_collapse_v36nq_aw.py h4_slice_async_gate_v1.py > $H/MD5SUMS_H4.txt
nice -n 10 $PY -I $H/h4_slice_async_gate_v1.py --code $H/code --out $H/gate --threads 8 ${1:+--only $1}
echo "rc $?" > $H/GATE_DONE
