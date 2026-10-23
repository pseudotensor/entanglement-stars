#!/bin/bash
# Strand 1 baseline rerun on the box: the native-action builder's own code, byte-identical to PR3 head
# (native_run.py etc. from commit 102af54cef3, native_kscan.py from d7a545fd74b, native_core.py/native_config.py from bba9ec75d56).
# Order: pytest controls, k-scan (AGG10 sec. 73), full native run (AGG10 sec. 70), then the two report scripts.
set -u
cd ${QI2_ROOT}/s1_baseline || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=${QI2_ROOT}/python
date -u '+start %Y-%m-%d %H:%M:%S UTC' > status.txt
nice -n 10 $PY -m pytest -q -s tests/test_native_action.py tests/test_native_core.py > pytest.log 2>&1; echo "pytest exit $?" >> status.txt
nice -n 10 $PY native_kscan.py --out out_kscan --workers 16 > kscan.log 2>&1; echo "kscan exit $?" >> status.txt
nice -n 10 $PY native_kscan_report.py > kscan_report.log 2>&1; echo "kscan_report exit $?" >> status.txt
nice -n 10 $PY native_run.py --out out --workers 14 --which pinned,main,cube,register > run.log 2>&1; echo "run exit $?" >> status.txt
nice -n 10 $PY native_report.py --out out --md native_tables.md > report.log 2>&1; echo "report exit $?" >> status.txt
date -u '+end %Y-%m-%d %H:%M:%S UTC' >> status.txt
echo DONE >> status.txt
