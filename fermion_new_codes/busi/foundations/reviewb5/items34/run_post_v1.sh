#!/bin/bash
# after the stage-2 chain: the background spectrum (s5), the independent checks (s3), the reader (s4), then MD5SUMS. Box CPU, nice 10.
set -u
PY=${QI2_ROOT}/python
C=${QI2_ROOT}/items34
cd ${QI2_ROOT}/items34
while [ ! -f s2_chain_v3.done ]; do sleep 30; done
nice -n 10 $PY -u $C/s5_background_spectrum_v1.py > s5_background_spectrum_v1.log 2>&1
nice -n 10 $PY -u $C/s3_check_v1.py > s3_check_v1.log 2>&1
nice -n 10 $PY -u $C/s4_read_v1.py > s4_read_v1.log 2>&1
echo DONE > post_v1.done
