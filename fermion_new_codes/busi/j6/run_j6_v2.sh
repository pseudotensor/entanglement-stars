#!/bin/bash
# J6 v2: the assembly on the existing dumps (no GPU), the extended light channels with the hold series (about 1 GPU hour),
# the assembly again. Run on the pod in the code directory holding the v1 dumps (mdensity.json, light.json or the --out
# names used in v1: substitute them). Stop the nk-48 grid job first (kill its PID; it is not needed for the delivery).
set -e
cd ${QI2_ROOT}/j6
MD=${MD:-mdensity.json}
LT=${LT:-light.json}
OUT=${QI2_ROOT}/j6
mkdir -p $OUT
python -c "import json;d=json.load(open('$MD'));print('max n_zero_modes',max(r['n_zero_modes'] for r in d['results']),'max nz0',max(r['nz0'] for r in d['results']))"
python j6_assemble_v1.py --mdensity $MD --light $LT --parent 1024 --rate-parent 2048 --tau 12 2>&1 | tee $OUT/assemble_v1.log
nice -n 10 python j6_transverse_chain_v2.py selftest --dt 1 2>&1 | tee $OUT/selftest_v3.log
nice -n 10 python j6_transverse_chain_v2.py light --nxs 1024,2048 --dt 1 --taus 0,1,2,3,4,5,6,7,8,9,10,11,12 --out light_v2.json 2>&1 | tee $OUT/light_v2.log
python j6_assemble_v1.py --mdensity $MD --light light_v2.json --parent 1024 --rate-parent 2048 --tau 12 2>&1 | tee $OUT/assemble_v2.log
