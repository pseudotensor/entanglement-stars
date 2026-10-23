#!/bin/bash
# install_logcheck_v2.sh (= install_logcheck_v1.sh with the field module's code dir $CODE, default code16f_flat: v16f, ReviewA_s4fail_v1) (busi, GATECOMP; ReviewA_cvvoid_v1 sec. 2, inbox #426): for solve's rented pods (F256_s4, F256_s16). Starts the
# LOG-ONLY centre check logcheck_v1.py beside the member: per record N(0), the six-term at r = sigma, 2 sigma, 4 sigma, nsub per channel.
# It reads the member's run json and fields npz only (the npz copied to a private temp), writes $M/LOGCHECK.tsv and $M/logcheck.out, and
# touches nothing else: not the engine, not nsub, not the stopper (no signal, no file of theirs). Needs the stopper kit installed first
# (gate_common_v1.py in ${QI2_ROOT}/code, gate_profiles_v1.py in ${QI2_ROOT}/busi_gatecomp), as install_stopper_v1.sh does.
#   bash install_logcheck_v1.sh S        (S = 4 or 16; M = ${QI2_ROOT}/F256_sS)
set -eu
S=$1; KIT=$(cd "$(dirname "$0")" && pwd); W=${QI2_ROOT}/busi_gatecomp; M=$W/flat/F256_s$S; PY=${PY:-${QI2_ROOT}/python}
[ -x $PY ] || PY=${QI2_ROOT}/python
cd $KIT && md5sum -c --quiet MD5SUMS && echo "logcheck kit MD5SUMS ok"
[ -e $W/code/gate_common_v1.py ] && [ -e $W/gate_profiles_v1.py ] || { echo "stopper kit not installed (gate_common_v1 / gate_profiles_v1 missing)"; exit 3; }
cp $KIT/logcheck_v1.py $W/code/
A=$(ls $M/run/args*.txt | head -1); TAG=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--tag"{p=1}')
[ -e $M/run/collapse_$TAG.json ] || { echo "no run json for $TAG in $M/run"; exit 3; }
for p in $(ps -eo pid,args | grep "[l]ogcheck_v1.py $M " | awk '{print $1}'); do kill -TERM $p && echo "replaced the earlier logcheck pid $p"; done
cd $M && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 setsid nohup nice -n 15 $PY $W/code/logcheck_v1.py $M $TAG ${CODE:-${QI2_ROOT}/code16f_flat} \
  --sigma $S --period 120 > $M/logcheck.out 2>&1 < /dev/null &
echo "F256_s$S logcheck pid $!; it writes $M/LOGCHECK.tsv (one row per record once its successor is stored) and $M/logcheck.out"
