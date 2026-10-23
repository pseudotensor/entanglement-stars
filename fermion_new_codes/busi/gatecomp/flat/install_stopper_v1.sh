#!/bin/bash
# install_stopper_v1.sh (busi, GATECOMP; ReviewA_rz256ext_v1 (2), inbox #427): for solve's rented pods (F256_s4, F256_s16). From the unpacked
# stopper_kit_v1: checks MD5SUMS, installs stop_flat_h100f_v2.py and gate_common_v1.py into ${QI2_ROOT}/code and
# gate_profiles_v1.py into ${QI2_ROOT}/busi_gatecomp (where stop_flat finds the six-term of record), then starts the member's stopper:
# --mode birth_plus --after-birth 1.25: the member is stopped by PID at the first json record at or past T_birth + 1.25 r_s (8 to 16 records
# past the plan's T_gate + 1), the run files backed up first; the collar charge alert at 3 percent (no stop). Read only until it acts. A live
# change: no restart (the engine's own --T-over-rs 6.5 lies past T_birth + 1.25 for both members).
#   bash install_stopper_v1.sh S        (S = 4 or 16; M = ${QI2_ROOT}/F256_sS)
set -eu
S=$1; KIT=$(cd "$(dirname "$0")" && pwd); W=${QI2_ROOT}/busi_gatecomp; M=$W/flat/F256_s$S; PY=${PY:-${QI2_ROOT}/python}
[ -x $PY ] || PY=${QI2_ROOT}/python
cd $KIT && md5sum -c --quiet MD5SUMS && echo "stopper kit MD5SUMS ok"
mkdir -p $W/code; cp $KIT/stop_flat_h100f_v2.py $KIT/gate_common_v1.py $W/code/; cp $KIT/gate_profiles_v1.py $W/
A=$(ls $M/run/args*.txt | head -1); TAG=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--tag"{p=1}')
[ -e $M/run/collapse_$TAG.json ] || { echo "no run json for $TAG in $M/run"; exit 3; }
for p in $(ps -eo pid,args | grep "[s]top_flat_h100f_v[12].py $M " | awk '{print $1}'); do kill -TERM $p && echo "replaced the earlier stopper pid $p"; done
cd $M && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 setsid nohup nice -n 10 $PY $W/code/stop_flat_h100f_v2.py $M $TAG ${QI2_ROOT}/code16d_flat \
  --sigma $S --mode birth_plus --after-birth 1.25 --charge-line 0.03 --t-cap 6.5 --period 120 > $M/stop_birthplus.out 2>&1 < /dev/null &
echo "F256_s$S stopper (birth_plus 1.25) pid $!; it writes $M/BIRTH.txt, GATE_LIVE.*, CHARGE.tsv, STOP.txt"
