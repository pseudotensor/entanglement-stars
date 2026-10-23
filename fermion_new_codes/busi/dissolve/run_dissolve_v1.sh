#!/bin/bash
# run_dissolve_v1.sh (busi f52b1152, 2026-10-20, inbox #323): the card-queue entry for one DISSOLVE member (REVIEWA_write_DISSOLVE_v1, MERGED
# SPECIFICATION). Model: ${QI2_ROOT}/run_J1_W2_anc_v1.sh. Verifies the kit's MD5SUMS first, launches through
# dissolve_launch_v1.sh, checks the pid is the kit's engine, blocks until it exits and exits with its rc. For G (C6-1) it then runs the
# regression gate c6_1_gate_v1.sh, which writes $DROOT/gates/C6-1_G.PASS or .FAIL (every later member refuses without the PASS file).
# Never kills anything. Usage: run_dissolve_v1.sh <member>   (KIT, DROOT, PY, GPU as in dissolve_launch_v1.sh)
export NUMPY_MADVISE_HUGEPAGE=0
KIT=${KIT:-${QI2_ROOT}/kit_v1}; ROOT=${DROOT:-${QI2_ROOT}/dissolve}; export KIT DROOT=$ROOT; now() { date -u +%FT%TZ; }
M=${1:?usage: run_dissolve_v1.sh <member>}
echo "$(now) $(hostname) DISSOLVE $M start; checking kit md5s"
( cd $KIT && md5sum -c --quiet MD5SUMS ) || { echo "$(now) kit md5 check FAILED; refusing"; exit 1; }
echo "$(now) kit md5s OK"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
bash $KIT/dissolve_launch_v1.sh $M main || { echo "$(now) launcher refused"; exit 1; }
PF=$ROOT/pids/DISSOLVE_$M.pid; P=$(cat $PF)
tr "\0" " " < /proc/$P/cmdline | grep -q "$KIT/code/engine_collapse_v36b.py" || { echo "$(now) pid $P is not the kit's engine"; exit 1; }
while kill -0 $P 2>/dev/null; do sleep 60; done
RC=$(cat $PF.rc 2>/dev/null || echo 1)
echo "$(now) engine pid $P ended rc $RC"
if [ "$M" = G ] && [ "$RC" = 0 ]; then bash $KIT/c6_1_gate_v1.sh; echo "$(now) C6-1 gate rc $?"; fi
exit $RC
