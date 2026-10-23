#!/bin/bash
# c6_2_run_v1.sh (busi f52b1152, 2026-10-20, inbox #323): C6-2 C1 of REVIEWA_write_DISSOLVE_v1, the frozen-field control on C6-1's slices,
# on the CPU of the host that holds them (numpy only, about 2 min per read at 4 threads; 4 reads). Needs $DROOT/gates/C6-1_G.PASS.
#   1. T/r_s 2.0, applied fields: the gate. Must print f_C 0.95, 1.00, 1.08 over 0.25, 0.5, 0.75 r_s and the quarter-zone share 0.378
#      (the write desk's numbers on the stored slice; busi's known answer on the desk's extract of that slice reproduced them on the box).
#   2. T/r_s 1.8125 (the deepest record), applied fields: f_C in the row-18 bands, the number inside the root, the momentum content, the half-clock.
#   3., 4. the same two clocks with the kernel-averaged N, v of the run's fields npz (the dense comparator).
# Writes $DROOT/c6_2/ (logs, npz) and $DROOT/gates/C6-2_T2.PASS or .FAIL. Usage: c6_2_run_v1.sh   (KIT, DROOT, PY as in dissolve_launch_v1.sh)
KIT=${KIT:-${QI2_ROOT}/kit_v1}; ROOT=${DROOT:-${QI2_ROOT}/dissolve}; PY=${PY:-${QI2_ROOT}/python}
T=pair_rs32_ell4a_v36b_G; RUN=$ROOT/runs/$T; ST=$ROOT/states/$T; O=$ROOT/c6_2; G=$ROOT/gates; mkdir -p $O $G
[ -s $G/C6-1_G.PASS ] || { echo "no $G/C6-1_G.PASS: refused"; exit 1; }
( cd $KIT && md5sum -c --quiet MD5SUMS ) || { echo "kit md5 check FAILED; refusing"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
F="--run-json $RUN/collapse_$T.json --records-fields $KIT/ref/collapse_pair_rs32_ell4a_v36b_fields.npz --fields $RUN/collapse_${T}_fields.npz --span 1.0 --out $O"
for c in 0002.00 0001.81; do for fs in applied dense; do
  S=$ST/orbitals_${T}_T$c.npz; [ -s $S ] || { echo "no slice $S: refused"; exit 1; }
  echo "== T$c $fs"; env -u PYTHONPATH nice -n 10 $PY -s -B $KIT/code/dissolve_frozen_c2_v1.py --slice $S --fieldset $fs $F | grep -v "^T/r_s"
done; done
L=$O/frozen_c2_orbitals_${T}_T0002.00_applied.log
ok=1
for want in "f_C over 0.25 r_s:.*f_C 0.95;" "f_C over 0.50 r_s:.*f_C 1.00;" "f_C over 0.75 r_s:.*f_C 1.08;" "^T/r_s 2.0000 (+0.0000):.*|ak|>=pi/4 0.378,"; do
  grep -q "$want" $L || { ok=0; echo "C6-2 gate: not found in $L: $want"; }; done
[ $ok = 1 ] && { echo PASS > $G/C6-2_T2.PASS; echo "C6-2 GATE (T/r_s 2.0 reproduces f_C 0.95, 1.00, 1.08 and 0.378): PASS"; } || { echo FAIL > $G/C6-2_T2.FAIL; echo "C6-2 GATE: FAIL"; exit 1; }
