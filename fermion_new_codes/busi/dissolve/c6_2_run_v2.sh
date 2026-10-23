#!/bin/bash
# c6_2_run_v2.sh (busi, reviewer A red team DISSOLVE_redteam_reviewa_v1, 2026-10-20): c6_2_run_v1.sh (8b516b71) with the red team's fixes:
#   (1) --span 2.5 (version 1's 1.0 cannot print the exit's half-clock: at 2.0 it is 1.25 r_s after the slice, at 1.8125 later still, and the
#       acceptance reads it against the episode window; the propagation is exact in the eigenbasis, the extra steps cost seconds);
#   (2) the applied-field reads only: version 1's "dense" fieldset read the fields npz's N, v, which the engine stores as the kernel-averaged
#       site fields (engine_collapse_v36b line 2059: solve() returns site_fields_kernel_average's output; line 2429 stores it), equal to
#       N_applied, v_applied within the Newton residual, so that read duplicated the applied one; a dense comparator needs a re-solve
#       (the ruling's section 2.3) and is not built here;
#   (3) any member: MEMBER=<launcher member> (default G) reads that member's slices with its own json and fields npz as the records (G reads
#       the stored run's fields in the kit, which run to 40, as version 1 did), at the clocks CLOCKS (default "0002.00 0001.81"; file-name
#       form), with R* at the member's own deepest record (the smallest dense min(N - |v|) among its rooted records; G's is 1.8125).
# The gate (C6-2_T2.PASS: f_C 0.95, 1.00, 1.08 over 0.25, 0.5, 0.75 r_s and the quarter-zone share 0.378 at 2.0) is written for G only.
# Needs $DROOT/gates/C6-1_G.PASS. CPU of the host that holds the slices (numpy only, about 1 to 2 min per read at 4 threads).
# Writes $DROOT/c6_2/<member>/ (logs, npz). Usage: [MEMBER=G] [CLOCKS="0002.00 0001.81"] c6_2_run_v2.sh   (KIT, DROOT, PY as in dissolve_launch_v2.sh)
KIT=${KIT:-${QI2_ROOT}/kit_v1}; ROOT=${DROOT:-${QI2_ROOT}/dissolve}; PY=${PY:-${QI2_ROOT}/python}
MEMBER=${MEMBER:-G}; CLOCKS=${CLOCKS:-"0002.00 0001.81"}
BT=pair_rs32_ell4a_v36b
case $MEMBER in G|P0|X05|X20|L2|L8|S1|S05|W) T=${BT}_$MEMBER;; P40_*) T=${BT}_${MEMBER#P40_}_P40;; M64) T=pair_rs64_ell8a_v36b_M64;; *) echo "unknown member $MEMBER"; exit 1;; esac
RUN=$ROOT/runs/$T; ST=$ROOT/states/$T; O=$ROOT/c6_2/$MEMBER; G=$ROOT/gates; mkdir -p $O $G
[ -s $G/C6-1_G.PASS ] || { echo "no $G/C6-1_G.PASS: refused"; exit 1; }
( cd $KIT && md5sum -c --quiet MD5SUMS ) || { echo "kit md5 check FAILED; refusing"; exit 1; }
[ -s $RUN/collapse_$T.json ] || { echo "no $RUN/collapse_$T.json: refused"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
if [ "$MEMBER" = G ]; then REC=$KIT/ref/collapse_${BT}_fields.npz; else REC=$RUN/collapse_${T}_fields.npz; fi
RSTAR=$(env -u PYTHONPATH $PY -s -B -c "
import json, sys
J = json.load(open(sys.argv[1])); r = [x for x in J['records'] if x.get('E') is not None and x.get('trapping_zeros')]
print('%.6f' % min(r, key=lambda x: x['min_N_minus_absv'])['T_over_rs'] if r else 'none')" $RUN/collapse_$T.json)
[ "$RSTAR" != none ] || { echo "member $MEMBER has no record with a dense zero: no C6-2 read (nothing written)"; exit 0; }
echo "C6-2 on $MEMBER ($T): records' fall from $REC; R* at the member's deepest record T/r_s $RSTAR; clocks $CLOCKS; span 2.5 r_s"
F="--run-json $RUN/collapse_$T.json --records-fields $REC --rstar-T $RSTAR --fieldset applied --span 2.5 --out $O"
for c in $CLOCKS; do
  S=$ST/orbitals_${T}_T$c.npz; [ -s $S ] || { echo "no slice $S: refused"; exit 1; }
  echo "== T$c applied"; env -u PYTHONPATH nice -n 10 $PY -s -B $KIT/code/dissolve_frozen_c2_v1.py --slice $S $F | grep -v "^T/r_s"
done
[ "$MEMBER" = G ] || exit 0
L=$O/frozen_c2_orbitals_${T}_T0002.00_applied.log
[ -s $L ] || { echo "C6-2 gate: no 2.0 read for G (CLOCKS without 0002.00): no gate written"; exit 0; }
ok=1
for want in "f_C over 0.25 r_s:.*f_C 0.95;" "f_C over 0.50 r_s:.*f_C 1.00;" "f_C over 0.75 r_s:.*f_C 1.08;" "^T/r_s 2.0000 (+0.0000):.*|ak|>=pi/4 0.378,"; do
  grep -q "$want" $L || { ok=0; echo "C6-2 gate: not found in $L: $want"; }; done
[ $ok = 1 ] && { echo PASS > $G/C6-2_T2.PASS; echo "C6-2 GATE (T/r_s 2.0 reproduces f_C 0.95, 1.00, 1.08 and 0.378): PASS"; } || { echo FAIL > $G/C6-2_T2.FAIL; echo "C6-2 GATE: FAIL"; exit 1; }
