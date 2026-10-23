#!/bin/bash
# merge_iso_cmp_h100f_v1.sh (busi, MERGE-1 self-test (d)): the ISO_128 T0005.00 -> 5.5 replay on h100-f read three ways (CPU, 2 threads).
# The default arm D is the RD-1 entry replay's D arm (${QI2_ROOT}/D: rd13 7d362033 defaults, same card, venv and threads,
# read only); our D0 arm (the merged engine's defaults, 2 records) is checked bitwise against it first (cmp_rd13m_v1).
#   1. cmp_rd13m_v1: D0 against D (bitwise expected at the common records 5.0625, 5.125);
#   2. tolcmp_v2 X = A (or A2, F), D, P = the pod6 production records; no production slice exists at 5.5, so the slice is read X-D only;
#   3. cmp_rd2_ruling_v2 I = D, C = A, the per-call field evidence (2(c)(3)) from the field gate on the production's records 81-88.
#   usage: merge_iso_cmp_h100f_v1.sh "A A2 F" [GATE_JSON]
set -u
W=${QI2_ROOT}/busi_merge; C=$W/cmp; PY="nice -n 10 env OMP_NUM_THREADS=2 ${QI2_ROOT}/python -B"
ARMS=${1:-A}; GJ=${2:-}
I=$W/inputs/iso128_h2/run_h2; TP=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
E=${QI2_ROOT}/D
O=$W/out/isocmp; mkdir -p $O/stage/D $O/stage/prod
lk() { for s in "" _fields _spectra _handoff; do x=npz; [ -z "$s" ] && x=json; ln -sfn $1$s.$x $2/collapse_mf$s.$x; done; }
lk $E/collapse_rd13_D $O/stage/D; lk $I/collapse_$TP $O/stage/prod
[ -e $W/out/iso/D0/collapse_mf_D0.json ] && $PY $C/cmp_rd13m_v1.py $W/out/iso/D0 mf_D0 $E rd13_D --out $O/cmp_D0_vs_entryD.json > $O/cmp_D0_vs_entryD.txt 2>&1
PC="RD-1 per call on the ISO_128 T0005.00 matrices (RD1_build/percall/rd13_percall_v1.txt, 2n 8192, kappa 1 and 10: eigenvalues 7.1e-15, residuals 2.1e-14 ||h||, tracked sets equal to the stored masks; ruling v2 sec. 5); fields: ${GJ:-the field gate not given}"
for X in $ARMS; do
  mkdir -p $O/stage/$X; lk $W/out/iso/$X/collapse_mf_$X $O/stage/$X
  SX=$W/out/iso/$X/states/orbitals_mf_${X}_T0005.50.npz; SD=$E/states/orbitals_rd13_D_T0005.50.npz
  $PY $C/tolcmp_v2.py $O/stage/$X $O/stage/D $O/stage/prod --tag mf --after 5.0 --rs 128 --slice-x $SX --slice-d $SD --names $X,D,prod --percall "$PC" --out $O/tolv2_$X > $O/tolcmp_v2_${X}_D_prod.txt 2>&1
  echo "tolcmp_v2 $X rc $?" >> $O/rc.txt
  PCR=""; [ -n "$GJ" ] && PCR="--percall C=gate:$GJ"
  $PY $C/cmp_rd2_ruling_v2.py --shared $C/tolcmp_v2.py --label "MERGE-1 (d) ISO_128 T0005.00 -> 5.5, h100-f, rd13mf $X on v16e vs the rd13 defaults" --arm I=$E:rd13_D --arm C=$W/out/iso/$X:mf_$X --rs 128 \
     --prod-json $I/collapse_$TP.json --prod-dir $I --prod-tag $TP --t-min 5.0 --slice I=$SD --slice C=$SX $PCR --out $O/cmp_rd2_ruling_v2_D_$X.json > $O/cmp_rd2_ruling_v2_D_$X.txt 2>&1
  echo "cmp_rd2_ruling_v2 $X rc $?" >> $O/rc.txt
done
