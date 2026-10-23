#!/bin/bash
# merge_replay64_cmp_v1.sh (busi, MERGE-1 self-test (c)): the P64_s4 replay read three ways, box CPU, nice 10, 1 thread, reads only.
#  1. tolcmp_v2.py (ruling v2, ReviewA_tolv2_v1.md: the lane's comparator) X = GA, D = G0, P = the qi2_h100_1 production, end slices at 12.5;
#  2. cmp_rd2_ruling_v2.py (RD-2's ruling-v2 comparator, sharing tolcmp_v2's flip rule) I = G0, C = GA;
#  3. cmp_rd13m_v1.py: G0 (the merged engine's defaults, code16e) against MEM-1's G0 (rd13m defaults, code v16c), bitwise expected.
# Comparators copied from the repo into build/cmp (md5s in cmp/MD5_cmp.txt).
M=${QI2_ROOT}/MERGE_build; C=$M/build/cmp; PY="nice -n 10 env OMP_NUM_THREADS=1 ${QI2_ROOT}/python -B"
P=${QI2_ROOT}/final; TP=lad_P64_s4_newton_v36nq_h100_1
O=$M/gate64; mkdir -p $O/stage/{GA,G0,prod}
for a in GA G0; do for s in "" _fields _spectra _handoff; do ln -sfn $M/replay64/$a/collapse_mf_$a$s.$([ -z "$s" ] && echo json || echo npz) $O/stage/$a/collapse_mf$s.$([ -z "$s" ] && echo json || echo npz); done; done
for s in "" _fields _spectra _handoff; do ln -sfn $P/run/collapse_$TP$s.$([ -z "$s" ] && echo json || echo npz) $O/stage/prod/collapse_mf$s.$([ -z "$s" ] && echo json || echo npz); done
md5sum $C/*.py > $O/MD5_cmp.txt
SGA=$M/replay64/GA/states/orbitals_mf_GA_T0012.50.npz; SG0=$M/replay64/G0/states/orbitals_mf_G0_T0012.50.npz; SP=$P/states/orbitals_${TP}_T0012.50.npz
PC="${PERCALL:-RD-1 per call on stored production matrices (RD1_build/percall/rd13_percall_v1.txt: eigenvalues 7.1e-15, residuals 2.1e-14 ||h||, sets equal to the stored masks; ruling v2 sec. 5); RD-2b field gate on stored negative-centre and intact records (RD2b_build/gate/: fields at most 2.0e-9, E 2.3e-10; ruling v2 sec. 3); MEM-1 bitwise by its gate}"
$PY $C/tolcmp_v2.py $O/stage/GA $O/stage/G0 $O/stage/prod --tag mf --after 11.5 --rs 64 --slice-x $SGA --slice-d $SG0 --slice-p $SP --names GA,G0,prod --percall "$PC" --out $O/tolv2 > $O/tolcmp_v2_GA_G0_prod.txt 2>&1
echo "tolcmp_v2 rc $?" >> $O/rc.txt
$PY $C/cmp_rd2_ruling_v2.py --shared $C/tolcmp_v2.py --label "MERGE-1 (c) P64_s4 T0011.50 -> 12.5, box GPU 1, rd13mf on v16e" --arm I=$M/replay64/G0:mf_G0 --arm C=$M/replay64/GA:mf_GA --rs 64 \
   --prod-json $P/run/collapse_$TP.json --prod-dir $P/run --prod-tag $TP --t-min 11.5 --slice I=$SG0 --slice C=$SGA ${PERCALL_RD2:-} --out $O/cmp_rd2_ruling_v2_G0_GA.json > $O/cmp_rd2_ruling_v2_G0_GA.txt 2>&1
echo "cmp_rd2_ruling_v2 rc $?" >> $O/rc.txt
R=${QI2_ROOT}/G0
$PY $C/cmp_rd13m_v1.py $M/replay64/G0 mf_G0 $R rd13m_G0 --slices-a $M/replay64/G0/states --slices-b $R/states --out $O/cmp_G0_vs_MEM1G0.json > $O/cmp_G0_vs_MEM1G0.txt 2>&1
echo "cmp_rd13m G0 vs MEM-1 G0 rc $?" >> $O/rc.txt
