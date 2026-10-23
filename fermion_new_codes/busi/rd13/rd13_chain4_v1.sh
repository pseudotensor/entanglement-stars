#!/bin/bash
# rd13_chain4_v1.sh (busi, analyst): part (c) on the P64_s4 Newton member, T0011.50 -> 12.5 (8 records across the first formation record
# 12.125), arms B (--eigh banded --jvp sternheimer), D (rd13 defaults), P (the parent v36nq), all 8 records; then cmp_rd13_replay_v2 (my
# budget, the ruling's gate, P against D bitwise) and reviewer A's tolcmp_v1 (B against D, each against the production); then (b) and the
# fixture's GPU smoke again with the final module.
set -u
B=${QI2_ROOT}/RD1_build; PY=${QI2_ROOT}/python; export GPU=${GPU:-2}
P=${QI2_ROOT}/final; TAG0=lad_P64_s4_newton_v36nq_h100_1
I=${QI2_ROOT}/iso128_h2; T=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
W=${QI2_ROOT}/T0013.50
WT=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_r2b
L=$B/logs/chain4_v1.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $L; }
cp -p $B/build/engine_banded_eigh_v1.py $B/code/engine_banded_eigh_v1.py
md5sum $B/code/engine_banded_eigh_v1.py $B/code/engine_collapse_v36nq_rd13.py $B/code/engine_collapse_v36nq.py $B/code/engine_fields_v16.py >> $L
for A in "B engine_collapse_v36nq_rd13.py --eigh banded --jvp sternheimer" "D engine_collapse_v36nq_rd13.py" "P engine_collapse_v36nq.py"; do
  set -- $A; nm=$1; eng=$2; shift 2
  say "arm $nm start"; bash $B/build/rd13_replay_v2.sh $nm $eng 12.5 "$@"; say "arm $nm rc $?"
done
cd $B
nice -n 10 $PY $B/build/cmp_rd13_replay_v2.py --replay $B/replay64 --prod $P/run/$TAG0 --arms D,B --parent P --t0 11.5 --out $B/replay64/cmp_rd13_replay_v2 > $B/logs/cmp64_v2.log 2>&1
say "cmp v2 rc $?"
mkdir -p $B/tolcmp64/B $B/tolcmp64/D $B/tolcmp64/P $B/tolcmp64/prod
for X in B D P; do for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $B/replay64/$X/collapse_rd13_${X}${s}.$e $B/tolcmp64/$X/collapse_rd13${s}.$e; done; done
for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $P/run/collapse_${TAG0}${s}.$e $B/tolcmp64/prod/collapse_rd13${s}.$e; done
SB=$(ls $B/replay64/B/states/*.npz | tail -1); SD=$(ls $B/replay64/D/states/*.npz | tail -1)
TC="nice -n 10 $PY ${QI2_ROOT}/tolcmp_v1.py"
$TC $B/tolcmp64/B $B/tolcmp64/D --tag rd13 --after 11.5 --rs 64 --nameA "B (banded + sternheimer, box A6000)" --nameB "D (defaults, box A6000)" --slice-a $SB --slice-b $SD --out $B/tolcmp64/tol_B_vs_D.json > $B/tolcmp64/tol_B_vs_D.txt 2>&1
say "tolcmp B-D rc $?"
$TC $B/tolcmp64/P $B/tolcmp64/D --tag rd13 --after 11.5 --rs 64 --nameA "P (parent v36nq, box)" --nameB "D (rd13 defaults, box)" --out $B/tolcmp64/tol_P_vs_D.json > $B/tolcmp64/tol_P_vs_D.txt 2>&1
say "tolcmp P-D rc $?"
for X in B D; do
  $TC $B/tolcmp64/$X $B/tolcmp64/prod --tag rd13 --after 11.5 --rs 64 --nameA "$X (box)" --nameB "production (qi2_h100_1 H100)" --out $B/tolcmp64/tol_${X}_vs_prod.json > $B/tolcmp64/tol_${X}_vs_prod.txt 2>&1
  say "tolcmp $X-prod rc $?"
done
say "(b) per-call again (final module)"; mkdir -p $B/percall_final
env CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY $B/build/rd13_percall_v1.py $B/percall_final $B/code \
  --w2 $W/orbitals_${WT}_T0013.50.npz --w2-fields $W/collapse_${WT}_fields.npz --iso $I/states_h2/orbitals_${T}_T0005.00.npz \
  --iso-fields $I/run_h2/collapse_${T}_fields.npz --synthetic 8192 > $B/logs/percall_final.log 2>&1
say "(b) per-call rc $?"
say "(a) fixture GPU part again (final module)"
env CUDA_VISIBLE_DEVICES=$GPU nice -n 10 $PY $B/build/rd13_fixture_gate_v1.py --code $B/code --out $B/fixture_final --threads 8 --only G > $B/logs/fixture_final.log 2>&1
say "(a) fixture G rc $?"
say "chain4 done"
