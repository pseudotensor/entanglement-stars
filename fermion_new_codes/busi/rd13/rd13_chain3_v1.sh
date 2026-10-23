#!/bin/bash
# rd13_chain3_v1.sh (busi, analyst): after rd13_chain_v1.sh ends (arms D to 5.1875 and P), with the OOM-safe module: arm B2 (--eigh banded
# --jvp sternheimer) from the T0005.00 slice straight to 5.5 (8 records; arm B hit a device OOM at 5.1875 on the 48 GB A6000), arm Dx
# (D continued from its own 5.1875 slice to 5.5), the per-call test (b) and the fixture's GPU smoke again with the final module, then
# cmp_rd13_replay_v2 (my budget and REVIEWA_perf_tolerance_ruling_v1 sec. 4) and reviewer A's tolcmp_v1.
set -u
B=${QI2_ROOT}/RD1_build; PY=${QI2_ROOT}/python; export GPU=${GPU:-2}
I=${QI2_ROOT}/iso128_h2; T=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
W=${QI2_ROOT}/T0013.50
WT=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_r2b
L=$B/logs/chain3_v1.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $L; }
say "chain3 waiting for chain 1"
until grep -q "chain done" $B/logs/chain_v1.log; do sleep 60; done
cp -p $B/build/engine_banded_eigh_v1.py $B/code/engine_banded_eigh_v1.py
md5sum $B/code/engine_banded_eigh_v1.py $B/code/engine_collapse_v36nq_rd13.py >> $L
say "B2 start"; bash $B/build/rd13_replay_v1.sh B2 engine_collapse_v36nq_rd13.py 5.5 --eigh banded --jvp sternheimer; say "B2 rc $?"
say "Dx start"; bash $B/build/rd13_extend_v1.sh Dx D engine_collapse_v36nq_rd13.py 5.5; say "Dx rc $?"
cd $B
nice -n 10 $PY $B/build/cmp_rd13_replay_v2.py --replay $B/replay --prod $I/run_h2/$T --arms Dx,B2 --arm-d Dx --arm-b B2 --parent P --t0 5.0 --out $B/replay/cmp_rd13_replay_v2 > $B/logs/cmp_v2.log 2>&1
say "cmp v2 rc $?"
mkdir -p $B/tolcmp/B2 $B/tolcmp/Dx $B/tolcmp/prod
for X in B2 Dx; do for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $B/replay/$X/collapse_rd13_${X}${s}.$e $B/tolcmp/$X/collapse_rd13${s}.$e; done; done
for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $I/run_h2/collapse_${T}${s}.$e $B/tolcmp/prod/collapse_rd13${s}.$e; done
SB=$(ls $B/replay/B2/states/*.npz | tail -1); SD=$(ls $B/replay/Dx/states/*.npz | tail -1)
TC="nice -n 10 $PY ${QI2_ROOT}/tolcmp_v1.py"
$TC $B/tolcmp/B2 $B/tolcmp/Dx --tag rd13 --after 5.0 --rs 128 --nameA "B2 (banded + sternheimer, box A6000)" --nameB "Dx (defaults, box A6000)" --slice-a $SB --slice-b $SD --out $B/tolcmp/tol_B2_vs_Dx.json > $B/tolcmp/tol_B2_vs_Dx.txt 2>&1
say "tolcmp B2-Dx rc $?"
for X in B2 Dx; do
  $TC $B/tolcmp/$X $B/tolcmp/prod --tag rd13 --after 5.0 --rs 128 --nameA "$X (box)" --nameB "production (pod6 H100)" --out $B/tolcmp/tol_${X}_vs_prod.json > $B/tolcmp/tol_${X}_vs_prod.txt 2>&1
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
say "chain3 done"
