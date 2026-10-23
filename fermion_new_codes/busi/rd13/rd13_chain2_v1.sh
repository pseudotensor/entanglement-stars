#!/bin/bash
# rd13_chain2_v1.sh (busi, analyst): after rd13_chain_v1.sh ends, the 8-record extension (REVIEWA_perf_tolerance_ruling_v1 sec. 4): Bx from
# B's 5.1875 slice and Dx from D's, to T/r_s 5.5 (8 records 5.0625 to 5.5 in each json), then cmp_rd13_replay_v2 (my budget and the
# ruling's gate) and reviewer A's tolcmp_v1 (the ruling's own comparator) on Bx against Dx and each against the production.
set -u
B=${QI2_ROOT}/RD1_build; PY=${QI2_ROOT}/python; export GPU=${GPU:-2}
I=${QI2_ROOT}/iso128_h2; T=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
L=$B/logs/chain2_v1.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> $L; }
say "chain2 waiting for chain 1"
until grep -q "chain done" $B/logs/chain_v1.log; do sleep 60; done
say "Bx start"; bash $B/build/rd13_extend_v1.sh Bx B engine_collapse_v36nq_rd13.py 5.5; say "Bx rc $?"
say "Dx start"; bash $B/build/rd13_extend_v1.sh Dx D engine_collapse_v36nq_rd13.py 5.5; say "Dx rc $?"
cd $B
nice -n 10 $PY $B/build/cmp_rd13_replay_v2.py --replay $B/replay --prod $I/run_h2/$T --arms Dx,Bx --arm-d Dx --arm-b Bx --parent P --t0 5.0 --out $B/replay/cmp_rd13_replay_v2 > $B/logs/cmp_v2.log 2>&1
say "cmp v2 rc $?"
mkdir -p $B/tolcmp/Bx $B/tolcmp/Dx
for X in Bx Dx; do for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $B/replay/$X/collapse_rd13_${X}${s}.$e $B/tolcmp/$X/collapse_rd13${s}.$e; done; done
SB=$(ls $B/replay/Bx/states/*.npz | tail -1); SD=$(ls $B/replay/Dx/states/*.npz | tail -1)
TC="nice -n 10 $PY ${QI2_ROOT}/tolcmp_v1.py"
$TC $B/tolcmp/Bx $B/tolcmp/Dx --tag rd13 --after 5.0 --rs 128 --nameA "Bx (banded + sternheimer, box A6000)" --nameB "Dx (defaults, box A6000)" --slice-a $SB --slice-b $SD --out $B/tolcmp/tol_Bx_vs_Dx.json > $B/tolcmp/tol_Bx_vs_Dx.txt 2>&1
say "tolcmp Bx-Dx rc $?"
mkdir -p $B/tolcmp/prod
for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $I/run_h2/collapse_${T}${s}.$e $B/tolcmp/prod/collapse_rd13${s}.$e; done
for X in Bx Dx; do
  $TC $B/tolcmp/$X $B/tolcmp/prod --tag rd13 --after 5.0 --rs 128 --nameA "$X (box)" --nameB "production (pod6 H100)" --out $B/tolcmp/tol_${X}_vs_prod.json > $B/tolcmp/tol_${X}_vs_prod.txt 2>&1
  say "tolcmp $X-prod rc $?"
done
say "chain2 done"
