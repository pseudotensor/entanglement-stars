#!/bin/bash
# rd1entry_read_v2.sh (busi, analyst, 21 Oct): the RD-1 entry replay (h100-f ${QI2_ROOT}/busi_rd1entry) read by tolcmp_v2 (ruling v2 sec. 2),
# X = B (--eigh banded), D = defaults, P = the stored ISO_128 nkqh2 production (pod6 H100). Runs on the card host (CPU, nice 10, 8 threads).
set -u
W=${QI2_ROOT}/busi_rd1entry; PY=${QI2_ROOT}/python
TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
R=$W/read; mkdir -p $R/X $R/D $R/P
for X in B D; do d=$R/X; [ $X = D ] && d=$R/D; for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $W/out/$X/collapse_rd13_${X}${s}.$e $d/collapse_rd13${s}.$e; done; done
for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json; ln -sf $W/in/run_h2/collapse_${TAG0}${s}.$e $R/P/collapse_rd13${s}.$e; done
SX=$(ls $W/out/B/states/*.npz | tail -1); SD=$(ls $W/out/D/states/*.npz | tail -1)
PC="RD-1 notes sec. 3.2 (box RD1_build/percall/rd13_percall_v1.txt): the per-call banded against dense test on ISO_128 nkqh2 T0005.00's own matrices (2n 8192, kappa 1 and 10): eigenvalues 7.1e-15, residuals 1.9e-13, orthonormality 2.6e-13, tracked set equal to the stored masks, route banded, 0 fallbacks (ruling v2 sec. 5)"
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY -I $R/tolcmp_v2.py $R/X $R/D $R/P --tag rd13 --after 5.0 --rs 128 \
    --slice-x $SX --slice-d $SD --names "B (--eigh banded; h100-f),D (defaults; h100-f),production (pod6 H100)" --percall "$PC" --out $R/v2 > $R/tolcmp_v2_entry.txt 2>&1
echo "read rc $?"
