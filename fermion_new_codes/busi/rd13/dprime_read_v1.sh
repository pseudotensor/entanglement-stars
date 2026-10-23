#!/bin/bash
# dprime_read_v1.sh (busi, analyst, 21 Oct): the D' arm (ReviewA_rd1entry_v1 sec. 3) read by tolcmp_v2 d9b9ee32 UNCHANGED (the entry
# read's copy), X = D' (parent route, threads 4), D = the entry replay's arm D (threads 8), P = the stored ISO_128 nkqh2 production.
# No slice arguments (D' ends at 5.125, D at 5.5). Then dprime_hist_v1.py on B, D, D', P at 5.0625 and 5.125. Card host CPU, nice 10, 8 threads.
set -u
E=${QI2_ROOT}/busi_rd1entry; W=${QI2_ROOT}/busi_dprime; PY=${QI2_ROOT}/python
TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
R=$W/read; mkdir -p $R/X $R/D $R/P
for s in "" _fields _spectra _handoff; do e=npz; [ -z "$s" ] && e=json
  ln -sf $W/out/Dp/collapse_rd13_Dp${s}.$e $R/X/collapse_rd13${s}.$e
  ln -sf $E/out/D/collapse_rd13_D${s}.$e $R/D/collapse_rd13${s}.$e
  ln -sf $E/in/run_h2/collapse_${TAG0}${s}.$e $R/P/collapse_rd13${s}.$e; done
md5sum $E/read/tolcmp_v2.py > $R/tolcmp_md5.txt
env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10 $PY -I $E/read/tolcmp_v2.py $R/X $R/D $R/P --tag rd13 --after 5.0 --rs 128 \
    --names "Dp (parent route; 4 threads; h100-f),D (parent route; 8 threads; h100-f),production (pod6 H100)" --out $R/v2 > $R/tolcmp_v2_dprime.txt 2>&1
echo "read rc $?"
for T in 5.0625 5.125; do
  nice -n 10 $PY -I $W/dprime_hist_v1.py --T $T B=$E/out/B/collapse_rd13_B.json D=$E/out/D/collapse_rd13_D.json Dp=$W/out/Dp/collapse_rd13_Dp.json P=$E/in/run_h2/collapse_${TAG0}.json > $W/hist_B_D_Dp_P_$T.txt 2>&1
done
echo "hist done"
