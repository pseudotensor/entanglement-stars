#!/bin/bash
# rd1d_tolcmp_v1.sh (busi, PERF RD-1d): the gate reads by tolcmp_v4 (98d9f2c2, busi/code/readings/perf_tolerance_v1) on the box CPU, nice 10,
# 8 threads, reads only. Staging as run_v4_all.sh (links collapse_g*). Gates:
#   fixture_a: X = D (rd13mfd 59a641d2 defaults), D = C (rd13mfb 02a1729a, the stored arm), P = C (no production for a fixture) -- 2 records;
#   replay_U : X = U3 (--eigh-module v2 --nk-stage-release on, 59a641d2), D = U1 (the parent's path), P = U2 (the first build's option arm);
#   replay_C : X = C2 (the fix under the cap), D = U1, P = U3 (uncapped fix).
# usage: rd1d_tolcmp_v1.sh fixture_a|replay_U|replay_C
set -u
R=${QI2_ROOT}/RD1d_build; O=$R/tolcmp; TC=${QI2_ROOT}/tolcmp_v4.py; PY=${QI2_ROOT}/python
E="env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 nice -n 10"
PC="RD-1d (b): v2's dense route equals v1's bitwise at 2n 8192 with and without pressure (unit_b/rd1d_unit_v1.txt, 8/8 PASS); the stage release is an exact copy (v36m HostCache); no per-call route changes"
stage() { mkdir -p $1; for s in "" _fields _spectra _handoff _s11; do x=npz; [ -z "$s" ] && x=json; [ -e $2$s.$x ] && ln -sfn $2$s.$x $1/collapse_g$s.$x; done; }
g=$1; S=$O/$g/stage; mkdir -p $O/$g
case $g in
  fixture_a) F=$R/fixture_a; C=${QI2_ROOT}/C
     stage $S/X $F/D/collapse_fb_D; stage $S/D $C/collapse_fb_C; stage $S/P $C/collapse_fb_C
     $E $PY -I $TC $S/X $S/D $S/P --tag g --rs 32 --min-records 2 --names "rd13mfd,rd13mfb,rd13mfb" --percall "$PC" --out $O/$g/v4 > $O/$g/tolcmp_v4.txt 2>&1;;
  replay_U) W=$R/replay_c
     stage $S/X $W/U3/collapse_rc_U3; stage $S/D $W/U1/collapse_rc_U1; stage $S/P $W/U2/collapse_rc_U2
     $E $PY -I $TC $S/X $S/D $S/P --tag g --rs 64 --min-records 2 --names "U3,U1,U2" --percall "$PC" --out $O/$g/v4 \
        --slice-x $W/U3/states/orbitals_rc_U3_T0000.06.npz --slice-d $W/U1/states/orbitals_rc_U1_T0000.06.npz --slice-p $W/U2/states/orbitals_rc_U2_T0000.06.npz > $O/$g/tolcmp_v4.txt 2>&1;;
  replay_C) W=$R/replay_c
     stage $S/X $W/C2/collapse_rc_C2; stage $S/D $W/U1/collapse_rc_U1; stage $S/P $W/U3/collapse_rc_U3
     $E $PY -I $TC $S/X $S/D $S/P --tag g --rs 64 --min-records 2 --names "C2,U1,U3" --percall "$PC" --out $O/$g/v4 \
        --slice-x $W/C2/states/orbitals_rc_C2_T0000.06.npz --slice-d $W/U1/states/orbitals_rc_U1_T0000.06.npz --slice-p $W/U3/states/orbitals_rc_U3_T0000.06.npz > $O/$g/tolcmp_v4.txt 2>&1;;
esac
echo "rc $?" >> $O/$g/tolcmp_v4.txt; tail -3 $O/$g/tolcmp_v4.txt
