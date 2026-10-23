#!/bin/bash
# ab_v36x_m3c.sh ARM (busi 9ae0d5c4, #299 PERF ISO-A2/A3): one arm of the H100 A/B v36r vs v36x on M3c's Newton line (ready_jobs_v3c.sh m3c_nk_r),
# from m3c_a's certified 2.5833 slice, three Newton records 2.667, 2.750, 2.833 (--segment-end-T-over-rs 2.8333333333, slice at 2.8333).
# One card, one arm at a time (never two arms at once: the walls are the measurement). Arms:
#   A     v36r 7b6d0f4f, the production line (the reference)
#   P     A + --pair-workers 2                                   (ISO-A2: a launch argument)
#   X0    v36x 8768f708, defaults                                (G0: must be bitwise A)
#   XC    v36x --nk-jvp-cache-gib 8
#   XD    v36x --nk-jvp-defer on
#   XCD   v36x --nk-jvp-cache-gib 8 --nk-jvp-defer on
#   XCDP  XCD + --pair-workers 2
# Kit: $K holds the ladder code set (${QI2_ROOT}/code_v36d with engine_collapse_v36r.py, engine_collapse_v36c.py, engine_pairsolve_v1.py)
# plus engine_collapse_v36x.py, engine_collapse_v36n_selftest.py and cmp_v36x_ab.py; $IN holds m3c_a's four run files and its 2.5833 slice.
# Never overwrites an arm's dir; never kills anything.
set -u
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=${THREADS:-8} MKL_NUM_THREADS=${THREADS:-8} OPENBLAS_NUM_THREADS=${THREADS:-8}
ARM=${1:?arm}
K=${K:-${QI2_ROOT}/code_v36d}; PY=${PY:-${QI2_ROOT}/python}
IN=${IN:-${QI2_ROOT}/A}; AB=${AB:-${QI2_ROOT}/ab_v36x}
TA=lad_ISO_M3c_epA_m0p1_rs96_n3072_R04_k12_tau8_ell4_prep4_A_R2_avg_plain_v36n
RUNJ=$(ls $IN/run/collapse_$TA.json 2>/dev/null || ls $IN/collapse_$TA.json)
SL=$(ls $IN/states/orbitals_${TA}_T0002.58.npz 2>/dev/null || ls $IN/orbitals_${TA}_T0002.58.npz)
md5ok() { [ "$(md5sum $1 | cut -c1-32)" = $2 ] || { echo "md5 mismatch on $1 (want $2): refused"; exit 1; }; }
case $ARM in
  A) E=engine_collapse_v36r.py; X="";;
  P) E=engine_collapse_v36r.py; X="--pair-workers 2";;
  X0) E=engine_collapse_v36x.py; X="";;
  XC) E=engine_collapse_v36x.py; X="--nk-jvp-cache-gib ${GIB:-8}";;
  XD) E=engine_collapse_v36x.py; X="--nk-jvp-defer on";;
  XCD) E=engine_collapse_v36x.py; X="--nk-jvp-cache-gib ${GIB:-8} --nk-jvp-defer on";;
  XCDP) E=engine_collapse_v36x.py; X="--nk-jvp-cache-gib ${GIB:-8} --nk-jvp-defer on --pair-workers 2";;
  cmp) shift; for B_ in "$@"; do $PY $K/cmp_v36x_ab.py --selftest-lib $K/engine_collapse_v36n_selftest.py --a $AB/A --b $AB/$B_ > $AB/cmp_A_$B_.txt 2>&1; echo "$B_: $(tail -1 $AB/cmp_A_$B_.txt)"; grep "segment wall" $AB/cmp_A_$B_.txt; done; exit 0;;
  *) echo "unknown arm $ARM"; exit 2;;
esac
md5ok $K/engine_collapse_v36r.py 7b6d0f4f2db82a98780bdca78166f61b
md5ok $K/engine_collapse_v36x.py 8768f708197c0f4f3be25066cf570b4a
md5ok $K/engine_pairsolve_v1.py 7e0f41b5220fa4cf2f77ff9364935d66
md5ok $K/engine_collapse_v36c.py 3e34c0a11876eaded51f632da05b311b
[ -s "$SL" ] && [ -s "$RUNJ" ] || { echo "no m3c_a 2.5833 slice or run json under $IN: refused"; exit 1; }
D=$AB/$ARM; test ! -e $D || { echo "refused: $D exists"; exit 1; }
mkdir -p $D/snap $D/run $D/states
P_=${RUNJ%.json}; cp $P_.json ${P_}_fields.npz ${P_}_handoff.npz ${P_}_spectra.npz $D/snap/ || exit 1
$PY -c "
import json,sys
p=sys.argv[1]; d=json.load(open(p)); d['args']['mix']='newton'
d['_busi_note']='busi ab_v36x_m3c: snapshot copy; args.mix set to newton for the v34 resume check (original untouched)'
json.dump(d,open(p,'w'))" $D/snap/collapse_$TA.json
M3CB="--m 0.1 --g 0.25 --rs 96 --R0-over-rs 4 --reading A --nsites 3072 --kappa-sea 12 --reference R2 --site-fields averaged --prep-ell 4 --ell 4 --tau 8 --T-over-rs 40 --gpu 0 --field-kernel-cache --v36-fixes --dust-set isotropic --iter-cache drop"
NK="--nk-krylov 40 --nk-halvings 6 --nk-inner 15 --nk-jvp exact --nk-line-search halving --nk-set resolve --nk-maxiter 30 --nk-restarts 2"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > $D/smi.csv 2>/dev/null & SMI=$!
cd $D/run
echo "$(date -u +%FT%TZ) $ARM start: $E $X" | tee -a $AB/STATUS
$PY $K/$E $M3CB --mix newton $NK --end-above ${END:-0.002} --nk-plain-fallback on $X \
    --slices-c128 2.8333333333 --slice-dir $D/states --out $D/run --tag ab_$ARM --resume-slice $SL --resume-run $D/snap/$TA \
    --segment-end-T-over-rs 2.8333333333 > $D/log.txt 2>&1
rc=$?
kill $SMI 2>/dev/null
echo "$(date -u +%FT%TZ) $ARM end rc $rc" | tee -a $AB/STATUS
exit $rc
