#!/bin/bash
# fx_evolve_v1.sh (busi, GATECOMP, ReviewA_resview_v1 tests (a) and (b), #405): the restart view's tests on the box's 1024-site fixture,
# the box (nice 10, 4 processes x 2 threads = 8 CPU threads) on box GPU ${GPU:-2} as MERGE-1's fixture ran (about 3 GB; never cuda:0), the
# merged engine 248f25f3 on code16d, a small fixture that converges: r_s/a 16, 512 sites (32 r_s), ell 2, prep-ell 2, K 10, the reopen2
# Newton block, pair workers 0, --eigh dense, ivp fields, the production stop rule (8 s a record; probe fx_probe: converged, kept residual
# 5e-11 to 1e-10). MERGE-1's F1 shape (r_s/a 32, 1024 sites) does not converge at its records (residual 1e-4, the MERGE notes say so) and
# ground at 10 to 30 min a record: two attempts stopped, kept as fx_evolve_cpu_tightstop_stopped and fx_evolve_F1shape_unconverged_stopped.
#  P   fresh tau 2 (tau/r_s 1/8) T 0 -> 2, slices 1, 2                         H   fresh tau 1, T 0 -> 2, slice 1
#  V2  view tau=1 from P's T 1 slice to 2                                     V4  view tau=0.5 from P's T 1 slice to 2
#  (a) model consistency, one record each at T_slice + 1e-6 a (a step that propagates nothing at the floor): Ct (view tau=1e-6),
#      Cm (view tau=1e-6 + mix=plain), Cs (view tau=1e-6 + site_fields=dense): each record against P's T 1 record.
set -u
K=${QI2_ROOT}/kits; CODE=${QI2_ROOT}/code16d; ENG=$CODE/engine_collapse_v36nq_rd13mf.py
PY=${QI2_ROOT}/python; O=$K/fx_evolve; mkdir -p $O
GPU=${GPU:-2}; [ "$GPU" = 0 ] && { echo "never cuda:0"; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 NUMPY_MADVISE_HUGEPAGE=0 CUDA_VISIBLE_DEVICES=$GPU
B="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea 10 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes \
--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --pair-workers 0 \
--rs 16 --nsites 512 --ell 2 --prep-ell 2 --T-over-rs 2.0 --eigh dense --gpu 0"
PLAIN=$(echo "$B" | sed -e 's/--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 --nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --pair-workers 0/--mix plain --mix-alpha 0.5 --iter-cache drop --prep-memory lean/')
run() { local n=$1; shift; mkdir -p $O/$n/states; echo "$*" > $O/$n/args.txt
  ( cd $CODE && nice -n 10 $PY -u $ENG "$@" --out $O/$n --slice-dir $O/$n/states --tag fx > $O/$n/log.txt 2>&1 ); echo "$(date -u +%FT%TZ) $n rc $?" >> $O/DONE.txt; }
waitrow() {  # until P's json and fields npz hold the T 1 record and its slice exists
  until [ -e $O/P/states/orbitals_fx_T0001.00.npz ] && $PY $K/code/resume_view_v1.py $O/P/fx $O/_probe --set tau=1 --at 1.0 --reason probe --check-only > /dev/null 2>&1; do sleep 20; done; }
: > $O/DONE.txt
run P $B --tau 2 --slices-c128 1.0,2.0 & PP=$!
run H $B --tau 1 --slices-c128 1.0 &
waitrow
SL=$O/P/states/orbitals_fx_T0001.00.npz; cp $SL $O/slice_P_T0001.00.npz; SL=$O/slice_P_T0001.00.npz   # P keeps running: a copy for the views
cp -r $O/P $O/P_at_slice_snapshot 2>/dev/null; P0=$O/P_at_slice_snapshot
mkv() { $PY $K/code/resume_view_v1.py $P0/fx $O/$1_view "${@:2}" --at 1.0 --reason "ReviewA_resview_v1 test" > $O/$1_view.log 2>&1; }
mkv V2 --set tau=1; mkv V4 --set tau=0.5; mkv Ct --set tau=1e-6; mkv Cm --set tau=1e-6 --set mix=plain; mkv Cs --set tau=1e-6 --set site_fields=dense
RV="--resume-slice $SL"
run V2 $B --tau 1 $RV --resume-run $O/V2_view/fx --segment-end-T-over-rs 2.0 --slices-c128 2.0 &
run V4 $B --tau 0.5 $RV --resume-run $O/V4_view/fx --segment-end-T-over-rs 2.0 --slices-c128 2.0 &
E1=$($PY -c "print(repr(1.0 + 1e-6/16))")
( wait_p() { while kill -0 $PP 2>/dev/null; do sleep 20; done; }; wait_p
  run Ct $B --tau 1e-6 $RV --resume-run $O/Ct_view/fx --segment-end-T-over-rs $E1 --slices-c128 40
  run Cm $PLAIN --tau 1e-6 $RV --resume-run $O/Cm_view/fx --segment-end-T-over-rs $E1 --slices-c128 40
  run Cs $(echo "$B" | sed 's/--site-fields averaged/--site-fields dense/') --tau 1e-6 $RV --resume-run $O/Cs_view/fx --segment-end-T-over-rs $E1 --slices-c128 40 ) &
wait
echo "$(date -u +%FT%TZ) ALL DONE" >> $O/DONE.txt
