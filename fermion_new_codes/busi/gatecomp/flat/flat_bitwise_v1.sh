#!/bin/bash
# flat_bitwise_v1.sh (busi, GATECOMP flat build, test (a)): the defaults of engine_collapse_v36nq_rd13mfb.py (02a1729a) against the parent
# engine_collapse_v36nq_rd13mf.py (248f25f3) on MERGE-1's 1024-site fixture (merge_fixture_v1.sh's COMMON word for word, pair workers 0),
# BOTH ON THE BOX CPU (no --gpu, CUDA_VISIBLE_DEVICES empty), 4 threads each, nice 10, T 0 to T/r_s 0.0625 (2 records), the end slice written.
# Read by cmp_prefix_v1.py (records every key but walls and labels; fields, spectra, handoff npz rows) and a slice array_equal.
set -u
F=${QI2_ROOT}/flat; C=$F/code16d_flat; PY=${QI2_ROOT}/python; W=$F/bitwise; THR=${THR:-4}
[ -e $W ] && { echo "refusing: $W exists"; exit 1; }
mkdir -p $W
NS=1024; RS=$((NS / 32)); ELL=$((RS / 16))
COMMON="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea 10 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes \
--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --pair-workers 0 \
--rs $RS --nsites $NS --ell $ELL --prep-ell 4 --tau $ELL --T-over-rs 0.0625 --slices-c128 0.0625"
cd $C; md5sum engine_collapse_v36nq_rd13mf.py engine_collapse_v36nq_rd13mfb.py engine_fields_v16.py > $W/md5_inputs.txt
for A in "P engine_collapse_v36nq_rd13mf.py" "C engine_collapse_v36nq_rd13mfb.py"; do
  set -- $A; nm=$1; eng=$2; mkdir -p $W/$nm/states
  cmd="$PY -u $C/$eng $COMMON --out $W/$nm --tag fb_$nm --slice-dir $W/$nm/states"
  echo "$cmd" > $W/$nm/cmd.txt
  ( echo "[$(date -u +%FT%TZ)] start $nm CPU THR $THR" >> $W/$nm/arm.log
    env CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR NUMBA_NUM_THREADS=$THR PYTHONDONTWRITEBYTECODE=1 \
        nice -n 10 $cmd > $W/$nm/log.txt 2>&1
    echo "[$(date -u +%FT%TZ)] end $nm rc $?" >> $W/$nm/arm.log ) &
  echo "$nm pid $!" >> $W/pids.txt
done
wait
echo "[$(date -u +%FT%TZ)] done" >> $W/done.txt
