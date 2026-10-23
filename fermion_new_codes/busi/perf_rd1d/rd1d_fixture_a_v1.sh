#!/bin/bash
# rd1d_fixture_a_v1.sh (busi, PERF RD-1d self-test (a)): the defaults of engine_collapse_v36nq_rd13mfd.py against its parent rd13mfb
# 02a1729a on GATECOMP flat's 1024-site fixture (flat_bitwise_v1.sh's COMMON word for word, pair workers 0), on the BOX CPU (no --gpu,
# CUDA_VISIBLE_DEVICES empty), 4 threads (as the stored arm), nice 10, T 0 to 0.0625 (2 records), the end slice written. The parent's arm
# is the stored one (gatecomp/flat/bitwise/C, rd13mfb, the same command, the same box, 4 threads; its slice md5 cf894fea).
# Read by cmp_rd13m_v1.py and the slice's md5.
set -u
R=${QI2_ROOT}/RD1d_build; C=$R/code16d_rd1d; PY=${QI2_ROOT}/python; W=$R/fixture_a; THR=4
[ -e $W ] && { echo "refusing: $W exists"; exit 1; }
mkdir -p $W/D/states
NS=1024; RS=$((NS / 32)); ELL=$((RS / 16))
COMMON="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea 10 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes \
--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --pair-workers 0 \
--rs $RS --nsites $NS --ell $ELL --prep-ell 4 --tau $ELL --T-over-rs 0.0625 --slices-c128 0.0625"
md5sum $C/engine_collapse_v36nq_rd13mfd.py $C/engine_fields_v16.py ${QI2_ROOT}/engine_collapse_v36nq_rd13mfb.py > $W/md5_inputs.txt
cmd="$PY -u $C/engine_collapse_v36nq_rd13mfd.py $COMMON --out $W/D --tag fb_D --slice-dir $W/D/states"
echo "$cmd" > $W/D/cmd.txt
echo "[$(date -u +%FT%TZ)] start D CPU THR $THR" >> $W/D/arm.log
cd $C
env CUDA_VISIBLE_DEVICES= NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR NUMBA_NUM_THREADS=$THR PYTHONDONTWRITEBYTECODE=1 \
    nice -n 10 $cmd > $W/D/log.txt 2>&1
echo "[$(date -u +%FT%TZ)] end D rc $?" >> $W/D/arm.log
S=${QI2_ROOT}/bitwise
nice -n 10 $PY ${QI2_ROOT}/cmp_rd13m_v1.py $S/C fb_C $W/D fb_D --out $W/cmp_rd13m.json > $W/cmp_rd13m.txt 2>&1; echo "cmp rc $?" >> $W/cmp_rd13m.txt
( cd $W/D/states && md5sum * ) > $W/slice_md5.txt; grep -h fb_C $S/slices_md5_deleted.txt >> $W/slice_md5.txt
echo "[$(date -u +%FT%TZ)] done" >> $W/done.txt
