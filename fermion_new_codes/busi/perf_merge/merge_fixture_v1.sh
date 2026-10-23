#!/bin/bash
# merge_fixture_v1.sh (busi, MERGE-1; busi/notes/PERF_MERGE_build_v1.md): self-test arms of engine_collapse_v36nq_rd13mf on the rd13m
# self-test fixture (rd13m_selftest_v1.sh's arguments word for word: 1024 sites, K 10, the reopen2 Newton block with v36m on, pair
# workers 0, fresh from T = 0 to T/r_s 0.125, 3 records, the end slice written), on the box GPU (CUDA_VISIBLE_DEVICES=$GPU, never 0),
# THR threads per process (default 1), nice 10, every engine from the merged code dir CODE (default code16d: engine_fields_v16.py = v16d
# 7a69faad; code16e: v16e 83fe8b2e).
# Arms run concurrently; each arm is "NAME ENGINE [options...]"; --pair-workers N in the options replaces the fixture's 0.
#   GPU=2 THR=1 merge_fixture_v1.sh SET "R0 engine_collapse_v36nq_rd13m.py" "M0 engine_collapse_v36nq_rd13mf.py" ...
set -u
SET=$1; shift
M=${QI2_ROOT}/MERGE_build; PY=${QI2_ROOT}/python
GPU=${GPU:-2}; THR=${THR:-1}; CODE=${CODE:-code16d}; [ "$GPU" = 0 ] && { echo "never cuda:0"; exit 2; }
W=$M/fixture/$SET; mkdir -p $W
NS=1024; RS=$((NS / 32)); ELL=$((RS / 16))
COMMON="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea 10 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-6 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes \
--mix newton --mix-alpha 0.5 --nk-maxiter 30 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 \
--nk-plain-fallback on --nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --PW \
--rs $RS --nsites $NS --ell $ELL --prep-ell 4 --tau $ELL --T-over-rs 0.125 --slices-c128 0.125"
cd $M/$CODE
md5sum engine_collapse_v36nq_rd13mf.py engine_collapse_v36nq_rd13m.py engine_collapse_v36nq_rd13.py engine_collapse_v36nq_rd2.py engine_banded_eigh_v1.py engine_fields_v16.py > $W/md5_inputs.txt
for A in "$@"; do
  set -- $A; nm=$1; eng=$2; shift 2
  opts="$*"; pw="--pair-workers 0"
  case " $opts " in *" --pair-workers "*) pw="";; esac
  mkdir -p $W/$nm/states
  cmd="$PY -u $M/$CODE/$eng ${COMMON/--PW/$pw} --out $W/$nm --tag fx_$nm --slice-dir $W/$nm/states $opts"
  echo "$cmd" > $W/$nm/cmd.txt
  ( echo "[$(date -u +%FT%TZ)] start $nm GPU $GPU THR $THR CODE $CODE" >> $W/$nm/arm.log
    env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR NUMBA_NUM_THREADS=$THR \
        nice -n 10 $cmd > $W/$nm/log.txt 2>&1
    echo "[$(date -u +%FT%TZ)] end $nm rc $?" >> $W/$nm/arm.log ) &
  echo "$nm pid $!" >> $W/pids.txt
done
wait
echo "[$(date -u +%FT%TZ)] set $SET done" >> $W/done.txt
