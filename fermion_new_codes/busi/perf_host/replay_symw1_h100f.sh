#!/bin/bash
# replay_symw1_h100f.sh (busi SYM-W1, inbox #370/#373): the W2 prim continuation (R3b v36v_anc) replayed from its mirrored T0038.00 slice to 39.0
# (8 records) under engine_collapse_v36w2 (3155abe4) on h100-f's H100 (GPU 0), the two arms one after the other on the same device: eigh
# (--hbar-prop eigh --hbar-prop-log on: v36v's route, its propagation block timed) then cheb (--hbar-prop chebyshev). Production arguments from the
# mirrored json's args (w2_cmd_from_args.py) with gpu 0, out/slice_dir under ${QI2_ROOT}/replay<arm>, T_over_rs 39.0, slices_c128 39.0;
# OMP/MKL/OPENBLAS 8 as the production launcher. Inputs md5-checked against the mirror's MD5SUMS before the start.
set -e
PY=${QI2_ROOT}/python
W=${QI2_ROOT}/busi_symw1; CODE=$W/code/kit
TAG=lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f_v36p_v36v_anc
M=$W/mirror/T0038.00; SLICE=$M/orbitals_${TAG}_T0038.00.npz; RRUN=$M/$TAG
( cd $M && md5sum -c --quiet MD5SUMS ) || { echo "input md5 FAILED"; exit 1; }
test "$(md5sum < $CODE/engine_collapse_v36w2.py | cut -c1-8)" = 3155abe4 || { echo "engine md5 FAILED"; exit 1; }
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for ARM in eigh cheb; do
  case $ARM in eigh) EXTRA="--hbar-prop eigh --hbar-prop-log on";; cheb) EXTRA="--hbar-prop chebyshev";; esac
  OUT=$W/replay/$ARM
  test ! -e $OUT || { echo "refusing: $OUT exists"; exit 1; }
  mkdir -p $OUT/states
  ARGS=$($PY -I $CODE/w2_cmd_from_args.py $M/collapse_$TAG.json gpu=0 out=$OUT tag=$TAG slice_dir=$OUT/states T_over_rs=39.0 slices_c128=39.0000000000)
  echo "$ARGS $EXTRA --resume-slice $SLICE --resume-run $RRUN" > $OUT/CMD.txt
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > $OUT/GPU_before.txt
  echo "$(date -u +%FT%TZ) start $ARM" >> $W/replay/chain.log
  ( cd $OUT && nice -n 10 $PY $CODE/engine_collapse_v36w2.py $ARGS $EXTRA --resume-slice $SLICE --resume-run $RRUN > $OUT/run.log 2>&1 < /dev/null & echo $! > $OUT/pid; wait $!; echo $? > $OUT/rc ) || true
  echo "$(date -u +%FT%TZ) end $ARM rc $(cat $OUT/rc)" >> $W/replay/chain.log
done
