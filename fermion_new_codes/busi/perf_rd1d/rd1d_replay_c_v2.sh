#!/bin/bash
# rd1d_replay_c_v2.sh (busi, PERF RD-1d self-test (c), the smaller-fixture form; v2: the engine rebuilt as 59a641d2 after U2 (stage 1 kept
# through stage 2 by a closure cycle; v36m/pinned set by the release), PHASE V = U3 (U2's options on 59a641d2) against v1's U1; PHASE C as v1): F256_s4's line (args word for word but the size, the
# span, the stop rule and the pair workers) at 1024 sites (r_s/a 64, tau 4: F256_s4's 4096 / 256 / 16 scaled by 1/4), K 15, flat taper
# W 0.5, --nk-cache-host off, --source-continuation two-stage, with the stop rule made unreachable (--stop-fields 1e-12, --nk-maxiter 3) so
# that the record T/r_s 0.0625 runs the direct solve, stage 1, stage 2 and the direct solve's repeated confirming step: the phase F256_s4
# died in at record 0.375. Each arm under rd1d_memtrace_v1.py (the peak allocated per phase). One box GPU (GPU env, never 0), 2 threads,
# shared with a running job: one arm at a time, each capped (U: 12 GiB, the coordinator's ceiling).
#   U1 = rd13mfd --eigh banded (module v1, no release: the parent's path)   U2 = U1 + --eigh-module v2 --nk-stage-release on
#   then C1, C2 = the same under the cap CAP (GiB; set from U1/U2's traces by rd1d_trace_read_v1.py): C1 must die as F256_s4 did, C2 not.
# usage: GPU=1 PHASE=U|V|C [CAP=x] rd1d_replay_c_v2.sh
set -u
R=${QI2_ROOT}/RD1d_build; C=$R/code16d_rd1d; PY=${QI2_ROOT}/python; W=$R/replay_c; THR=2
GPU=${GPU:?}; PHASE=${PHASE:?}
[ "$GPU" = 0 ] && { echo "never cuda:0"; exit 1; }
mkdir -p $W
BASE="--m 0.2 --g 0.25 --R0-over-rs 4 --reading A --kappa-sea 15 --reference R2 --half-offset 0 --collar-policy flag --support-rel 1e-14 --gpu 0 \
--site-fields averaged --stop-E 1e-7 --stop-fields 1e-12 --maxit 20 --field-rtol 1e-12 --field-atol 1e-16 --delta-rtol-every 10 --commutator-every 1 \
--streams 0 --m1-simpson-steps auto --c-LZ 1.0 --window-max 4 --s11-collars 8,16 --s11-stop 1 --field-kernel-cache --v36-fixes --T-over-rs 0.0625 \
--rs 64 --nsites 1024 --ell 4 --prep-ell 4 --tau 4 --dust-set well --prep flatbox --prep-edge taper --prep-taper 0.5 --mix newton --mix-alpha 0.5 \
--nk-maxiter 3 --nk-inner 15 --nk-krylov 40 --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-restarts 2 --nk-plain-fallback on \
--nk-host-release on --nk-set resolve --iter-cache drop --prep-memory lean --end-above 0.001 --source-continuation two-stage --pair-workers 0 \
--slices-c128 0.0625 --fields collocation --eigh banded --jvp eigenbasis --nk-cache-host off"
arm() {   # name, cap or -, extra options
  local nm=$1 cap=$2; shift 2
  local o=$W/$nm; [ -e $o ] && { echo "refusing: $o exists"; return 1; }
  mkdir -p $o/states
  local capo=""; [ "$cap" != "-" ] && capo="--cap-gib $cap"
  echo "$BASE --out $o --slice-dir $o/states --tag rc_$nm $*" > $o/args.txt
  echo "[$(date -u +%FT%TZ)] start $nm GPU $GPU cap $cap" >> $o/arm.log
  ( cd $C; env CUDA_VISIBLE_DEVICES=$GPU NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR NUMBA_NUM_THREADS=$THR PYTHONDONTWRITEBYTECODE=1 \
      nice -n 10 $PY -u $R/code/rd1d_memtrace_v1.py --engine-dir $C --engine engine_collapse_v36nq_rd13mfd.py --trace $o/memtrace.txt $capo -- \
      $BASE --out $o --slice-dir $o/states --tag rc_$nm "$@" > $o/log.txt 2>&1; echo "[$(date -u +%FT%TZ)] end $nm rc $?" >> $o/arm.log )
}
if [ "$PHASE" = V ]; then
  md5sum $C/engine_collapse_v36nq_rd13mfd.py $C/engine_banded_eigh_v1.py $C/engine_banded_eigh_v2.py $C/engine_fields_v16.py $R/code/rd1d_memtrace_v1.py > $W/md5_inputs_V.txt
  arm U3 ${UCAP:-12} --eigh-module v2 --nk-stage-release on
  nice -n 10 $PY ${QI2_ROOT}/cmp_rd13m_v1.py $W/U1 rc_U1 $W/U3 rc_U3 --slices-a $W/U1/states --slices-b $W/U3/states --out $W/cmp_U1_U3.json > $W/cmp_U1_U3.txt 2>&1; echo "cmp rc $?" >> $W/cmp_U1_U3.txt
  echo "[$(date -u +%FT%TZ)] V done" >> $W/done.txt
elif [ "$PHASE" = U ]; then
  md5sum $C/engine_collapse_v36nq_rd13mfd.py $C/engine_banded_eigh_v1.py $C/engine_banded_eigh_v2.py $C/engine_fields_v16.py $R/code/rd1d_memtrace_v1.py > $W/md5_inputs.txt
  arm U1 ${UCAP:-12}     # one arm at a time on the shared card, each under the coordinator's 12 GiB ceiling
  arm U2 ${UCAP:-12} --eigh-module v2 --nk-stage-release on
  nice -n 10 $PY ${QI2_ROOT}/cmp_rd13m_v1.py $W/U1 rc_U1 $W/U2 rc_U2 --slices-a $W/U1/states --slices-b $W/U2/states --out $W/cmp_U1_U2.json > $W/cmp_U1_U2.txt 2>&1; echo "cmp rc $?" >> $W/cmp_U1_U2.txt
  echo "[$(date -u +%FT%TZ)] U done" >> $W/done.txt
else
  CAP=${CAP:?}
  md5sum $C/engine_collapse_v36nq_rd13mfd.py $C/engine_banded_eigh_v1.py $C/engine_banded_eigh_v2.py > $W/md5_inputs_C.txt
  arm C1 $CAP
  arm C2 $CAP --eigh-module v2 --nk-stage-release on
  nice -n 10 $PY ${QI2_ROOT}/cmp_rd13m_v1.py $W/U1 rc_U1 $W/C2 rc_C2 --slices-a $W/U1/states --slices-b $W/C2/states --out $W/cmp_U1_C2.json > $W/cmp_U1_C2.txt 2>&1; echo "cmp rc $?" >> $W/cmp_U1_C2.txt
  echo "[$(date -u +%FT%TZ)] C done" >> $W/done.txt
fi
