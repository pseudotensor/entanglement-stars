#!/bin/bash
# rd1entry_replay_v1.sh (busi, analyst): RD-1 entry gate for job (b), h100-f. rd13_replay_v1.sh's arm, paths moved to the card:
# ISO_128 nkqh2 from its T0005.00 complex128 slice to 5.5 (8 records), the production's json args word for word, 8 threads, nice 10.
# Arms in sequence: D (rd13 defaults) then B (rd13 --eigh banded). Each arm starts only if nvidia-smi shows no compute process.
set -u
W=${QI2_ROOT}/busi_rd1entry; PY=${QI2_ROOT}/python; END=5.5
TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
SL=$W/in/states_h2/orbitals_${TAG0}_T0005.00.npz; RUN=$W/in/run_h2
ENG=engine_collapse_v36nq_rd13.py
for ARM in D B; do
  case $ARM in D) X="";; B) X="--eigh banded";; esac
  out=$W/out/$ARM; mkdir -p $out/states
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  if [ "$n" != "0" ]; then echo "[$(date -u +%FT%TZ)] $ARM NOT started: $n compute processes on the GPU" >> $W/out/chain.log; nvidia-smi >> $W/out/chain.log; exit 3; fi
  args=$($PY $W/build/rd13_cmd_from_args_v1.py $RUN/collapse_${TAG0}.json out=$out tag=rd13_$ARM slice_dir=$out/states slices_c128=$END gpu=0)
  cmd="$PY -u $W/code/$ENG $args --resume-slice $SL --resume-run $RUN/$TAG0 --segment-end-T-over-rs $END $X"
  echo "$cmd" > $out/cmd.txt
  md5sum $W/code/$ENG $W/code/engine_banded_eigh_v1.py $W/code/engine_fields_v16.py $RUN/collapse_${TAG0}.json > $out/md5_inputs.txt 2>&1
  echo "aee9560e79eedb290e9fc1f1972c8f95  $SL (checked 14:08 UTC, hard link of busi_merge's copy)" >> $out/md5_inputs.txt
  echo "[$(date -u +%FT%TZ)] start $ARM" | tee -a $out/arm.log >> $W/out/chain.log
  cd $W/code
  env CUDA_VISIBLE_DEVICES=0 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
      nice -n 10 $cmd > $out/log.txt 2>&1
  rc=$?
  echo "[$(date -u +%FT%TZ)] end $ARM rc $rc" | tee -a $out/arm.log >> $W/out/chain.log
done
echo "[$(date -u +%FT%TZ)] ALL DONE" >> $W/out/chain.log
