#!/bin/bash
# dprime_replay_v1.sh (busi, analyst, 21 Oct): the pre-registered D' arm of ReviewA_rd1entry_v1 sec. 3, on h100-f only.
# D' = the parent route (engine_collapse_v36nq_rd13.py 7d362033, rd13 defaults, bitwise the parent) from the ISO_128 nkqh2 T0005.00 slice
# aee9560e, the production's json args word for word through rd13_cmd_from_args_v1.py (the entry replay's overrides: out, tag, slice_dir,
# slices_c128 = segment end, gpu 0), 2 records (5.0625, 5.125). ONE perturbation against arm D of rd1entry_replay_v1.sh: the thread count
# OMP/MKL/OPENBLAS 4 instead of 8. Everything else as arm D (same code dir, venv, run inputs, env, nice 10).
# Slice: $E/in/states_h2 is the same inode as ${QI2_ROOT}/states_h2 (aee9560e), the path arm D used.
# Run as: flock ${QI2_ROOT}/busi_gpu.lock bash dprime_replay_v1.sh   (the lock is held for the whole arm); nvidia-smi logged before/during/after.
set -u
E=${QI2_ROOT}/busi_rd1entry; W=${QI2_ROOT}/busi_dprime; PY=${QI2_ROOT}/python; END=5.125; NT=4
TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
SL=$E/in/states_h2/orbitals_${TAG0}_T0005.00.npz; RUN=$E/in/run_h2
ENG=engine_collapse_v36nq_rd13.py; ARM=Dp
out=$W/out/$ARM; mkdir -p $out/states; L=$W/out/chain.log
echo "[$(date -u +%FT%TZ)] lock taken (pid $$)" >> $L
for i in $(seq 1 120); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$n" = "0" ] && break
  echo "[$(date -u +%FT%TZ)] waiting: $n compute processes on the GPU" >> $L; sleep 30
done
[ "$n" != "0" ] && { echo "[$(date -u +%FT%TZ)] NOT started: GPU busy" >> $L; nvidia-smi >> $L; exit 3; }
{ echo "== before $(date -u +%FT%TZ)"; nvidia-smi; } > $out/nvidia_smi.log
args=$($PY $E/build/rd13_cmd_from_args_v1.py $RUN/collapse_${TAG0}.json out=$out tag=rd13_$ARM slice_dir=$out/states slices_c128=$END gpu=0)
cmd="$PY -u $E/code/$ENG $args --resume-slice $SL --resume-run $RUN/$TAG0 --segment-end-T-over-rs $END"
echo "$cmd" > $out/cmd.txt
echo "threads OMP/MKL/OPENBLAS=$NT (arm D: 8)" >> $out/cmd.txt
md5sum $E/code/$ENG $E/code/engine_banded_eigh_v1.py $E/code/engine_fields_v16.py $RUN/collapse_${TAG0}.json $SL > $out/md5_inputs.txt 2>&1
echo "[$(date -u +%FT%TZ)] start $ARM" | tee -a $out/arm.log >> $L
( while true; do sleep 120; { echo "== during $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; } >> $out/nvidia_smi.log; done ) &
MON=$!
cd $E/code
env CUDA_VISIBLE_DEVICES=0 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT \
    nice -n 10 $cmd > $out/log.txt 2>&1
rc=$?
kill $MON
{ echo "== after $(date -u +%FT%TZ)"; nvidia-smi; } >> $out/nvidia_smi.log
echo "[$(date -u +%FT%TZ)] end $ARM rc $rc" | tee -a $out/arm.log >> $L
cd $out && md5sum collapse_* states/* cmd.txt md5_inputs.txt log.txt > MD5SUMS_Dp.txt
echo "[$(date -u +%FT%TZ)] ALL DONE" >> $L
