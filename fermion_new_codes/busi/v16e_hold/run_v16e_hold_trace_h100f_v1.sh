#!/bin/bash
# run_v16e_hold_trace_h100f_v1.sh (busi; ReviewA_v16e_v1.md sec. 2 (i), supporting run): Fd's 5.1875 record alone on h100-f, from arm
# Fd5's own T0005.12 slice and json (MERGE-1: Fd's engine, code dir and arguments to 5.125, its 2 records bitwise Fd's; read only), with
# Fd's command word for word except the paths, the slice clock and the segment end (5.1875), in this lane's code16e copy, with the trace
# hook (tracehook/sitecustomize.py: every field solve's route to a jsonl; nothing computed changes). The record is checked against Fd's
# own 5.1875 record byte for byte afterwards. Shared lock, free card, nvidia-smi before, during (2 s; compute apps every 5 s) and after.
#   usage: run_v16e_hold_trace_h100f_v1.sh [ARM] [WAIT_FOR_N_DONE]
set -u
W=${QI2_ROOT}/busi_v16e_hold; PY=${QI2_ROOT}/python; CODE=$W/code16e
ARM=${1:-dON_trace}; NWAIT=${2:-4}
F5=${QI2_ROOT}/Fd5
O=$W/out/$ARM
[ -e $O/RUN.txt ] && { echo "$O exists"; exit 1; }
while [ "$(cat $W/out/done.txt 2>/dev/null | wc -l)" -lt "$NWAIT" ]; do sleep 60; done
mkdir -p $O/states
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
base=$(head -1 $F5/RUN.txt | sed 's/^.*Fd5 start: //')
args=$(echo "$base" | sed -e 's#^[^ ]* -B -u [^ ]*engine_collapse_v36nq_rd13mf.py ##' \
  -e "s#--out [^ ]*#--out $O#" -e "s#--tag [^ ]*#--tag vh_$ARM#" -e "s#--slice-dir [^ ]*#--slice-dir $O/states#" \
  -e "s#--slices-c128 [^ ]*#--slices-c128 5.1875#" -e "s#--resume-slice [^ ]*#--resume-slice $F5/states/orbitals_mf_Fd5_T0005.12.npz#" \
  -e "s#--resume-run [^ ]*#--resume-run $F5/mf_Fd5#" -e "s#--segment-end-T-over-rs [^ ]*#--segment-end-T-over-rs 5.1875#")
echo "$(date -u +%FT%TZ) $ARM: waiting for the lock" >> $W/out/wait.log
exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; echo "$(date -u +%FT%TZ) $ARM: lock held" >> $W/out/wait.log
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) $ARM: lock held, a compute process on the GPU" >> $W/out/wait.log; sleep 30; done
cmd="$PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $args"
{ echo "$(date -u +%FT%TZ) $ARM start (QI2_FIELDS_FLAT unset = on; trace hook): $cmd"; md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_banded_eigh_v1.py $CODE/engine_fields_v16.py $F5/states/orbitals_mf_Fd5_T0005.12.npz $W/tracehook/sitecustomize.py; } > $O/RUN.txt
{ date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/smi_before.txt 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader -lms 2000 > $O/smi.csv 2>&1 &
smi=$!
( while true; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 5; done ) > $O/smi_apps.txt 2>&1 &
apps=$!
( cd $CODE && PYTHONPATH=$W/tracehook QI2_TRACE_DIR=$O/trace nice -n 10 $cmd ) > $O/log.txt 2>&1
rc=$?
kill $smi $apps
{ date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/smi_after.txt 2>&1
echo "$(date -u +%FT%TZ) $ARM rc $rc; nvidia-smi peak $(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' $O/smi.csv) MiB" >> $O/RUN.txt
echo "$ARM rc $rc $(date -u +%FT%TZ)" >> $W/out/done.txt
flock -u 9; exec 9>&-; echo "$(date -u +%FT%TZ) $ARM: lock released" >> $W/out/wait.log
