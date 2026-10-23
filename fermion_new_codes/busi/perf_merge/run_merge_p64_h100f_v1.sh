#!/bin/bash
# run_merge_p64_h100f_v1.sh (busi, MERGE-1): the per-record factor at 2048 sites on the H100. The P64_s4 Newton member's segment (self-test
# (c)'s, qi2_h100_1 production: v36nq + H2, --pair-workers 2) from its T0011.50 slice to 12.5 (8 records), the production's arguments word for
# word, on h100-f, the merged engine 248f25f3 in ${QI2_ROOT}/code (v16e), the RD-1 numba venv, OMP/MKL/OPENBLAS/NUMBA 8, the peak
# hook, nvidia-smi sampled every 2 s. Arms one after the other, each only when nvidia-smi shows no compute process:
#   G0 the defaults;  GA --eigh banded --nk-cache-host on --fields collocation;  GB --eigh banded --fields collocation (no cache).
# Every arm holds the shared h100-f lock ${QI2_ROOT}/busi_gpu.lock (flock on fd 9) for its whole run (from 17:4x UTC).   usage: run_merge_p64_h100f_v1.sh "GA G0 GB" [END]
set -u
W=${QI2_ROOT}/busi_merge; PY=${QI2_ROOT}/python; CODE=$W/code
ARMS=${1:-"GA G0"}; END=${2:-12.5}
I=$W/inputs/p64; TAG0=lad_P64_s4_newton_v36nq_h100_1; SL=$I/states/orbitals_${TAG0}_T0011.50.npz
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=$W/peakhook
exec 9>>${QI2_ROOT}/busi_gpu.lock
for arm in $ARMS; do
  case $arm in G0) OPT="";; GA) OPT="--eigh banded --nk-cache-host on --fields collocation";; GB) OPT="--eigh banded --fields collocation";; *) echo "unknown arm $arm"; exit 2;; esac
  O=$W/out/p64/$arm
  [ -e $O/RUN.txt ] && { echo "$O exists; skipped"; continue; }
  flock 9; echo "$(date -u +%FT%TZ) p64 $arm: lock held" >> $W/out/p64_wait.log
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) p64 $arm waits: a compute process on the GPU" >> $W/out/p64_wait.log; sleep 30; done
  mkdir -p $O/states
  args=$($PY $W/code/rd13_cmd_from_args_v1.py $I/run/collapse_${TAG0}.json out=$O tag=mf_$arm slice_dir=$O/states slices_c128=$END gpu=0)
  cmd="$PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $args --resume-slice $SL --resume-run $I/run/$TAG0 --segment-end-T-over-rs $END $OPT"
  echo "$(date -u +%FT%TZ) p64 $arm start: $cmd" > $O/RUN.txt
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv >> $O/RUN.txt
  md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_banded_eigh_v1.py $CODE/engine_fields_v16.py $SL >> $O/RUN.txt
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 2000 > $O/smi.csv 2>&1 &
  smi=$!
  ( cd $CODE && $cmd ) > $O/log.txt 2>&1
  rc=$?
  kill $smi
  echo "$(date -u +%FT%TZ) p64 $arm rc $rc; nvidia-smi peak $(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' $O/smi.csv) MiB" >> $O/RUN.txt
  echo "p64 $arm rc $rc $(date -u +%FT%TZ)" >> $W/out/iso_done.txt
  flock -u 9; echo "$(date -u +%FT%TZ) p64 $arm: lock released" >> $W/out/p64_wait.log
done
