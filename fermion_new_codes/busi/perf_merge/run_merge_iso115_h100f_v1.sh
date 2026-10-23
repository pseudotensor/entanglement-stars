#!/bin/bash
# run_merge_iso115_h100f_v1.sh (busi, MERGE-1): job (b)'s own start, timing only. ISO_128 nkqh2 from the T0011.50 slice (the first formation
# record; the run json ends at 11.25, so the resume reads the json's records to 11.25 and the slice's state at 11.5) to END (default 11.75:
# 4 records), the production's arguments word for word, on h100-f, the merged engine 248f25f3 (code dir with v16e), the RD-1 venv, 8 threads,
# the peak hook, nvidia-smi sampled; every arm holds the shared h100-f lock ${QI2_ROOT}/busi_gpu.lock (flock on fd 9, the lead's rule
# after the 17:03 collision) for its whole run, the nvidia-smi guard kept. No end slice is written (timing; slices_c128 set past END). Arms, one after the other, GPU free first:
#   D  defaults; AB --eigh banded --fields collocation; AC AB + --nk-cache-host on; Fd --fields collocation; P2 AB + --pair-workers 2
#   usage: run_merge_iso115_h100f_v1.sh "D AB AC Fd P2" [END]
set -u
W=${QI2_ROOT}/busi_merge; PY=${QI2_ROOT}/python; CODE=$W/code
ARMS=${1:-"D AB AC Fd P2"}; END=${2:-11.75}
I=$W/inputs/iso128_h2; TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
SL=$I/states_h2/orbitals_${TAG0}_T0011.50.npz
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=$W/peakhook
exec 9>>${QI2_ROOT}/busi_gpu.lock
for arm in $ARMS; do
  case $arm in
    D) OPT="";; AB) OPT="--eigh banded --fields collocation";; AC) OPT="--eigh banded --fields collocation --nk-cache-host on";;
    Fd) OPT="--fields collocation";; P2) OPT="--eigh banded --fields collocation --pair-workers 2";; *) echo "unknown arm $arm"; exit 2;;
  esac
  O=$W/out/iso115/$arm
  [ -e $O/RUN.txt ] && { echo "$O exists; skipped"; continue; }
  flock 9; echo "$(date -u +%FT%TZ) iso115 $arm: lock held" >> $W/out/iso115_wait.log
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) iso115 $arm waits" >> $W/out/iso115_wait.log; sleep 30; done
  mkdir -p $O/states
  args=$($PY $W/code/rd13_cmd_from_args_v1.py $I/run_h2/collapse_${TAG0}.json out=$O tag=mf115_$arm slice_dir=$O/states slices_c128=40.0 gpu=0)
  cmd="$PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $args --resume-slice $SL --resume-run $I/run_h2/$TAG0 --segment-end-T-over-rs $END $OPT"
  echo "$(date -u +%FT%TZ) iso115 $arm start: $cmd" > $O/RUN.txt
  md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_fields_v16.py $SL >> $O/RUN.txt
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 2000 > $O/smi.csv 2>&1 &
  smi=$!
  ( cd $CODE && $cmd ) > $O/log.txt 2>&1
  rc=$?
  kill $smi
  echo "$(date -u +%FT%TZ) iso115 $arm rc $rc; nvidia-smi peak $(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' $O/smi.csv) MiB" >> $O/RUN.txt
  echo "iso115 $arm rc $rc $(date -u +%FT%TZ)" >> $W/out/iso_done.txt
  flock -u 9; echo "$(date -u +%FT%TZ) iso115 $arm: lock released" >> $W/out/iso115_wait.log
done
