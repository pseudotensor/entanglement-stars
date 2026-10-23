#!/bin/bash
# run_v16e_hold_v_h100f_v1.sh (busi; ReviewA_v16e_v1.md erratum v3 sec. 2 (v); busi/notes/PERF_v16e_hold_readings_v1.md):
# the four-arm record on h100-f, free card. ISO_128 nkqh2 from the stored T0005.00 slice (aee9560e) through 5.1875 (3 records:
# 5.0625, 5.125, 5.1875), merged engine engine_collapse_v36nq_rd13mf.py 248f25f3 in this lane's own copy of code16e
# (engine_fields_v16.py = v16e 83fe8b2e, banded module 73ee46fa), the production's arguments word for word through
# rd13_cmd_from_args_v1.py (out, tag, slice_dir, slices_c128 = END, gpu 0 overridden), --fields collocation on every arm.
#   dON   dense eigh,  QI2_FIELDS_FLAT=on   (Fd as run)
#   dOFF  dense eigh,  QI2_FIELDS_FLAT=off  (v16d's route exactly)
#   bON   --eigh banded, FLAT on            (AB as run)
#   bOFF  --eigh banded, FLAT off
#   suffix _pw2: the same with --pair-workers 2 (replaces the production's 0; argparse keeps the last occurrence)
# Each arm takes ${QI2_ROOT}/busi_gpu.lock (shared with the other busi agents) and starts only when nvidia-smi shows no compute
# process. nvidia-smi is logged in full before and after each arm; during it, the GPU every 2 s and the compute-process list every
# 5 s (so any other process on the card during the arm is on record). The RD-1 numba venv; OMP/MKL/OPENBLAS/NUMBA 8; nice 10.
#   usage: run_v16e_hold_v_h100f_v1.sh "dON dOFF bON bOFF" [END]
set -u
W=${QI2_ROOT}/busi_v16e_hold; PY=${QI2_ROOT}/python; CODE=$W/code16e
ARMS=${1:-"dON dOFF bON bOFF"}; END=${2:-5.1875}
I=${QI2_ROOT}/iso128_h2; TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
SL=$I/states_h2/orbitals_${TAG0}_T0005.00.npz
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=$W/peakhook
for arm in $ARMS; do
  base=${arm%_pw2}; PW=""; [ "$base" != "$arm" ] && PW="--pair-workers 2"
  case $base in
    dON)  OPT="--fields collocation"; FLAT=on;;
    dOFF) OPT="--fields collocation"; FLAT=off;;
    bON)  OPT="--eigh banded --fields collocation"; FLAT=on;;
    bOFF) OPT="--eigh banded --fields collocation"; FLAT=off;;
    *) echo "unknown arm $arm"; exit 2;;
  esac
  OPT="$OPT $PW"
  O=$W/out/$arm
  [ -e $O/RUN.txt ] && { echo "$O exists; skipped"; continue; }
  mkdir -p $O/states
  args=$($PY $CODE/rd13_cmd_from_args_v1.py $I/run_h2/collapse_${TAG0}.json out=$O tag=vh_$arm slice_dir=$O/states slices_c128=$END gpu=0)
  echo "$(date -u +%FT%TZ) $arm: waiting for the lock" >> $W/out/wait.log
  exec 9>>${QI2_ROOT}/busi_gpu.lock; flock 9; echo "$(date -u +%FT%TZ) $arm: lock held" >> $W/out/wait.log
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) $arm: lock held, a compute process on the GPU" >> $W/out/wait.log; sleep 30; done
  cmd="$PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $args --resume-slice $SL --resume-run $I/run_h2/$TAG0 --segment-end-T-over-rs $END $OPT"
  { echo "$(date -u +%FT%TZ) $arm start (QI2_FIELDS_FLAT=$FLAT): $cmd"; md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_banded_eigh_v1.py $CODE/engine_fields_v16.py $SL; } > $O/RUN.txt
  { date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/smi_before.txt 2>&1
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader -lms 2000 > $O/smi.csv 2>&1 &
  smi=$!
  ( while true; do echo "$(date -u +%FT%TZ) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 5; done ) > $O/smi_apps.txt 2>&1 &
  apps=$!
  ( cd $CODE && QI2_FIELDS_FLAT=$FLAT nice -n 10 $cmd ) > $O/log.txt 2>&1
  rc=$?
  kill $smi $apps
  { date -u +%FT%TZ; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $O/smi_after.txt 2>&1
  echo "$(date -u +%FT%TZ) $arm rc $rc; nvidia-smi peak $(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' $O/smi.csv) MiB" >> $O/RUN.txt
  echo "$arm rc $rc $(date -u +%FT%TZ)" >> $W/out/done.txt
  flock -u 9; exec 9>&-; echo "$(date -u +%FT%TZ) $arm: lock released" >> $W/out/wait.log
  sleep 20   # let the other busi agents' waiters take the lock between arms
done
