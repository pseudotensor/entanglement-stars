#!/bin/bash
# run_merge_iso_h100f_v1.sh (busi, MERGE-1; busi/notes/PERF_MERGE_build_v1.md, self-test (d)): the ISO_128 nkqh2 replay at 4096 sites,
# K 10, on h100-f (H100 80GB), from the stored T0005.00 complex128 slice to END (default 5.5: the 8 records 5.0625 ... 5.5), the
# production's arguments from its json word for word (rd13_cmd_from_args_v1.py; gpu 0, the paths moved), the end slice written; engine
# engine_collapse_v36nq_rd13mf.py (248f25f3) from ${QI2_ROOT}/code (engine_fields_v16.py = v16e 83fe8b2e, the banded module
# 73ee46fa), the RD-1 numba venv's python (numba over np1263), OMP/MKL/OPENBLAS/NUMBA 8, the peak hook on PYTHONPATH (prints torch's
# CUDA peak at exit), nvidia-smi sampled every 2 s during the arm. Arms one after the other; each starts only when nvidia-smi shows no
# compute process.
#   D0  the defaults (v36nq 8549c542's route)
#   A   --eigh banded --nk-cache-host on --fields collocation              (every adopted option)
#   A2  A + --pair-workers 2                                               (SYM-N1)
#   F   --fields collocation --nk-cache-host on                            (RD-2 alone; the cache bitwise)
#   usage: run_merge_iso_h100f_v1.sh "A D0 A2 F" [END]
set -u
W=${QI2_ROOT}/busi_merge; PY=${QI2_ROOT}/python; CODE=$W/code
ARMS=${1:-"A D0 A2 F"}; END=${2:-5.5}
I=$W/inputs/iso128_h2; TAG0=lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq
SL=$I/states_h2/orbitals_${TAG0}_T0005.00.npz
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE
export PYTHONPATH=$W/peakhook
for arm in $ARMS; do
  case $arm in
    D0) OPT="";;
    A)  OPT="--eigh banded --nk-cache-host on --fields collocation";;
    A2) OPT="--eigh banded --nk-cache-host on --fields collocation --pair-workers 2";;
    F)  OPT="--fields collocation --nk-cache-host on";;
    *) echo "unknown arm $arm"; exit 2;;
  esac
  O=$W/out/iso/$arm
  [ -e $O/RUN.txt ] && { echo "$O exists; skipped"; continue; }
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do echo "$(date -u +%FT%TZ) $arm waits: a compute process on the GPU" >> $W/out/iso_wait.log; sleep 60; done
  mkdir -p $O/states
  args=$($PY $W/code/rd13_cmd_from_args_v1.py $I/run_h2/collapse_${TAG0}.json out=$O tag=mf_$arm slice_dir=$O/states slices_c128=$END gpu=0)
  # --pair-workers in OPT replaces the production's value (argparse keeps the last occurrence)
  cmd="$PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $args --resume-slice $SL --resume-run $I/run_h2/$TAG0 --segment-end-T-over-rs $END $OPT"
  echo "$(date -u +%FT%TZ) $arm start: $cmd" > $O/RUN.txt
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv >> $O/RUN.txt
  md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_banded_eigh_v1.py $CODE/engine_fields_v16.py $SL >> $O/RUN.txt
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 2000 > $O/smi.csv 2>&1 &
  smi=$!
  ( cd $CODE && $cmd ) > $O/log.txt 2>&1
  rc=$?
  kill $smi
  echo "$(date -u +%FT%TZ) $arm rc $rc; nvidia-smi peak $(awk -F', ' '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' $O/smi.csv) MiB" >> $O/RUN.txt
  echo "$arm rc $rc $(date -u +%FT%TZ)" >> $W/out/iso_done.txt
done
