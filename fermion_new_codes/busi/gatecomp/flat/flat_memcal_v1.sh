#!/bin/bash
# flat_memcal_v1.sh (busi, GATECOMP flat build; coordinator's (3): calibrate the memory law for K 15/16 at cache off): three seg0-shaped arms
# on box GPU 2 (CUDA_VISIBLE_DEVICES=2; never cuda:0), one at a time, at 2048 sites (r_s/a 128, tau 8, ell 4, prep-ell 4), the staged
# lines' route (--eigh banded --fields collocation --nk-cache-host off), the reopen2 Newton block, pair workers 0, T 0 to 0.0625 (the
# preparation and the records 0, 0.03125, 0.0625), no slice written (the slice clock 9 is never reached). The peak hook prints torch's
# max_memory_allocated/reserved at exit; nvidia-smi every 2 s. Arms: Mw14 = the well (--dust-set well) at K 14 (ties the law to W256_s4's
# measured seg0 peak 58.61 GiB allocated at 4096 sites, K 14); Mt15 = --prep flatbox taper W 1 at K 15; Mh16 = --prep flatbox hard at K 16.
set -u
F=${QI2_ROOT}/flat; C=$F/code16d_flat; X=engine_collapse_v36nq_rd13mfb.py; PY=${QI2_ROOT}/python; D=$F/memcal
[ -e $D ] && { echo "refusing: $D exists"; exit 1; }; mkdir -p $D
[ -n "$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader)" ] && { echo "GPU 2 busy"; exit 2; }
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$F/peakhook
B=$(sed -e 's/--nk-cache-host on/--nk-cache-host off/' -e 's/--pair-workers 2/--pair-workers 0/' -e 's/--rs 256 --nsites 4096 --ell 8 --prep-ell 4 --tau 16/--rs 128 --nsites 2048 --ell 4 --prep-ell 4 --tau 8/' \
  -e 's/--slices-c128 [^ ]*/--slices-c128 9.0000/' -e 's/--T-over-rs 8/--T-over-rs 0.0625/' -e 's/ --prep flatbox//' $F/kit/lines_flat/F256_s8.args)
for A in "Mw14 14 " "Mt15 15 --prep flatbox --prep-edge taper --prep-taper 1.0" "Mh16 16 --prep flatbox"; do
  set -- $A; nm=$1; k=$2; shift 2; o=$D/$nm; mkdir -p $o/states
  args="$(echo "$B" | sed -e "s/--kappa-sea [0-9]*/--kappa-sea $k/" -e "s#--slice-dir [^ ]*#--slice-dir $o/states#" -e "s#--out [^ ]*#--out $o#" -e "s#--tag [^ ]*#--tag mc_$nm#") $*"
  echo "$args" > $o/args.txt
  nvidia-smi -i 2 --query-gpu=timestamp,memory.used --format=csv,noheader -lms 2000 > $o/smi.csv 2>&1 & S=$!
  t0=$(date +%s); ( cd $C && nice -n 10 $PY -u $C/$X $args ) > $o/log.txt 2>&1; rc=$?; kill $S
  echo "$nm K $k rc $rc ($(( $(date +%s) - t0 )) s): $(grep peakhook $o/log.txt) smi max $(awk -F, '{gsub(/ MiB/,"",$2); if ($2+0>m) m=$2+0} END {print m}' $o/smi.csv) MiB"
done
echo DONE
