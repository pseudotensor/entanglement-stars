#!/bin/bash
# run_s8half_card_v1.sh (busi, inbox #459 (2); ReviewA_s8end_v1 sec. 2): C_half450, the halved-step control of F256_s8 (tau 8 a from the
# T0004.50 slice, view tau=8, to T/rs 5.0). ON THE CARD (one H100 80 GB of the owner's 8x H100 box), from the unpacked s8half_kit_v1.
#   bash run_s8half_card_v1.sh check|go       env: GPU (required: the physical GPU index, 1, 2 or 3), ROOT (default ${QI2_ROOT}/busi_s8half),
#                                             V (venv; default $ROOT/venv, built from the pins if absent), THR (threads, default 14),
#                                             STOP_AT (self-stop, default 2026-10-22T20:30:00Z)
# check: kit MD5SUMS; the GPU is an H100 80 GB with no compute process; host RAM >= 96 GiB (the pinned Newton caches, --nk-cache-host on);
#        >= 150 GB free under ROOT (7 slices of 16 GB); installs code, peakhook, view; the venv (python 3.11, torch 2.4.1+cu124, numpy 1.26.3,
#        scipy 1.14.1, numba 0.61.2, llvmlite 0.44.0) and its imports + CUDA; the slice md5 (fa4e0ec3; push_slice_s8half_v1.sh puts it there);
#        resolves the args (@ROOT@) and runs the engine's own argparse on them (prep_dry parse equivalent: the engine with --help-free parse).
# go:    check, then the engine (nohup, CUDA_VISIBLE_DEVICES=$GPU, --gpu 0), nvidia-smi log, and the SELF-STOP: at STOP_AT the engine is sent
#        TERM (KILL after 120 s), by PID only. Writes $ROOT/RC when the engine exits (rc, or "selfstop"). The box's pull_s8half_box_v1.sh copies
#        every slice and the run files back md5-checked as they land; nothing here copies off.
set -u
MODE=${1:?check|go}; KIT=$(cd "$(dirname "$0")" && pwd); GPU=${GPU:?set GPU to the physical index}
ROOT=${ROOT:-${QI2_ROOT}/busi_s8half}; V=${V:-$ROOT/venv}; PY=$V/bin/python; THR=${THR:-14}; STOP_AT=${STOP_AT:-2026-10-22T20:30:00Z}
T8=lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb
ENG=engine_collapse_v36nq_rd13mfb.py; CODE=$ROOT/code16d_flat
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ROOT/RUN_LOG.txt; }
mkdir -p $ROOT/parent $ROOT/run $ROOT/states $ROOT/logs
cd $KIT && md5sum -c --quiet MD5SUMS || { echo "kit MD5SUMS FAIL"; exit 2; }; log "kit MD5SUMS OK ($(wc -l < MD5SUMS) files)"
Q=$(nvidia-smi -i $GPU --query-gpu=name,memory.total --format=csv,noheader); log "GPU $GPU: $Q"
echo "$Q" | grep -q "H100" && [ "$(nvidia-smi -i $GPU --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc 0-9)" -ge 79000 ] || { log "not an H100 80 GB"; exit 3; }
[ -z "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader)" ] || { log "GPU $GPU has a compute process: not starting"; exit 3; }
MEM=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); DISK=$(df -BG --output=avail $ROOT | tail -1 | tr -dc 0-9)
log "host RAM available $MEM GiB, $ROOT free $DISK GB"; [ $MEM -ge 96 ] || { log "host RAM below 96 GiB"; exit 3; }; [ $DISK -ge 150 ] || { log "disk below 150 GB"; exit 3; }
[ -e $CODE ] || cp -a $KIT/code16d_flat $CODE; ( cd $CODE && md5sum -c --quiet $KIT/MD5SUMS_code16d_flat.txt ) || { log "code dir differs"; exit 4; }
mkdir -p $ROOT/peakhook $ROOT/view; cp -n $KIT/peakhook/sitecustomize.py $ROOT/peakhook/; cp -n $KIT/view/* $ROOT/view/
( cd $ROOT/view && for f in $KIT/view/*; do cmp -s $f $(basename $f) || exit 1; done ) || { log "view in place differs from the kit"; exit 4; }
if [ ! -x $PY ]; then
  PYB=$(command -v python3.11 || command -v python3); log "venv from $PYB ($($PYB --version 2>&1))"
  $PYB -m venv $V && $PY -m pip install -q --upgrade pip && $PY -m pip install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.4.1 \
    && $PY -m pip install -q -r $KIT/requirements_flat_v1.txt || { log "venv build failed"; exit 5; }
fi
CUDA_VISIBLE_DEVICES=$GPU $PY - <<'PYX' || { log "imports/pins/CUDA FAIL"; exit 5; }
import sys, numpy, scipy, torch, numba, llvmlite
from numba import cuda
print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__, "torch", torch.__version__, torch.version.cuda,
      "numba", numba.__version__, "llvmlite", llvmlite.__version__, "torch.cuda", torch.cuda.is_available(), "numba.cuda", cuda.is_available())
assert torch.cuda.is_available() and cuda.is_available()
assert numpy.__version__ == "1.26.3" and scipy.__version__ == "1.14.1" and numba.__version__ == "0.61.2" and llvmlite.__version__ == "0.44.0" and torch.__version__.startswith("2.4.1")
PYX
SL=$ROOT/parent/orbitals_${T8}_T0004.50.npz; [ -s $SL ] || { log "no slice $SL (push it first: push_slice_s8half_v1.sh on the box)"; exit 6; }
if [ ! -e $ROOT/parent/SLICE_OK ]; then ( cd $ROOT && md5sum -c --quiet $KIT/SLICE_MD5.txt ) || { log "slice md5 FAIL"; exit 6; }; touch $ROOT/parent/SLICE_OK; fi; log "slice md5 fa4e0ec3 OK"
A=$ROOT/C_half450.args; sed "s#@ROOT@#$ROOT#g" $KIT/C_half450.args.tmpl > $A; log "args resolved: $A md5 $(md5sum < $A | cut -c1-32) (template $(md5sum < $KIT/C_half450.args.tmpl | cut -c1-32))"
( cd $CODE && $PY - $CODE/$ENG $A <<'PYX' ) || { log "the engine's argparse refused the line"; exit 7; }
import argparse, runpy, sys
eng, af = sys.argv[1], sys.argv[2]; o = argparse.ArgumentParser.parse_args
def pa(self, a=None, ns=None):
    r = o(self, a, ns)
    if any(q.dest == "tag" for q in self._actions):
        print("PARSED %d args; tau %s; resume_run %s; segment_end %s; nk_cache_host %s; source_continuation %s" % (len(vars(r)), r.tau, r.resume_run, r.segment_end_T_over_rs, r.nk_cache_host, r.source_continuation)); sys.exit(0)
    return r
argparse.ArgumentParser.parse_args = pa; sys.path.insert(0, "."); sys.argv = [eng] + open(af).read().split(); runpy.run_path(eng, run_name="__main__")
PYX
[ "$MODE" = go ] || { log "check done; 'go' launches"; exit 0; }
[ -e $ROOT/run/log.txt ] && { log "refused: $ROOT/run/log.txt exists (a launch already ran here; new ROOT for a new launch)"; exit 8; }
export OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR NUMEXPR_NUM_THREADS=$THR NUMBA_NUM_THREADS=$THR NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset QI2_FIELDS_ROUTE QI2_FIELDS_WARM QI2_FIELDS_FLAT H2_PW_ONE H2_LOCAL_SPLINE; export PYTHONPATH=$ROOT/peakhook CUDA_VISIBLE_DEVICES=$GPU
cp $KIT/LABEL.txt $ROOT/LABEL.txt
nohup bash -c "
  nvidia-smi -i $GPU --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -lms 5000 >> $ROOT/logs/smi.csv 2>&1 & SMI=\$!
  ( cd $CODE && exec $PY -B -u $CODE/$ENG \$(cat $A) ) > $ROOT/run/log.txt 2>&1 < /dev/null & P=\$!; echo \$P > $ROOT/run/pid
  echo \"\$(date -u +%FT%TZ) C_half450 engine pid \$P on GPU $GPU\" >> $ROOT/RUN_LOG.txt
  ( S=\$(date -d $STOP_AT +%s); while kill -0 \$P 2>/dev/null; do [ \$(date +%s) -ge \$S ] && { echo \"\$(date -u +%FT%TZ) SELF-STOP $STOP_AT: TERM \$P\" >> $ROOT/RUN_LOG.txt; echo selfstop > $ROOT/SELFSTOP; kill -TERM \$P; for i in \$(seq 24); do kill -0 \$P 2>/dev/null || break; sleep 5; done; kill -0 \$P 2>/dev/null && kill -KILL \$P; break; }; sleep 30; done ) &
  wait \$P; rc=\$?; kill \$SMI; [ -e $ROOT/SELFSTOP ] && echo \"selfstop rc \$rc\" > $ROOT/RC || echo \"rc \$rc\" > $ROOT/RC
  echo \"\$(date -u +%FT%TZ) C_half450 engine exit: \$(cat $ROOT/RC)\" >> $ROOT/RUN_LOG.txt
" > $ROOT/logs/runner.out 2>&1 < /dev/null &
sleep 5; log "launched (runner pid $!); tail -f $ROOT/run/log.txt"
