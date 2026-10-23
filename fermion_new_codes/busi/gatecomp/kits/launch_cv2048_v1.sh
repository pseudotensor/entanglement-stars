#!/bin/bash
# launch_cv2048_v1.sh (busi, GATECOMP kit (i), #400/#402): the chain validation, ISO at r_s/a 128 on a 2048-site chain (16 r_s), fresh
# from T = 0 to its own end (--T-over-rs 13; the engine's end rule 32.1(3) is expected to end it near job (b)'s 12.0625 stall, read
# independently). ON THE AWS CARD compute-host, operated by solve. usage: launch_cv2048_v1.sh [dry]
# Gates (any failure: no launch): the kit's MD5SUMS; the args md5 (LINES.tsv); engine 248f25f3, fields v16d 7a69faad, banded 73ee46fa in
# CODE; numba importable in the venv; no output json for the tag; the GPU holds no compute process; dry: the engine's own argparse only.
# Route (ReviewA_s3b_v1 sec. 1): merged engine on code16d, --eigh banded --fields collocation, --nk-cache-host off, pair workers 2.
set -u
K=${K:-${QI2_ROOT}/cv2048}; CODE=${CODE:-${QI2_ROOT}/code_mf}; PY=${PY:-${QI2_ROOT}/python}
MODE=${1:-run}; A=$K/lines/CV2048.args
die() { echo "REFUSE: $*" | tee -a $K/LAUNCH_REFUSED.txt; exit 9; }
( cd $K && md5sum -c --quiet MD5SUMS ) || die "kit MD5SUMS"
[ "$(md5sum < $A | cut -c1-32)" = "$(awk '$1=="CV2048"{print $3}' $K/lines/LINES.tsv)" ] || die "args md5"
[ "$(md5sum < $CODE/engine_collapse_v36nq_rd13mf.py | cut -c1-32)" = 248f25f35e48a88a84b102ddd99a5713 ] || die "engine md5"
[ "$(md5sum < $CODE/engine_fields_v16.py | cut -c1-32)" = 7a69faadfad4a1e824a41107f474c69c ] || die "fields md5 (v16d required; v16e HELD)"
[ "$(md5sum < $CODE/engine_banded_eigh_v1.py | cut -c1-32)" = 73ee46facc09b8cb77f095b2803a5c8e ] || die "banded md5"
$PY -c "import numba" 2>/dev/null || die "numba not importable in $PY (the RD-1 side venv)"
TAG=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--tag"{p=1}'); OUT=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--out"{p=1}')
SD=$(tr ' ' '\n' < $A | awk 'p{print; exit} $0=="--slice-dir"{p=1}')
[ -e $OUT/collapse_$TAG.json ] && die "output json exists for $TAG"
mkdir -p $OUT $SD $K/logs
CUDA_VISIBLE_DEVICES= $PY $K/code/prep_dry_v1.py parse $CODE engine_collapse_v36nq_rd13mf.py $A $K/logs > $K/logs/dryparse.log 2>&1 || die "dry parse"
[ $MODE = dry ] && { echo "DRY OK: $TAG; $(tail -1 $K/logs/dryparse.log)"; exit 0; }
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && die "GPU holds a compute process"
NV=$(nproc); TH=$((NV - 2))
ENVS="NUMPY_MADVISE_HUGEPAGE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMEXPR_NUM_THREADS=$TH NUMBA_NUM_THREADS=$TH"
cd $CODE
env -u H2_PW_ONE -u H2_LOCAL_SPLINE $ENVS $PY -B -u $CODE/engine_collapse_v36nq_rd13mf.py $(cat $A) > $K/logs/$TAG.log 2>&1 < /dev/null &
P=$!; echo $P > $K/engine.pid
setsid nohup bash -c "while kill -0 $P 2>/dev/null; do echo \$(date -u +%FT%TZ),\$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits); sleep 30; done" > $K/logs/gpumem.csv 2>&1 < /dev/null &
{ echo "START $(date -u +%FT%TZ) pid $P"; echo "card $(hostname) vCPU $NV threads $TH"; echo "cwd $CODE"; echo "env $ENVS"; echo "cmd $PY $CODE/engine_collapse_v36nq_rd13mf.py $(cat $A)"
  md5sum $CODE/engine_collapse_v36nq_rd13mf.py $CODE/engine_fields_v16.py $CODE/engine_banded_eigh_v1.py $A; } > $K/LAUNCH.txt
cat $K/LAUNCH.txt
