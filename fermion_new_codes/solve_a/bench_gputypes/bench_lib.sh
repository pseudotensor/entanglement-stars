# bench_lib.sh (solve bench_gputypes v1): sourced by each kit's run.sh. Sets GPU, NT, the thread env (the H1 limits), PY, the run dir.
# Inputs: $1 GPU index (nvidia-smi numbering; CUDA_DEVICE_ORDER=PCI_BUS_ID makes CUDA agree), $2 thread count (default nproc - 2 capped by the
# cgroup quota: min(nproc, floor(quota)) - 2, at least 1), env BENCH_PY (else KIT/../../envs/$ENVNAME/bin/python), BENCH_RESULTS (else
# KIT/../../results), BENCH_TAG (an extra word in the run dir name). The run dir: $BENCH_RESULTS/<gpu slug>/<class>/<UTC stamp><tag>; refused if present.
GPU=${1:?GPU index}; NT=${2:-}
if [ -z "$NT" ]; then
  NP=$(nproc --all 2>/dev/null || nproc); Q=$(awk '$1 != "max" {printf "%d", $1 / $2}' /sys/fs/cgroup/cpu.max 2>/dev/null)
  [ -n "$Q" ] && [ "$Q" -lt "$NP" ] && NP=$Q; NT=$((NP - 2)); [ "$NT" -lt 1 ] && NT=1
fi
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT NUMPY_MADVISE_HUGEPAGE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
PY=${BENCH_PY:-$KIT/../../envs/$ENVNAME/bin/python}
[ -x "$PY" ] || { echo "no python $PY (set BENCH_PY, or build the envs with setup_env.sh)"; exit 2; }
$PY -c "import sys,numpy,scipy,torch; ok=(numpy.__version__,scipy.__version__,torch.__version__)==tuple(sys.argv[1:4]); print('python', sys.version.split()[0], numpy.__version__, scipy.__version__, torch.__version__, 'stack OK' if ok else 'STACK MISMATCH'); sys.exit(0 if ok else 1)" $STACK || { echo "refusing: the python stack differs from the kit's ($STACK)"; exit 2; }
SLUG=$(nvidia-smi -i $GPU --query-gpu=name --format=csv,noheader | sed 's/NVIDIA //; s/[^A-Za-z0-9]\{1,\}/_/g; s/_$//')
RES=${BENCH_RESULTS:-$KIT/../../results}
D=$RES/$SLUG/$CLASS/$(date -u +%Y%m%dT%H%M%SZ)${BENCH_TAG:+_$BENCH_TAG}
test ! -e $D || { echo "refusing: $D exists"; exit 2; }
mkdir -p $D
( cd $KIT && md5sum -c --quiet MD5SUMS ) || { echo "refusing: kit md5 check failed"; exit 2; }
echo "kit $CLASS gpu $GPU ($SLUG) NT $NT py $PY run dir $D"
