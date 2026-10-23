#!/bin/bash
# J4 (SPEC_final4b_observables.md sec. 4.2): the 3-dimensional box on the exact Painleve-Gullstrand profile (control), L = 48 then L = 64,
# r_s = 8 (kappa a = 1/16), R_m = 2, R_cut = 2.5 r_s, m = 0, t_r = 2 r_s, dt = 1, the holds 0, 12, 24, 48/kappa; one region (the ball
# R < 1.5 r_s, B7) holding the balls R < r_s and R < 1.5 r_s and the shells d = 1, 2, 4; the partner of the 3-dimensional packet at
# omega/kappa = 4 (kappa_j = -1 and +1, j = 1/2); JSON and markdown written by obs_box3d_v2.py (mode j4) after every hold.
# PREPARED, NOT RUN (builder analyst).  Usage on the box, detached:
#   nohup bash launch_obs_J4_v1.sh <gpu> [host budget GB] [extra args] > J4_launcher.log 2>&1 < /dev/null &
# The GPU must hold less than 1 GB and no process at launch (checked before each size); nice 10; 2 host threads.
# Memory plan (B6): Z~ = 2 L^3 x 28832 complex64 per hold on the host (51 GB at L = 48, 121 GB at L = 64); the holds are grouped so that a
# group's Z~ fits the host budget (default 130 GB: groups {48, 24}, {12, 0} at L = 48 and one hold per group at L = 64); the GPU holds one
# chunk's evolution (chunk 512 at L = 48 about 10 GB, chunk 256 at L = 64 about 12 GB), then the slab Gram (C 13.3 GB complex128 plus one
# slab product 6.7 GB), then one sub-block at a time for the eigenvalues (at most 13.3 GB plus the solver's copy); the flat reference and
# the formed blocks stay on the host between uses.
GPU=$1
BUDGET=${2:-130}
shift 2 2>/dev/null
EXTRA="$@"
cd ${QI2_ROOT}/J45
PY=${QI2_ROOT}/python
STATUS=status_J4.txt
mkdir -p J4
gpu_free() {
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v g=$GPU '$1==g{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$GPU '$1==g{print $2}')
  local nproc=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u' | wc -l)
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
for spec in "48 512" "64 256"; do
  set -- $spec
  L=$1; CH=$2
  if ! gpu_free; then echo "J4 L$L not started: GPU $GPU not free at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS; exit 1; fi
  echo "J4 L$L start $(date -u +%Y-%m-%dT%H:%M:%SZ) (gpu $GPU, host budget $BUDGET GB)" >> $STATUS
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/obs_box3d_v2.py j4 --L $L --rs 8 --Rm 2 \
    --rcut-over-rs 2.5 --mass 0 --dt 1 --holds 0 12 24 48 --shells 1 2 4 --chunk $CH --slab 256 --host-budget-gb $BUDGET \
    --omega-over-kappa 4 --R0-over-rs 2.2 --sigma-x 4 --outdir J4 --tag v1 $EXTRA > J4/j4_L${L}_v1.log 2>&1 < /dev/null
  echo "J4 L$L exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS
done
