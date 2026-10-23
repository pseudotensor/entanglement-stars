#!/bin/bash
# J4 (SPEC_final4b_observables.md secs. 4.2 and 8.4): the 3-dimensional box on the exact Painleve-Gullstrand profile (control), r_s = 8
# (kappa a = 1/16), R_m = 2, m = 0, t_r = 2 r_s, dt = 1, the holds 0, 12, 24, 48/kappa, the compact C^2 quintic taper (B11: R_2 = L/2 - 2,
# R_1 = R_2 - 8), the antiperiodic twist (B1), the J4-cut ladder R_c in {6, 8, 10, 12} with the Euclidean area and the face count (B17),
# the shells d = 1, 2, 4 with the two-point bins, the partner of the 3-dimensional packet at omega/kappa = 4 (a diagnostic).  Version 2 of
# launch_obs_J4_v1.sh (which was never run): obs_box3d_v3.py and the route split of sec. 8.4:
#   step A  L = 48, the Gram route (B6) with the particle-hole Gram matrices (B10) over the region R < 12 (every region; 28832 modes at the
#           cell centre), complex64 evolution;
#   step B  L = 48, the direct-column route (B13) on the ball R < r_s (8704 modes): the two routes' entropy difference on the ball R < r_s is
#           the precision control of sec. 8.4;
#   step C  L = 64, the Gram route with B10 on the ball R < r_s (8704 modes);
#   step D  L = 64, the direct-column route on the region R < 12 (28832 modes; the Gram route's 119 GB per hold, 238 GB with B10, exceeds the
#           host budget);
#   step E  L = 64, the placement control: step D with the ball centred at a site (B12; 28492 modes).
# PREPARED, NOT RUN (builder analyst).  Usage on the box, detached, one step or several in order:
#   nohup bash launch_obs_J4_v2.sh <gpu> <host budget GB> <steps, e.g. "A B C D E"> > J4_launcher_v2.log 2>&1 < /dev/null &
# The GPU must hold less than 1 GB and no process at each step's launch; nice 10; 2 host threads.  Memory: see the builder's report (the
# Gram route's Z~ on the host by hold groups that fit the budget; the direct route's C_A per hold on the host, 13.3 GB at 28832 modes).
GPU=$1
BUDGET=${2:-210}
STEPS=${3:-"A B C D E"}
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
COMMON="--rs 8 --Rm 2 --mass 0 --dt 1 --holds 0 12 24 48 --shells 1 2 4 --rc-ladder 6 8 10 12 --slab 256 --host-budget-gb $BUDGET --omega-over-kappa 4 --R0-over-rs 2.2 --sigma-x 4 --outdir J4"
for st in $STEPS; do
  case $st in
    A) ARGS="--L 48 --chunk 512 --route gram --ph --dtype complex64 --tag v2A";;
    B) ARGS="--L 48 --chunk 512 --route direct --dtype complex64 --region-R 8 --tag v2B";;
    C) ARGS="--L 64 --chunk 256 --route gram --ph --dtype complex64 --region-R 8 --tag v2C";;
    D) ARGS="--L 64 --chunk 256 --route direct --dtype complex64 --tag v2D";;
    E) ARGS="--L 64 --chunk 256 --route direct --dtype complex64 --centre site --tag v2E";;
    *) echo "J4 unknown step $st" >> $STATUS; continue;;
  esac
  if ! gpu_free; then echo "J4 step $st not started: GPU $GPU not free at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS; exit 1; fi
  echo "J4 step $st start $(date -u +%Y-%m-%dT%H:%M:%SZ) (gpu $GPU, host budget $BUDGET GB)" >> $STATUS
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 $PY code/obs_box3d_v3.py j4 $COMMON $ARGS > J4/j4_step$st.log 2>&1 < /dev/null
  echo "J4 step $st exit $? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $STATUS
done
