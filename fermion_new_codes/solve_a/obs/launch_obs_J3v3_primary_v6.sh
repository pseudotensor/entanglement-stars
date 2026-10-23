#!/bin/bash
# Version 6 of launch_obs_J3v3_primary_v1.sh: version 5 on the primary restarted on engine_collapse_v11 (the v9 primary stopped at T/rs 5.88 on a
# solver stall, dN 1.77e-11, dW 1.31e-10 after 200 iterations; version 11 keeps such a pair with the certificate's label). Version 5: version 4 on
# the primary restarted on engine_collapse_v9 (the pair block exported per step). Version 4: version 3 with a guard: J3' runs only if the primary's record reaches T/rs 30 (a rejected run
# ends early and its history is not J3''s input; the guard logs and exits). Version 3: version 2 on the primary restarted on engine_collapse_v8 (version 6 was rejected at T/rs 0.25 on a
# solver stall at the integrator floor; its J3' run started on the two-slice history and was stopped). Version 2: (the primary restarted on engine_collapse_v6 at tau = 8, engine 16.2; observables 12.3's plan:
# both j = 1/2 and 3/2): J3' on the primary's history (B14 = obs_raymap_solved_v3) on GPU 1 after the collapse primary: waits until the C3
# launcher (launch_engine_C3_v4.sh) has exited and the primary's fields npz exists and GPU 1 is free (the resident litellm services are not
# jobs), then runs, one process at a time: (1) the main set (kappa_j = +1; m_p/kappa 0 and 1/4; omega/kappa 1/2, 1, 2; T_surf 12 to 28 r_s);
# (2) the kappa_j = -1 check (m_p = 0, omega/kappa = 1, T_surf 24). Then on the CPU: PL-ext's labelled ell = 4 comparison on the last slice
# (obs_plateau_v5, dense and spline) and B3/B6 on it (obs_plateau_mix_v4). nice 10, 4 host threads.
# Usage (on the box, in ${QI2_ROOT}/obs, detached): nohup bash code/launch_obs_J3v3_primary_v6.sh > J3v3/launcher_v6.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/obs
mkdir -p J3v3
PY=${QI2_ROOT}/python
REC=../engine/C3/collapse_primary_rs64_A_R2_tau8_v11.json
FLD=../engine/C3/collapse_primary_rs64_A_R2_tau8_v11_fields.npz
gpu_free() {
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==1{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==1{print $2}')
  local nproc=0
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u {print $2}'); do
    ps -o args= -p $p | grep -q litellm || nproc=$((nproc + 1))
  done
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
c3_alive() { ps -eo args | awk '$1=="bash" && $2=="code/launch_engine_gpu_v4.sh" && $3=="1"' | grep -q .; }
echo "J3v3 primary (v11, guarded) waiter start $(date -u +%H:%M:%S) UTC (GPU 1 after the primary)" >> status.txt
while c3_alive || [ ! -f "$FLD" ] || ! gpu_free; do sleep 60; done
sleep 30
Tend=$($PY -c "import json; r=[q for q in json.load(open('$REC'))['records'] if 'T_over_rs' in q]; print(r[-1]['T_over_rs'] if r else 0)")
if ! $PY -c "import sys; sys.exit(0 if float('$Tend') >= 30.0 else 1)"; then
  echo "J3v3 (v11) not started: the primary's record ends at T/rs $Tend (below 30) $(date -u +%H:%M:%S) UTC" >> status.txt
  exit 0
fi
run() {
  local tag=$1; shift
  if ! gpu_free; then echo "J3v3 $tag skipped: GPU 1 not free $(date -u +%H:%M:%S) UTC" >> status.txt; return 1; fi
  echo "J3v3 $tag start $(date -u +%H:%M:%S) UTC (gpu 1)" >> status.txt
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nice -n 10 $PY code/obs_raymap_solved_v3.py --record $REC --fields $FLD "$@" --gpu 0 \
    --out-json J3v3/$tag.json --md J3v3/$tag.md > J3v3/$tag.log 2>&1 < /dev/null
  echo "J3v3 $tag exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
}
run primary_main
run primary_kjm1 --sign -1 --mp-over-kappa 0 --omega-over-kappa 1 --tsurf-list 24
run primary_j1.5 --j 1.5
echo "J3v3 PL-ext ell 4 comparison start $(date -u +%H:%M:%S) UTC (CPU)" >> status.txt
OMP_NUM_THREADS=4 nice -n 10 $PY code/obs_plateau_v5.py --fields $FLD --records $REC --slice -1 --site-order 5 --R-lead 32 64 \
  --out-json J3v3/plext_ell4_last.json --md J3v3/plext_ell4_last.md > J3v3/plext_ell4_last.log 2>&1 < /dev/null
OMP_NUM_THREADS=4 nice -n 10 $PY code/obs_plateau_mix_v4.py --profile solved --fields $FLD --records $REC --slice -1 --site-order 5 --r 0.25 0.5 1 \
  --out-json J3v3/mix4_ell4_last.json --md J3v3/mix4_ell4_last.md > J3v3/mix4_ell4_last.log 2>&1 < /dev/null
echo "J3v3 primary list DONE $(date -u +%H:%M:%S) UTC" >> status.txt
