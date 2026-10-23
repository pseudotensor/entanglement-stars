#!/bin/bash
# The next families on GPU 0 (SPEC_final4b_engine.md secs. 12.8 and 14.3: "the (0.1, 1/4) family to its end, its ell-control, the (0.2, 1/4)
# approach case, (0.2, 1/16), (0.1, 1/16)"): waits until the S-fam5 launcher (launch_engine_Sfam5_v1.sh) has exited and GPU 0 holds under 1 GB
# and no process, then runs the families in order on engine_star_v5 with the (0.1, 1/4) run's options (zero temperature, fixed N_B,
# --closed-count, the ell-control 4, 8, 16); nsites from the continuum families' sizes (BUILDER: 1024, 1536, 4096).
# Usage (on the box, detached): nohup bash code/launch_engine_Sfam6_v1.sh > Sfam6_launcher.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
mkdir -p Sfam6
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
gpu_free() {
  local used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==0{print $2}')
  local uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')
  local nproc=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u=$uuid '$1==u' | wc -l)
  [ "$used" -lt 1024 ] && [ "$nproc" -eq 0 ]
}
sfam5_alive() { ps -eo args | awk '$1=="bash" && $2=="launch_engine_Sfam5_v1.sh"' | grep -q .; }
echo "S-fam6 waiter start $(date -u +%H:%M:%S) UTC (GPU 0 after S-fam5)" >> status.txt
while sfam5_alive || ! gpu_free; do sleep 60; done
sleep 30
for spec in "0.2 0.25 1024" "0.2 0.0625 1536" "0.1 0.0625 4096"; do
  set -- $spec; m=$1; g=$2; ns=$3; tag=m${m}_g${g}_ell8
  if ! gpu_free; then echo "S-fam6 ($m, $g) skipped: GPU 0 not free at $(date -u +%H:%M:%S) UTC" >> status.txt; continue; fi
  echo "S-fam6 ($m, $g) ell 8 nsites $ns start $(date -u +%H:%M:%S) UTC (gpu 0)" >> status.txt
  nice -n 10 $PY code/engine_star_v5.py --m $m --g $g --ell 8 --nsites $ns --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json \
    --ell-control 4,8,16 --closed-count --gpu 0 --out Sfam6 --tag $tag > Sfam6/$tag.out 2>&1 < /dev/null
  echo "S-fam6 ($m, $g) exit $? $(date -u +%H:%M:%S) UTC" >> status.txt
done
echo "S-fam6 DONE $(date -u +%H:%M:%S) UTC" >> status.txt
