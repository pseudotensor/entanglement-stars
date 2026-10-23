#!/bin/bash
# The two star families queued on GPU 0 behind the ell = 8 control (list_gpu0_v6.txt lines 2 and 3: (0.1, 1/16) at nsites 4096 and (0.05, 1/4)
# at nsites 3072) run instead on the CPU with engine_star_v6 (sparse channels, shift-invert levels; equal to v5 to the star's tolerance on
# three members: results/engine/Stest/compare_v6_v5_three.md), both at once, 4 threads each, nice 10, the same arguments. Builder analyst.
# Usage (on the box, detached, from engine): nohup bash code/launch_star_v6_cpu_v1.sh > star_v6_cpu_launcher_v1.log 2>&1 < /dev/null &
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
run() {
  local tag=$1; shift
  echo "CPU star_v6 $tag start $(date -u +%H:%M:%S) UTC" >> status.txt
  ( OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 $PY code/engine_star_v6.py "$@" --tag $tag > logs_cpu_star_v6_$tag.log 2>&1 < /dev/null
    echo "CPU star_v6 $tag exit $? $(date -u +%H:%M:%S) UTC" >> status.txt ) &
}
run m0.1_g0.0625_ell8 --m 0.1 --g 0.0625 --ell 8 --nsites 4096 --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json --ell-control 4,8,16 --closed-count --out Sfam6
run m0.05_g0.25_ell8 --m 0.05 --g 0.25 --ell 8 --nsites 3072 --eos EOS/eos_v4_tables.npz --ovpg OVPG/ovpg_families.json --ell-control 4,8,16 --closed-count --out Sfam7
wait
