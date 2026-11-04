#!/bin/bash
# B29 core stage (route 2), inbox #349: the fixture against route 1, then the four slices at dr 0.1, then their dr 0.05 reruns
# (the discretization budget; the explicit response reused, the same evaluation rows). One GPU (cuda:0, free per nvidia-smi), nice 10.
set -u
PY=${QI2_ROOT}/python
C=${QI2_ROOT}/code
O=${QI2_ROOT}/route2_core
L=$O/logs
R1=${QI2_ROOT}/core_hf_v1
cd $C
run() { echo "$(date -u +%FT%TZ) start $*" >> $L/chain.log; nice -n 10 $PY -B route2_core_b29_v1.py "$@" --device cuda:0 --threads 8; echo "$(date -u +%FT%TZ) end rc=$? $*" >> $L/chain.log; }
run --slice prod 16 --kappas 1 --tag k1x --save-kernels --out $O/fixture --route1-arrays $O/fixture_route1/prod/core_hf_prod_T0016.00_k1_arrays.npz > $L/fixture_prod16_k1x.log 2>&1
for s in "prod 16 card_bench1b/prod/core_hf_prod_T0016.00" "prod 15 card_bench1b/prod/core_hf_prod_T0015.00" "s8 14 card_R2/s8/core_hf_s8_T0014.00" "s8 16 card_R2/s8/core_hf_s8_T0016.00"; do
  set -- $s
  run --slice $1 $2 --save-kernels --out $O --route1-arrays $R1/$3_arrays.npz > $L/${1}_T$2.log 2>&1
done
for s in "prod 16" "prod 15" "s8 14" "s8 16"; do
  set -- $s
  t=$(printf "%07.2f" $2)
  run --slice $1 $2 --dr 0.05 --tag dr05 --skip-nocur --no-D-eig --out $O --pa-from $O/${1}_T${t}_arrays.npz --r-eval-from $O/${1}_T${t}.json > $L/${1}_T$2_dr05.log 2>&1
done
echo "$(date -u +%FT%TZ) chain done" >> $L/chain.log
