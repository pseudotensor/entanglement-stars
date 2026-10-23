#!/bin/bash
# The r_s 32 set on engine_collapse_v17 on the H100 pod: the list's lanes (first column A, B, C, D) run side by side, each lane's runs one
# after another (the GPU is shared with the r_s 64 primary, whose step is bound by its CPU constraint solves); 2 host threads per run.
# Status lines in logs_h100_set17_status.txt. Usage (on the pod, detached): cd ${QI2_ROOT}/engine; nohup bash code/launch_h100_set17_v1.sh
# code/list_h100_set17_v1.txt > logs_h100_set17_launch.log 2>&1 < /dev/null &
# Builder analyst.
cd ${QI2_ROOT}/engine || exit 1
LIST=$1
PY=/usr/bin/python3.11
ST=logs_h100_set17_status.txt
mkdir -p C3set
lane() {
  local L=$1
  grep -v '^#' "$LIST" | awk -v l="$L" '$1==l' | while read -r ln tag args; do
    echo "lane $L $tag start $(date -u +%H:%M:%S) UTC" >> $ST
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY code/engine_collapse_v17.py $args --out C3set --gpu 0 --tag $tag > logs_h100_$tag.log 2>&1 < /dev/null
    echo "lane $L $tag exit $? $(date -u +%H:%M:%S) UTC" >> $ST
  done
  echo "lane $L DONE $(date -u +%H:%M:%S) UTC" >> $ST
}
for L in $(grep -v '^#' "$LIST" | awk '{print $1}' | sort -u); do
  lane $L &
done
wait
echo "set17 DONE $(date -u +%H:%M:%S) UTC" >> $ST
