#!/bin/bash
# Item 7, the repair pass, version 2 (strand7_page_repair_v2.py: the unresolved pairs classified by X's singular values) on GPU 0 (one process
# at a time, checked free), one CPU thread: P at j = 1/2, 3/2, 5/2, B40, S0 on L, then F0 (S7p_run1), C at j = 1/2, and K1, K2, H, R when their
# v4 runs end.  Version 1's _r2 outputs were moved to S7p_run3/repair_v1_superseded/ (and S7p_run1's) before this chain.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7p_run3
PY=${QI2_ROOT}/python
gpu_free() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
run() {  # tag-list dir md
  until gpu_free 0; do sleep 30; done
  date -u "+repair v2 $1 start %H:%M:%S on cuda:0" >> $O/repair_status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strand7_page_repair_v2.py --dir $2 --tags $1 --device cuda:0 --md $3 > $3.log 2>&1
  echo "repair v2 $1 exit $? $(date -u +%H:%M:%S)" >> $O/repair_status.txt
}
run P_j2.5,P_j0.5,P_j1.5,B40,S0_L $O $O/repair2_P_B40_S0.md
run F0 S7p_run1 $O/repair2_F0.md
run C_j0.5 $O $O/repair2_C05.md
for t in K1 K2 H R; do
  until grep -q "^$t exit" $O/status.txt; do sleep 30; done
  run $t $O $O/repair2_$t.md
done
echo "DONE v2" >> $O/repair_status.txt
