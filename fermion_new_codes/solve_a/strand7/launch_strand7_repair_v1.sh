#!/bin/bash
# Item 7, the repair pass (SPEC_strand7_page.md sec. 14.5; strand7_page_repair_v1.py) on GPU 0 (an RTX A6000, free at launch; one process at a
# time), one CPU thread: P at j = 1/2, 3/2, 5/2 and B40 now; F0 (S7p_run1); C at j = 1/2 when its run ends; K1 when its run ends (both on
# version 4 in version 5's queue).  S0 on L was repaired first as the script's test (repair_test_S0.md).  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
O=S7p_run3
PY=${QI2_ROOT}/python
gpu_free() { [ -z "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
run() {  # tag-list dir md
  until gpu_free 0; do sleep 30; done
  date -u "+repair $1 start %H:%M:%S on cuda:0" >> $O/repair_status.txt
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY code/strand7_page_repair_v1.py --dir $2 --tags $1 --device cuda:0 --md $3 > $3.log 2>&1
  echo "repair $1 exit $? $(date -u +%H:%M:%S)" >> $O/repair_status.txt
}
run P_j0.5,P_j1.5,P_j2.5,B40 $O $O/repair_P_B40.md
run F0 S7p_run1 $O/repair_F0.md
until grep -q "^C_j0.5 exit" $O/status.txt; do sleep 30; done
run C_j0.5 $O $O/repair_C05.md
until grep -q "^K1 exit" $O/status.txt; do sleep 30; done
run K1 $O $O/repair_K1.md
echo DONE >> $O/repair_status.txt
