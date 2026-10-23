#!/bin/bash
# GAP10 (required; reorientation 3 item 1): the wavelength-class test on the T7 record (busi/notes/GAP10_wavelength_test_v1.md
# section 10), started when the matrix script (pod_gap10_matrix_v4.sh) has written "all done": the 16 raw runs, the 8 matched
# 'in' component runs, then gap10_analysis_v3.py over runs/T7_wl. One process at a time on the GPU.
cd ${QI2_ROOT}/gap10; ST=${QI2_ROOT}/matrix_status.txt
# v2: started at once beside the matrix's CPU analysis (the GPU was idle; inbox #60)
echo "T7wl start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
B="GAP10_OPT2_BACKEND=compiled python ${QI2_ROOT}/gap10_run_v2_wl.py --rs 29 --ma 0.2 --ell 4 --Rb_frac 3.86 --profile file --profile_file ${QI2_ROOT}/record_slice.npz --sampling point --H 232 --sigma 6 --tmax_rs 10 --out ${QI2_ROOT}/T7_wl"
for PREP in matched fixed; do for DD in "100 2" "100 8" "111 2" "111 8"; do set -- $DD; for CLS in in sub; do
  bash -c "$B --direction $1 --depth $2 --prep $PREP --wl_classes $CLS --wl_components raw --tag _$PREP" >> logs_T7_wl.log 2>&1; echo "T7wl $1 d$2 $PREP $CLS raw exit $? $(date -u +%H:%M:%S)" >> $ST
done; done; done
for DD in "100 2" "100 8" "111 2" "111 8"; do set -- $DD; for COMP in above below; do
  bash -c "$B --direction $1 --depth $2 --prep matched --wl_classes in --wl_components $COMP --tag _matched" >> logs_T7_wl.log 2>&1; echo "T7wl $1 d$2 matched in $COMP exit $? $(date -u +%H:%M:%S)" >> $ST
done; done
mkdir -p T7_wl && python ${QI2_ROOT}/gap10_analysis_v3.py runs/T7_wl --csv T7_wl/results.csv --tcsv T7_wl/transmissivity.csv --kcsv T7_wl/wavelength.csv --md T7_wl/results.md --figs T7_wl/figs > T7_wl/analysis.log 2>&1; echo "T7wl analysis exit $? $(date -u +%H:%M:%S)" >> $ST
echo "T7wl all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
