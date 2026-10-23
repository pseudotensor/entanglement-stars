#!/bin/bash
# GAP10 (required; reorientation 3 item 1): the wavelength-class test on the T7 record (busi/notes/GAP10_wavelength_test_v1.md
# section 10). v3: the same 24 runs and analysis as pod_gap10_wl_v2.sh, moved off the pod's network volume /workspace, whose
# FUSE mount (MooseFS) stopped answering at about 09:50 UTC on 2026-10-17 and left v2's 16th run in uninterruptible sleep
# holding the whole H100 (inbox #61). Everything here lives on the container disk under ${QI2_ROOT}/busi_local and never touches
# /workspace. It waits until the GPU's memory is free (the stuck process exits only when the volume answers), then runs
# one process at a time; a run whose output tag is already logged as exit 0 is skipped, so the script can be restarted.
R=${QI2_ROOT}/busi_local; G=$R/gap10; mkdir -p $G/runs/T7_wl $G/T7_wl; cd $G; ST=$G/wl_status.txt
echo "T7wl v3 waiting for the GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 4000 ]; do sleep 60; done
echo "T7wl v3 start $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
B="GAP10_OPT2_BACKEND=compiled python $R/code/gap10/gap10_run_v2_wl.py --rs 29 --ma 0.2 --ell 4 --Rb_frac 3.86 --profile file --profile_file $R/records/record_slice.npz --sampling point --H 232 --sigma 6 --tmax_rs 10 --out $G/runs/T7_wl"
run() {  # $1 label for the status line, rest the run's own arguments
  local L="$1"; shift
  if grep -q "^T7wl $L exit 0 " $ST; then return; fi
  if [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -lt 5 ]; then echo "T7wl $L SKIPPED disk under 5 GB $(date -u +%H:%M:%S)" >> $ST; return; fi
  bash -c "$B $*" >> logs_T7_wl.log 2>&1; echo "T7wl $L exit $? $(date -u +%H:%M:%S)" >> $ST
}
for PREP in matched fixed; do for DD in "100 2" "100 8" "111 2" "111 8"; do set -- $DD; for CLS in in sub; do
  run "$1 d$2 $PREP $CLS raw" --direction $1 --depth $2 --prep $PREP --wl_classes $CLS --wl_components raw --tag _$PREP
done; done; done
for DD in "100 2" "100 8" "111 2" "111 8"; do set -- $DD; for COMP in above below; do
  run "$1 d$2 matched in $COMP" --direction $1 --depth $2 --prep matched --wl_classes in --wl_components $COMP --tag _matched
done; done
python $R/code/gap10/gap10_analysis_v3.py runs/T7_wl --csv T7_wl/results.csv --tcsv T7_wl/transmissivity.csv --kcsv T7_wl/wavelength.csv --md T7_wl/results.md --figs T7_wl/figs > T7_wl/analysis.log 2>&1; echo "T7wl analysis exit $? $(date -u +%H:%M:%S)" >> $ST
echo "T7wl all done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> $ST
