#!/bin/bash
# pod5_after_A2gen_v1.sh (busi lane; reorientation 4's pod 5 order and #144): when A2gen_B2 has ended on pod 5 (its exit line in
# ${QI2_ROOT}/A2gen_status.txt), run O5's 4 H 232 cube runs (63 GB each, one at a time; o5_cube_runs_H232_v1.sh), then C's sigma/a 2
# domain flags (calculation (i) of C_s2_fold_reading_v1.2; about 19 GB; C_s2_domain_flags_v1.py from ${QI2_ROOT}/busi_compC_df). Each step logs
# to ${QI2_ROOT}/pod5_after_A2gen.log; nothing else on the pod is touched. v6n is placed separately (it needs a big disk).
L=${QI2_ROOT}/pod5_after_A2gen.log; log(){ echo "$(date -u +%FT%TZ) $*" >> $L; }
log "waiting for A2gen_B2's exit line"
until grep -q "A2gen_B2 exit" ${QI2_ROOT}/A2gen_status.txt 2>/dev/null; do sleep 60; done
log "A2gen_B2 ended: $(grep 'A2gen_B2 exit' ${QI2_ROOT}/A2gen_status.txt | tail -1)"
cd ${QI2_ROOT}/o5 && O5_CODE=${QI2_ROOT}/gap10 O5_REC=${QI2_ROOT}/record_slice.npz O5_LIFT=${QI2_ROOT}/lift O5_RUNS=${QI2_ROOT}/runs O5_LOGS=${QI2_ROOT}/logs O5_DEVICE=cuda:0 OMP_NUM_THREADS=8 nice -n 10 bash code/o5_cube_runs_H232_v1.sh > logs/cube_H232.out 2>&1
log "O5 H232 runs exit $?"
mkdir -p ${QI2_ROOT}/domain_flags_v1
cd ${QI2_ROOT}/code && BUSI_ENGINE_DIR=${QI2_ROOT}/engine BUSI_PAGE_DIR=${QI2_ROOT}/engine OMP_NUM_THREADS=16 nice -n 10 python C_s2_domain_flags_v1.py --engine-dir ${QI2_ROOT}/code --scan-dir ${QI2_ROOT}/soft_scan_v1 --out ${QI2_ROOT}/domain_flags_v1 --gpu 0 > ${QI2_ROOT}/C_s2_domain_flags_v1.log 2>&1
log "domain flags exit $?"
