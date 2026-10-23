#!/bin/bash
# box_run_v3b_checks_v1.sh (busi lane; RUN_C_readout_v3b sections 0, 2, 3 and 4 on the box): the three self-tests, the json check and the
# adapter's real-data check at the join for the member and the control; stops at the first non-zero exit. Never stops or kills any process.
set -u
cd ${QI2_ROOT}/compC || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export BUSI_COMPC_DIR=${QI2_ROOT}/compC BUSI_ENGINE_DIR=${QI2_ROOT}/engine BUSI_PAGE_DIR=${QI2_ROOT}/page
ENG=$BUSI_COMPC_DIR; O=${QI2_ROOT}/compC
R=$O
MEM0=$R/C32_s4_pod3/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTL0=$R/C32_s4off_pod1off/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEM=$R/C32_s4c80_pod1; CTL=$R/C32_s4offc50_pod1off50
MEMJ=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTLJ=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEMF=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3_fields.npz; CTLF=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_fields.npz
MEM40=${QI2_ROOT}/orbitals_C32_s4_ell4_tau2_twofield_v3_T0040.00.npz
CTL40=${QI2_ROOT}/orbitals_C32_s4_ell4_tau2_twofield_v3_off_T0040.00.npz
step() { echo "=== $(date -u '+%F %T UTC') $1 (load $(cut -d' ' -f1 /proc/loadavg))"; shift; "$@"; rc=$?; echo "=== exit $rc"; [ $rc = 0 ] || { echo "STOP"; exit $rc; }; }
ls $ENG/engine_twofield_v3.py $BUSI_ENGINE_DIR/engine_collapse_v36.py $BUSI_ENGINE_DIR/engine_collapse_v5.py $BUSI_PAGE_DIR/page_ops_v5.py || exit 1
md5sum C_readout_v3.py C_readout_v3b.py C_readout_v3_hbuilder.py
step "selftest v2" nice -n 10 $PY -I -B C_readout_selftest_v2.py
step "selftest v3b" nice -n 10 $PY -I -B C_readout_v3b.py --selftest --engine-dir $ENG
step "selftest adapter" nice -n 10 $PY -I -B C_readout_v3_hbuilder.py --selftest --engine-dir $ENG
step "json check" $PY -I -B -c "import json,sys
for p in sys.argv[1:]:
    d=json.load(open(p)); r=d['records']; print(p.split('/')[-2], len(r), r[0]['T_over_rs'], r[-1]['T_over_rs'], d['args']['light_coupling'], d['script'], d['args']['kappa_sea'], d['args']['m'], d['args']['nsites'], d['args']['half_offset'])" $MEM0 $CTL0 $MEMJ $CTLJ
step "adapter check member T40" nice -n 10 $PY -I -B C_readout_v3_hbuilder.py --check --slice $MEM40 --run-json $MEM0,$MEMJ --fields-npz $MEMF --engine-dir $ENG --out $O/C_readout_s4_v3b_adapter_check_member_T40.json
step "adapter check control T40" nice -n 10 $PY -I -B C_readout_v3_hbuilder.py --check --slice $CTL40 --run-json $CTL0,$CTLJ --fields-npz $CTLF --engine-dir $ENG --out $O/C_readout_s4_v3b_adapter_check_control_T40.json
echo "ALL CHECKS DONE $(date -u '+%F %T UTC')"
