#!/bin/bash
# box_run_v3c_all_v1.sh (busi lane): the red team's blocks applied (busi/reviews/redteam_C_readout_v3b.md; C_readout_v3c.py and
# C_readout_v3c_hbuilder.py), run as RUN_C_readout_v3b sections 2 to 6 with the v3c names: waits for the running v3b readout to end (its
# driver log's "readout exit"), then the self-tests, the adapter's real-data checks at T/r_s 40, the inspection and the readout tagged
# C_readout_s4_v3c; stops at the first non-zero exit. The solve modules from code/compC_engine_v36 (the tree the pods ran). Never stops or
# kills any process.
set -u
cd ${QI2_ROOT}/compC || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export BUSI_COMPC_DIR=${QI2_ROOT}/compC BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/compC_engine_v36
ENG=$BUSI_COMPC_DIR; O=${QI2_ROOT}/compC; R=$O
MEM0=$R/C32_s4_pod3/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTL0=$R/C32_s4off_pod1off/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEM=$R/C32_s4c80_pod1; CTL=$R/C32_s4offc50_pod1off50
MEMJ=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTLJ=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEMF=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3_fields.npz; CTLF=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_fields.npz
MEMS=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3_s11.npz; CTLS=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_s11.npz
MEMSL=${QI2_ROOT}/s4c80_pod1,${QI2_ROOT}/s4c80_pod1
CTLSL=${QI2_ROOT}/s4off_pod1off,${QI2_ROOT}/s4offc50_pod1off50,${QI2_ROOT}/s4offc50_pod1off50
MEM40=${QI2_ROOT}/orbitals_C32_s4_ell4_tau2_twofield_v3_T0040.00.npz
CTL40=${QI2_ROOT}/orbitals_C32_s4_ell4_tau2_twofield_v3_off_T0040.00.npz
until grep -q "readout exit" $O/C_readout_s4_v3b_driver.log; do sleep 60; done
echo "the v3b readout ended: $(grep 'readout exit' $O/C_readout_s4_v3b_driver.log)"
step() { echo "=== $(date -u '+%F %T UTC') $1 (load $(cut -d' ' -f1 /proc/loadavg))"; shift; "$@"; rc=$?; echo "=== exit $rc"; [ $rc = 0 ] || { echo "STOP"; exit $rc; }; }
md5sum C_readout_v3b.py C_readout_v3c.py C_readout_v3_hbuilder.py C_readout_v3c_hbuilder.py
step "selftest v2" nice -n 10 $PY -I -B C_readout_selftest_v2.py
step "selftest v3c" nice -n 10 $PY -I -B C_readout_v3c.py --selftest --engine-dir $ENG
step "selftest adapter v3c" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --selftest --engine-dir $ENG
step "adapter check member T40" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --check --slice $MEM40 --run-json $MEM0,$MEMJ --fields-npz $MEMF --engine-dir $ENG --out $O/C_readout_s4_v3c_adapter_check_member_T40.json
step "adapter check control T40" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --check --slice $CTL40 --run-json $CTL0,$CTLJ --fields-npz $CTLF --engine-dir $ENG --out $O/C_readout_s4_v3c_adapter_check_control_T40.json
echo "=== $(date -u '+%F %T UTC') inspection"
nice -n 10 $PY -I -B C_readout_v3c.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
  --slices $MEMSL --control-slices $CTLSL --inspect-slices > $O/C_readout_s4_v3c_inspect.json 2> $O/C_readout_s4_v3c_inspect.err
rc=$?; echo "=== inspection exit $rc"; [ $rc = 0 ] || exit $rc
echo "=== $(date -u '+%F %T UTC') readout v3c (load $(cut -d' ' -f1 /proc/loadavg))"
nice -n 10 $PY -I -B C_readout_v3c.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
  --slices $MEMSL --control-slices $CTLSL --run-cont-fields-npz $MEMF --control-cont-fields-npz $CTLF --run-cont-s11-npz $MEMS --control-cont-s11-npz $CTLS \
  --v2-json $O/C_readout_s4_v2.json --h-builder C_readout_v3c_hbuilder:h_of --engine-dir $ENG \
  --out-dir $O --tag C_readout_s4_v3c > $O/C_readout_s4_v3c.log 2>&1
echo "=== readout v3c exit $? $(date -u '+%F %T UTC')"
