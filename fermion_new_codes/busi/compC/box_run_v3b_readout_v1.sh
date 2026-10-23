#!/bin/bash
# box_run_v3b_readout_v1.sh (busi lane; RUN_C_readout_v3b sections 5 and 6 on the box): waits for box_run_v3b_checks_v2.sh to end
# ("ALL CHECKS DONE" in its log; a "STOP" there stops this too), then the slices' schema and the readout, with the solve modules from
# code/compC_engine_v36 (the tree the pods ran). No --allow-arg-mismatch; no subset of clocks. Never stops or kills any process.
set -u
cd ${QI2_ROOT}/compC || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export BUSI_COMPC_DIR=${QI2_ROOT}/compC BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/compC_engine_v36
ENG=$BUSI_COMPC_DIR; O=${QI2_ROOT}/compC; R=$O; CL=$O/C_readout_s4_v3b_checks.log
MEM0=$R/C32_s4_pod3/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTL0=$R/C32_s4off_pod1off/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEM=$R/C32_s4c80_pod1; CTL=$R/C32_s4offc50_pod1off50
MEMJ=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTLJ=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEMF=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3_fields.npz; CTLF=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_fields.npz
MEMS=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3_s11.npz; CTLS=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_s11.npz
MEMSL=${QI2_ROOT}/s4c80_pod1,${QI2_ROOT}/s4c80_pod1
CTLSL=${QI2_ROOT}/s4off_pod1off,${QI2_ROOT}/s4offc50_pod1off50,${QI2_ROOT}/s4offc50_pod1off50
until grep -q "ALL CHECKS DONE" $CL; do grep -q "^STOP" $CL && { echo "the checks stopped; nothing run"; exit 2; }; sleep 30; done
echo "=== $(date -u '+%F %T UTC') inspection (load $(cut -d' ' -f1 /proc/loadavg))"
nice -n 10 $PY -I -B C_readout_v3b.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
  --slices $MEMSL --control-slices $CTLSL --inspect-slices > $O/C_readout_s4_v3b_inspect.json 2> $O/C_readout_s4_v3b_inspect.err
rc=$?; echo "=== inspection exit $rc"; tail -3 $O/C_readout_s4_v3b_inspect.err; [ $rc = 0 ] || exit $rc
echo "=== $(date -u '+%F %T UTC') readout (load $(cut -d' ' -f1 /proc/loadavg))"
nice -n 10 $PY -I -B C_readout_v3b.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
  --slices $MEMSL --control-slices $CTLSL --run-cont-fields-npz $MEMF --control-cont-fields-npz $CTLF --run-cont-s11-npz $MEMS --control-cont-s11-npz $CTLS \
  --v2-json $O/C_readout_s4_v2.json --h-builder C_readout_v3_hbuilder:h_of --engine-dir $ENG \
  --out-dir $O --tag C_readout_s4_v3b > $O/C_readout_s4_v3b.log 2>&1
echo "=== readout exit $? $(date -u '+%F %T UTC')"
