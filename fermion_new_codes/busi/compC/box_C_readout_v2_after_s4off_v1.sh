#!/bin/bash
# box_C_readout_v2_after_s4off_v1.sh (busi lane, box side): waits for C's coupling-off control at sigma/a 4 (pod 1, tag ..._off) to end and its
# mirror (c_mirror_v6 s4off pod1off) to log "c_mirror_v6 ends", then checks the control's json has 641 records to T/r_s 40 and runs
# C_readout_v2.py (RUN_C_readout_v2.md) into results/compC/C_readout_s4_v2.*. CPU only, nice 10, 8 threads.
R=${QI2_ROOT}/compC; CTL=$R/C32_s4off_pod1off; ML=$R/C32mirror_s4off_pod1/mirror.log; PY=${QI2_ROOT}/python
until grep -q "c_mirror_v6 ends" $ML 2>/dev/null; do sleep 120; done
cd ${QI2_ROOT}/busi_numerics || exit 1
N=$($PY -I -c "import json,sys; d=json.load(open(sys.argv[1])); R=d['records']; print(len(R), R[-1]['T_over_rs'])" $CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off.json 2>&1)
echo "$(date -u +%FT%TZ) control records: $N" >> $R/C_readout_s4_v2_waiter.log
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
nice -n 10 $PY -I -B code/compC/C_readout_v2.py \
  --run-json $R/C32_s4_pod3/collapse_C32_s4_ell4_tau2_twofield_v3.json --s11-npz $R/C32_s4_pod3/collapse_C32_s4_ell4_tau2_twofield_v3_s11.npz \
  --control-json $CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off.json --control-fields-npz $CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_fields.npz \
  --control-s11-npz $CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_s11.npz --follow-json $R/follow_s4/collapse_follow_s4_lightfollow.json \
  --prod-json ${QI2_ROOT}/collapse_c31_rs32_A_R2_tau2_ell4_prep4_avg.json \
  --out-dir $R --tag C_readout_s4_v2 > $R/C_readout_s4_v2.log 2>&1
echo "$(date -u +%FT%TZ) C_readout_v2 exit $?" >> $R/C_readout_s4_v2_waiter.log
