#!/bin/bash
# box_run_C32_pathfinder_readout_v1.sh (solve, composition 1, Reorientation 6): C_readout_v3b (as issued after its red team: busi's
# C_readout_v3c.py + C_readout_v3c_hbuilder.py, run in place read-only, -B so nothing is written beside them) over the r_s/a 32 pathfinder
# s4c80 (C32_s4_ell4_tau2_twofield_v3, light coupling on, 0 to 80) against its control s4offc80 (..._v3_off, coupling off, 0 to 80; 50 to 80
# on engine_twofield_v8 --reference-solve implicit, R2). RUN_C_readout_v3b sections 2 to 6, the control's continuation inputs swapped for
# the 80 ones. Outputs to $O. Stops at the first non-zero exit except the planned P2 probe (step 5a).
set -u
cd ${QI2_ROOT}/compC || exit 1
PY=${QI2_ROOT}/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export BUSI_COMPC_DIR=${QI2_ROOT}/compC BUSI_ENGINE_DIR=${QI2_ROOT}/compC_engine_v36 BUSI_PAGE_DIR=${QI2_ROOT}/compC_engine_v36
ENG=$BUSI_COMPC_DIR; R=${QI2_ROOT}/compC
O=${QI2_ROOT}/C32_pathfinder_v1; mkdir -p $O
MEM0=$R/C32_s4_pod3/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTL0=$R/C32_s4off_pod1off/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEM=$R/C32_s4c80_pod1; CTL=${QI2_ROOT}/T0080.00
MEMJ=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3.json; CTLJ=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off.json
MEMF=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3_fields.npz; CTLF=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_fields.npz
MEMS=$MEM/collapse_C32_s4_ell4_tau2_twofield_v3_s11.npz; CTLS=$CTL/collapse_C32_s4_ell4_tau2_twofield_v3_off_s11.npz
NEW=${QI2_ROOT}/slices
MEMSL=${QI2_ROOT}/s4c80_pod1,${QI2_ROOT}/s4c80_pod1
CTLSL=${QI2_ROOT}/s4off_pod1off,${QI2_ROOT}/s4offc50_pod1off50,${QI2_ROOT}/s4offc50_pod1off50,$NEW
MEM40=${QI2_ROOT}/orbitals_C32_s4_ell4_tau2_twofield_v3_T0040.00.npz
CTL40=${QI2_ROOT}/orbitals_C32_s4_ell4_tau2_twofield_v3_off_T0040.00.npz
MEM80=${QI2_ROOT}/orbitals_C32_s4_ell4_tau2_twofield_v3_T0080.00.npz
CTL80=$NEW/orbitals_C32_s4_ell4_tau2_twofield_v3_off_T0080.00.npz
CTL60=$NEW/orbitals_C32_s4_ell4_tau2_twofield_v3_off_T0060.00.npz
step() { echo "=== $(date -u '+%F %T UTC') $1 (load $(cut -d' ' -f1 /proc/loadavg))"; shift; "$@"; rc=$?; echo "=== exit $rc"; [ $rc = 0 ] || { echo "STOP"; exit $rc; }; }
md5sum C_readout_v3b.py C_readout_v3c.py C_readout_v3_hbuilder.py C_readout_v3c_hbuilder.py C_readout_v2.py C_readout_v1.py C_readout_selftest_v2.py engine_twofield_v3.py \
  $BUSI_ENGINE_DIR/engine_collapse_v36.py $BUSI_ENGINE_DIR/engine_collapse_v5.py $BUSI_PAGE_DIR/page_ops_v5.py
md5sum $MEM0 $CTL0 $MEMJ $CTLJ $MEMF $CTLF $MEMS $CTLS $R/C_readout_s4_v2.json
step "selftest v2" nice -n 10 $PY -I -B C_readout_selftest_v2.py
step "selftest v3c" nice -n 10 $PY -I -B C_readout_v3c.py --selftest --engine-dir $ENG
step "selftest adapter v3c" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --selftest --engine-dir $ENG
step "adapter check member T40" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --check --slice $MEM40 --run-json $MEM0,$MEMJ --fields-npz $MEMF --engine-dir $ENG --out $O/adapter_check_member_T40.json
step "adapter check control T40" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --check --slice $CTL40 --run-json $CTL0,$CTLJ --fields-npz $CTLF --engine-dir $ENG --out $O/adapter_check_control_T40.json
CL=${QI2_ROOT}/C32_s4offc80_R2
until grep -q "box md5 done" $CL/copy_log.txt; do sleep 60; done
echo "=== copies: $(grep -c '^copied' $CL/copy_log.txt) copied, $(grep -c FAIL $CL/copy_log.txt) FAIL"
step "md5 R2 against box" ${QI2_ROOT}/python -I -c "
import sys
a=dict((l.split()[1],l.split()[0]) for l in open('$CL/md5_R2.txt') if l.strip()); b=dict((l.split()[1],l.split()[0]) for l in open('$CL/md5_box.txt') if l.strip())
bad=[f for f in b if a.get(f)!=b[f]]; print('box files',len(b),'R2 md5s',len(a),'mismatch',bad)
import glob,os
for t in ('49','50'):
    f='orbitals_C32_s4_ell4_tau2_twofield_v3_off_T00%s.00.npz'%t
    ref=[open(p).read().split()[0] for p in glob.glob('${QI2_ROOT}/s4offc50_pod1off50'+f+'.md5')]
    print('join', f, 'R2', a.get(f), 'box c50 .md5', ref)
sys.exit(1 if bad or len(b)!=60 else 0)"
step "adapter check member T80" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --check --slice $MEM80 --run-json $MEM0,$MEMJ --fields-npz $MEMF --engine-dir $ENG --out $O/adapter_check_member_T80.json
step "adapter check control T80 (v8 record)" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --check --slice $CTL80 --run-json $CTL0,$CTLJ --fields-npz $CTLF --engine-dir $ENG --out $O/adapter_check_control_T80.json
step "adapter check control T60 (v8 record)" nice -n 10 $PY -I -B C_readout_v3c_hbuilder.py --check --slice $CTL60 --run-json $CTL0,$CTLJ --fields-npz $CTLF --engine-dir $ENG --out $O/adapter_check_control_T60.json
echo "=== $(date -u '+%F %T UTC') inspection (no --allow-arg-mismatch)"
nice -n 10 $PY -I -B C_readout_v3c.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
  --slices $MEMSL --control-slices $CTLSL --inspect-slices > $O/inspect.json 2> $O/inspect.err
rc=$?; echo "=== inspection exit $rc"; cat $O/inspect.err | tail -3
if [ $rc != 0 ]; then
  nice -n 10 $PY -I -B C_readout_v3c.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
    --slices $MEMSL --control-slices $CTLSL --inspect-slices --allow-arg-mismatch > $O/inspect_flag.json 2> $O/inspect_flag.err
  echo "=== inspection with --allow-arg-mismatch exit $?"
fi
echo "=== $(date -u '+%F %T UTC') 5a readout as the run file states it (no --allow-arg-mismatch)"
nice -n 10 $PY -I -B C_readout_v3c.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
  --slices $MEMSL --control-slices $CTLSL --run-cont-fields-npz $MEMF --control-cont-fields-npz $CTLF --run-cont-s11-npz $MEMS --control-cont-s11-npz $CTLS \
  --v2-json $R/C_readout_s4_v2.json --h-builder C_readout_v3c_hbuilder:h_of --engine-dir $ENG \
  --out-dir $O --tag C32_pathfinder_noflag > $O/C32_pathfinder_noflag.log 2>&1
echo "=== 5a exit $? $(date -u '+%F %T UTC')"; tail -5 $O/C32_pathfinder_noflag.log
if [ -s $O/C32_pathfinder_noflag.md ]; then echo "5a wrote the ledger without the flag"; exit 0; fi
echo "=== $(date -u '+%F %T UTC') 5b readout with --allow-arg-mismatch (the P2 differences printed)"
nice -n 10 $PY -I -B C_readout_v3c.py --run-json $MEM0 --run-cont-json $MEMJ --control-json $CTL0 --control-cont-json $CTLJ \
  --slices $MEMSL --control-slices $CTLSL --run-cont-fields-npz $MEMF --control-cont-fields-npz $CTLF --run-cont-s11-npz $MEMS --control-cont-s11-npz $CTLS \
  --v2-json $R/C_readout_s4_v2.json --h-builder C_readout_v3c_hbuilder:h_of --engine-dir $ENG --allow-arg-mismatch \
  --out-dir $O --tag C32_pathfinder_v1 > $O/C32_pathfinder_v1.log 2>&1
echo "=== readout exit $? $(date -u '+%F %T UTC')"
