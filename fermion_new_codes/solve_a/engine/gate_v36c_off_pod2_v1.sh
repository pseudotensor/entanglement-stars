#!/bin/bash
# gate_v36c_off_pod2_v1.sh (solve builder 9, analyst; engine spec 51.1 (2)): the off gate of engine_collapse_v36c on pod 2's H100 (the stored
# records' device): v36c with --nk-set off, at launch_v32cont_pod2_v1.sh's exact arguments and tag, from the same upcast T/r_s 16 slice, to the
# engine's own end (the continuation's records 16.125 to 16.875, the end at 16.875 included; the turn names 16.125 and 16.25, the rest is a
# superset), compared with the v32 continuation's json and fields bitwise by compare_v36b_gate_v1 (wall-clock keys, script and args
# excluded). On PASS (0 json differences and 0 differing arrays) it writes GATE_OFF_PASS and starts launch_v36cc_pod2_v1.sh (the relaunch);
# on FAIL it stops there for the lead.
E=${QI2_ROOT}/engine; G=$E/v36c_gate_off; mkdir -p $G && cd $G || exit 1
T=c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36bc
[ -e collapse_$T.json ] && { echo "refusing: collapse_$T.json exists"; exit 1; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=$E/code_v36c PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SL=$(seq -s, 17 40)
RES="--resume-slice $E/v32cont/in/orbitals_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_T0016.00_c128.npz --resume-run $E/v32cont/in/run/c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b"
echo "gate off start $(date -u '+%Y-%m-%d %H:%M:%S UTC') pid $$" >> status.txt
nice -n 10 python $E/code_v36c/engine_collapse_v36c.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 40 --prep-ell 4 --tau 8 --gpu 0 --slices-c64 20,25,30,40 --slices-c128 $SL --ell 2 --mix newton --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --slice-dir states/$T --nk-jvp exact --field-kernel-cache --v36-fixes --pair-workers 2 $RES --out . --tag $T > $T.log 2>&1
echo "gate off engine exit $? $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
cd $E/code_v36c && python compare_v36b_gate_v1.py $G/$T $E/v32cont/$T > $G/gate_off_v1.txt 2>&1
cd $G
if grep -q "differences 0," gate_off_v1.txt && grep -q "differing 0," gate_off_v1.txt; then
  echo "PASS $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > GATE_OFF_PASS; echo "gate off PASS $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
  setsid nohup bash $E/code_v36c/launch_v36cc_pod2_v1.sh > /dev/null 2>&1 < /dev/null &
  echo "relaunch started $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
else
  echo "gate off FAIL $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> status.txt
fi
