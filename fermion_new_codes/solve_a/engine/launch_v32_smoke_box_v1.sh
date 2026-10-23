#!/bin/bash
# launch_v32_smoke_box_v1.sh (builder analyst; engine spec 32.1 (3)): box CPU smoke tests of engine_collapse_v32 at r_s 16 (512 sites,
# ell 2, prepared at ell 2, tau 2, --mix newton; about 2 s per evaluation there, from NK31/log_nk31_smoke_rs16.log):
#   S1 version 31 (SMOKE32/a) and version 32 with every option off (SMOKE32/b), the same arguments, --out O --tag s1 from their own
#      working directories so that the two jsons' args agree; compared by engine_collapse_v32_gate_v1 (the CPU twin of the pod 1 gate);
#   S2 the options on with the stop rule's field tolerance 1e-14 (no solve meets it), 2 Newton steps, 8 Krylov vectors and 1 restart:
#      the first record is the direct solve's (no previous record), the second runs the two-stage continuation; --end-above 1e-3;
#   S3 as S2 with --end-above 1e-12: the member ends at its first record (the end rule's path);
#   S4 the members' options (30 Newton steps, 40 Krylov vectors, 2 restarts, 6 halvings, two-stage, end above 1e-3) at the default stop rule.
# nice 10, 4 threads per run (16 at most at once), no GPU. Run on the box from engine/: nohup bash code/launch_v32_smoke_box_v1.sh > SMOKE32_launcher.log 2>&1 < /dev/null &
cd ${QI2_ROOT}/engine || exit 1
PY=${QI2_ROOT}/python
E=${QI2_ROOT}/engine
mkdir -p SMOKE32/a SMOKE32/b SMOKE32/c
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
COMMON="--m 0.2 --g 0.25 --rs 16 --R0-over-rs 4 --reading A --nsites 512 --kappa-sea 10 --reference R2 --site-fields averaged --ell 2 --prep-ell 2 --tau 2 --mix newton"
ON="--nk-maxiter 2 --nk-krylov 8 --nk-restarts 1 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage"
S=$E/SMOKE32/status.txt
run_in() { local dir=$1 tag=$2; shift 2; (cd $dir; nice -n 10 $PY "$@" > log_$tag.log 2>&1; echo "$tag exit $? $(date -u +%H:%M:%S) UTC" >> $S); }
echo "start $(date -u +%H:%M:%S) UTC" >> $S
run_in SMOKE32/a s1_v31 $E/code/engine_collapse_v31.py $COMMON --T-over-rs 0.25 --out O --tag s1 &
run_in SMOKE32/b s1_v32 $E/code/engine_collapse_v32.py $COMMON --T-over-rs 0.25 --out O --tag s1 &
run_in SMOKE32 s2_on $E/code/engine_collapse_v32.py $COMMON --T-over-rs 0.125 --stop-fields 1e-14 $ON --end-above 1e-3 --out O --tag s2_on &
run_in SMOKE32 s3_end $E/code/engine_collapse_v32.py $COMMON --T-over-rs 0.125 --stop-fields 1e-14 $ON --end-above 1e-12 --out O --tag s3_end &
wait
run_in SMOKE32 s4_members $E/code/engine_collapse_v32.py $COMMON --T-over-rs 0.25 --nk-maxiter 30 --nk-krylov 40 --nk-restarts 2 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3 --out O --tag s4_members &
run_in SMOKE32/c s1_v31b $E/code/engine_collapse_v31.py $COMMON --T-over-rs 0.25 --out O --tag s1 &
wait
$PY $E/code/engine_collapse_v32_gate_v1.py --ref SMOKE32/a/O/s1 --new SMOKE32/b/O/s1 --md SMOKE32/gate_s1_v32_vs_v31.md --json SMOKE32/gate_s1_v32_vs_v31.json > SMOKE32/log_gate_s1.log 2>&1
echo "gate_s1 exit $? $(date -u +%H:%M:%S) UTC" >> $S
$PY $E/code/engine_collapse_v32_gate_v1.py --ref SMOKE32/a/O/s1 --new SMOKE32/c/O/s1 --md SMOKE32/gate_s1_v31_vs_v31b.md --json SMOKE32/gate_s1_v31_vs_v31b.json > SMOKE32/log_gate_s1b.log 2>&1
echo "gate_s1b exit $? $(date -u +%H:%M:%S) UTC" >> $S
echo "DONE $(date -u +%H:%M:%S) UTC" >> $S
