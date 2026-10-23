#!/bin/bash
# Strand 7 S7a revision 2 (SPEC_strand7_collapse.md sec. 9.2): strand7_shell_v2.py, four single-thread jobs at a time (threads from the stopped
# J5mem2 h = 0.5 run), the lead's order: the held-lock diagnostic and the continuation at E_0 = 16 first, then E_0 = 32 (N_p = 2e4) and 8, the
# warm variant B' (v_perp = 0.1) at E_0 = 8, and C3's step-halving at E_0 = 16 for both; all case A from R_0 = 4 r_s to T = 60 r_s unless the
# stop rules end them; the held-lock runs keep their fields (S7b's input).  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
mkdir -p S7a_run2
date -u '+start %Y-%m-%d %H:%M:%S UTC' > S7a_run2/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=${QI2_ROOT}/python
run1() { tag=$1; shift; nice -n 10 $PY code/strand7_shell_v2.py "$@" --out S7a_run2 --tag $tag > S7a_run2/$tag.log 2>&1; echo "$tag exit $? $(date -u +%H:%M:%S)" >> S7a_run2/status.txt; }
export -f run1
cat > S7a_run2/jobs.txt <<'JOBS'
heldlock_E16_R4_A --run heldlock --E0 16 --R0-over-rs 4 --case A --T-over-rs 60 --save-fields 1
self_E16_R4_A --run self --E0 16 --R0-over-rs 4 --case A --T-over-rs 60 --save-fields 1
heldlock_E32_R4_A --run heldlock --E0 32 --R0-over-rs 4 --case A --T-over-rs 60 --Np 20000 --save-fields 1
self_E32_R4_A --run self --E0 32 --R0-over-rs 4 --case A --T-over-rs 60 --Np 20000
self_E8_R4_A --run self --E0 8 --R0-over-rs 4 --case A --T-over-rs 60
heldlock_E8_R4_A --run heldlock --E0 8 --R0-over-rs 4 --case A --T-over-rs 60 --save-fields 1
self_E8_R4_Bp --run self --E0 8 --R0-over-rs 4 --case B --vperp 0.1 --T-over-rs 60
self_E16_R4_A_dt --run self --E0 16 --R0-over-rs 4 --case A --T-over-rs 60 --dt-over-rs 0.005
heldlock_E16_R4_A_dt --run heldlock --E0 16 --R0-over-rs 4 --case A --T-over-rs 60 --dt-over-rs 0.005
JOBS
xargs -P 4 -L 1 bash -c 'run1 "$@"' _ < S7a_run2/jobs.txt
echo DONE >> S7a_run2/status.txt
