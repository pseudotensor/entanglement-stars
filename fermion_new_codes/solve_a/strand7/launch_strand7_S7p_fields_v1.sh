#!/bin/bash
# Item 7's background runs (SPEC_strand7_page.md secs. 2 and 10): strand7_shell_v3.py with --stop-rule page (to T = 60 r_s, N(r_sh) < 1e-12, a
# Newton failure or a non-positive lapse at a marker), fields kept at every output: the held lock at E_0 = 16 (h = 1, the primary P; h = 0.5,
# the check H), at E_0 = 32 (the check R), and the continuation (self) at E_0 = 16 (the comparison C); two single-thread jobs at a time on the
# free threads.  Builder analyst.
D=${QI2_ROOT}/s7_collapse
cd $D || exit 1
mkdir -p S7p_fields
date -u '+start %Y-%m-%d %H:%M:%S UTC' > S7p_fields/status.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PY=${QI2_ROOT}/python
run1() { tag=$1; shift; nice -n 10 $PY code/strand7_shell_v3.py "$@" --stop-rule page --T-over-rs 60 --save-fields 1 --out S7p_fields --tag $tag > S7p_fields/$tag.log 2>&1; echo "$tag exit $? $(date -u +%H:%M:%S)" >> S7p_fields/status.txt; }
export -f run1
cat > S7p_fields/jobs.txt <<'JOBS'
heldlock_E16_h1 --run heldlock --E0 16 --R0-over-rs 4 --case A
heldlock_E16_h05 --run heldlock --E0 16 --R0-over-rs 4 --case A --h 0.5
heldlock_E32_h1 --run heldlock --E0 32 --R0-over-rs 4 --case A --Np 20000
self_E16_h1 --run self --E0 16 --R0-over-rs 4 --case A
JOBS
xargs -P 2 -L 1 bash -c 'run1 "$@"' _ < S7p_fields/jobs.txt
echo DONE >> S7p_fields/status.txt
