#!/bin/bash
# Strand 2a red-team checks result (SPEC_strand2a_redteam_checks.md), revision 1: generate the tables on the box
# (strand2a_rtchecks_report_v1.py over s2a_gate/rtchecks1), fetch them with the JSON, logs and run record, and assemble
# solve/results/RESULT_strand2a_redteam_checks.md (text assembly only).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2a_gate
OUT=$W/solve/results/strand2a_redteam_checks
PY=${QI2_ROOT}/python
mkdir -p $OUT
timeout 600 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 nice -n 10 $PY code/strand2a_rtchecks_report_v1.py --dir rtchecks1 --md rtchecks1/RESULT_tables.md > rtchecks1/report.log 2>&1"
for f in status.txt RESULT_tables.md report.log rtchecks_AB.json rtchecks_C.json rtchecks_D.json rtchecks_E.json AB.log C.log D.log E.log gate/md5.txt; do
  timeout 300 scp -q compute-host:$B/rtchecks1/$f $OUT/ || echo "missing $f"
done
R=$W/solve/results/RESULT_strand2a_redteam_checks.md
{
echo "# Strand 2a red-team checks, revision 1: the repaired G7 diagnostic, the G8 Hessian's symmetry and origin, and the one-body checks of A.1, A.1c and A.4"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand2a_redteam_checks.md (lead, 2026-10-14 07:13 UTC), prompted by the reviewer B red team"
echo "(solve/reviewb/review_output/reviewb_azure_redteam_strand2a_20261013_231843/result.md). Numbers, controls and commands only; no reading."
echo "Tables emitted by solve/code/strand2/strand2a_rtchecks_report_v1.py."
echo
echo "## Inputs"
echo "- Code (solve/code/strand2/): strand2a_rtchecks_AB_v1.py (parts A and B, with the gate's library strand2a_lib.py, its coupling rows"
echo "  and backgrounds; the gate stored no V_* matrices, so V_*, S_* and the dressed blocks are rebuilt and the rebuilt lowest eigenvalues"
echo "  are checked against the gate's G7 records, column 'E_min - gate's E_min' of A.4), strand2a_rtchecks_CDE_v1.py (parts C, D, E: the"
echo "  addendum's one-body ring objects), the report strand2a_rtchecks_report_v1.py."
echo "- The gate records read: gate_L2.json and gate_R4.json (run2) and gate_C8.json (run1), md5s:"
sed 's/^/    /' $OUT/md5.txt
echo "- Box compute-host, conda env qi2, CPU, two threads; records in solve/results/strand2a_redteam_checks/."
echo
echo "## Commands and run record"
echo "launch_queue_slotD2_v1.sh (after strand 4 Part B version 2): strand2a_rtchecks_AB_v1.py --out rtchecks1 --gate rtchecks1/gate, then"
echo "strand2a_rtchecks_CDE_v1.py --part C, D, E."
sed 's/^/    /' $OUT/status.txt
echo
echo "## Readings and conventions taken by the builder"
echo "- A.2 and A.3 use the gate's nonuniform R4 background as a whole (psi = 1 + 0.2 cos(2 pi x/4) and N = 1 + 0.3 (-1)^x); A.3 keeps that"
echo "  psi with N = 1. dS_*/dg_M is S_* at g_M = 1 (S_* is linear in the couplings); the first-order operator of H_exc = e^S H_0[N] e^S"
echo "  is {H_0[N], S_1}; the free ground multiplet is the set of H_0[N] levels within 1e-9 of the lowest."
echo "- A.4's corrected remainder subtracts the free pair minimum E_free,min (not 2 eps_min) and the lowest eigenvalue of V_* on the"
echo "  free ground multiplet (the first-order operator at unit lapse on the flat lattices); the gate's tabulated columns are repeated"
echo "  unchanged and labelled 'gate'."
echo "- B.2 recomputes the gate's Hessian stencil at three steps; clusters group eigenvalues within 1e-3 (relative above 1)."
echo "- C.1 differentiates the matrix B = E^(1/2) psi E^(1/2) by central differences (the second derivative's step is printed) and"
echo "  uses the exact flat plane-wave basis; H2 is the 2 x 2 second-order effective matrix of the {0, pi} doublet."
echo "- D's packets are the addendum's construction (Gaussian in p about each valley, p in [-pi/2, pi/2) about it) at L = 2048; the"
echo "  unit-cell value is the mean of the sites 0 and 1."
echo "- E keeps the addendum's elements and rotation W; at L = 4096 only j = 1, 2 are run (j = 1 to 4 at L = 1024); the windows are"
echo "  max(|p|, |p'|) <= w m with w = 1/8, 1/4, 1/2 and p measured from the valley."
echo
sed '1,2d' $OUT/RESULT_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
