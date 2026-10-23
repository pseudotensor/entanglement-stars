#!/bin/bash
# Strand 0: fetch the full_v2 outputs from the box and assemble solve/results/RESULT_strand0_source_allocation.md.
# Text assembly only; the tables come verbatim from strand0_report_v2.py.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s0_source_allocation
OUT=$W/solve/results/strand0
mkdir -p $OUT
for f in status.txt pytest.log partA.log ringL.log cube.log report.log partA.json ringL.json cube.json RESULT_strand0_tables.md; do
  timeout 120 scp -q compute-host:$B/full_v3/$f $OUT/ || echo "missing $f"
done
for f in law_B0_m0.1_U0.1.npz law_B2_e0.2_m0.1_U0.1.npz; do
  timeout 120 scp -q compute-host:$B/full_v3/mats/$f $OUT/ || echo "missing $f"
done
R=$W/solve/results/RESULT_strand0_source_allocation.md
{
echo "# Strand 0 result (test T2): the law's clock derivative against the midpoint densities, ring4 and the open cube"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand0_source_allocation.md (lead, 2026-10-14, with its sec. 10)."
echo
echo "## Revision 2 (after the independent reviewer B red team of 2026-10-14, solve/reviewb/review_output/reviewb_azure_redteam_strand0_*/result.md)"
echo "Rerun as full_v3 with data script v3, ring-L v3, cube v2 and report v4. The physics objects are unchanged (the controls"
echo "below reproduce revision 1's values); the changes are in what is computed and reported:"
echo "- T3: the anomalous one-body norm is now given three ways: revision 1's sum of the two block Frobenius norms (labelled), the"
echo "  Frobenius norm of the combined off-diagonal part, and the single d^dag f^dag block."
echo "- T4: the contact difference is normalized both by the law's contact norm in the block (revision 1) and by rho^P_x's norm"
echo "  in the block (T9's convention), each labelled."
echo "- T5: the native c-number is reported at every background (B0, B2, B3, B4 and the 18 B1 sbar), not only at s = 1; the native"
echo "  vacuum expectation is given at the row's U (density and vacuum at U) beside the U = 0 one (labelled); the midpoint density's"
echo "  reference c-number per site is reported."
echo "- T2: the native one-particle minimizer now carries <W v|X1_x|W v> and its weight on the law's positive subspace."
echo "- T7: the native re-solves carry the explicit sbar uniformity flag at 1e-10; the full bridge S is saved."
echo "- npz: Delta_PM, Delta_MN, Delta_PN saved by name; the midpoint uses h_* at the background lapse (all backgrounds have N = 1)."
echo "- T8: offsets j, j' in {-3..3} modulo L at every L, labelled by the offsets (revision 1 wrapped negative offsets in the valley"
echo "  at 0, so its (1,-1), (2,-2), (3,-3) entries were missing there), and the complex elements X_pp, M_pp, M_anom, Delta_pp"
echo "  written to ringL.json."
echo "- Part B: every cube site reported (revision 1 printed 2 of the 8 computed); the massless fallback implemented (not triggered)."
echo "The C5e diagnostic below was computed in revision 1 (full_v2/c5e_diag.json); the objects it uses are unchanged."
echo "Numbers, controls and commands only; no physics reading. Every table below is emitted by"
echo "solve/code/strand0/strand0_report_v4.py from the JSON files in solve/results/strand0/."
echo
echo "## Inputs"
echo "- Code (solve/code/strand0/): strand0_lib.py; strand0_partA_data_v3.py (v1 crashed in a smoke test on an indexing"
echo "  bug before producing anything; v2 produced revision 1); strand0_ringL_onebody_v3.py; strand0_cube_onebody_v2.py; strand0_report_v4.py;"
echo "  tests/strand0_controls_test_v3.py; frozen native modules in frozen/ (FROZEN_SOURCES.txt: paths, commits, md5)."
echo "- Box compute-host, conda env qi2: $(jq -r '.meta | "host \(.host), python \(.python), numpy \(.numpy), scipy \(.scipy), OMP threads \(.threads)"' $OUT/partA.json)."
echo "- Stored native energies for control C9: ${QI2_ROOT}/native_jobs.tar.gz (U = 0.1) and"
echo "  native_kscan.json.gz (U = 0, k = 1, wide grid, P = 0), copies of the committed files of review_output/native_action_20261012/."
echo "- Matrices for every case (npz): box $B/full_v3/mats/; two cases copied to solve/results/strand0/."
echo
echo "## Command"
echo "solve/code/strand0/launch_strand0_full_v3.sh (run in $B): pytest tests/strand0_controls_test_v3.py;"
echo "strand0_partA_data_v3.py --out full_v3 --stored-jobs ... --stored-kscan ...; strand0_ringL_onebody_v3.py --out full_v3;"
echo "strand0_cube_onebody_v2.py --out full_v3; strand0_report_v4.py --partA --ringL --cube --md full_v3/RESULT_strand0_tables.md."
echo "The C5e diagnostic at the end: launch_strand0_c5e_diag.sh (revision 1)."
echo
echo "Run record (status.txt):"
sed 's/^/    /' $OUT/status.txt
echo "pytest: $(grep -E 'passed|failed' $OUT/pytest.log | tail -1)"
echo
echo "## Readings and conventions taken by the builder (accepted by the lead in the spec's sec. 10 unless noted)"
echo "- Native contact density: psi_x^-6 Gamma(W) co.V[x] Gamma(W)^dag (co.V[x] already carries U)."
echo "- B3 reported at x = 0 and x = 1; B2 and B4 at x = 0 and 2; uniform backgrounds at x = 0; all x in the JSON."
echo "- T9 'block': the (n_p, n_h) block of the lowest eigenvector of H_exc in that sector; norms compressed to it."
echo "- B1 backgrounds: J_W6 re-solves at U in {0, 0.1} and END at U = 0.1, k = 1, wick; the law evaluated at that U."
echo "- T8: Delta_pp := <v^+_p|M1_0|v^+_p'> - <v^+_p|X1_0|v^+_p'> (rho^M - rho^P), the order of LEAD.tex eq. (Delta-exact);"
echo "  the spec text said 'their difference'. A first ring-L script (v1) took X1 - M1 and failed C12 on that element only."
echo "  T3's Delta^PM = rho^P - rho^M keeps the spec's order."
echo "- One-particle witness energies are expectations of H_free^exc (= H_exc on the one-excitation sector); in the extended"
echo "  search both <H_free^exc> and <H_exc> are given, and <rho^P_x> is the dressed density (the bare one beside it)."
echo "- The 'lowest state' in a block is the block-averaged lowest eigenspace of H_exc restricted to that block (tolerance 1e-9)."
echo
sed "1,2d" $OUT/RESULT_strand0_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
{
echo
echo "## Builder diagnostic of control C5e (not a spec table): ||H_exc - H_free^exc - V_*|| at three couplings"
echo
echo "Script solve/code/strand0/strand0_c5e_diag.py via launch_strand0_c5e_diag.sh (box, full_v2/c5e_diag.json); C5e's"
echo "U = 0.01 / U = 0.1 ratio leaves the spec's 10 percent band only on B3_e0.2, so the next decade was added there and on B0."
echo
echo "| bg | m | q | max U psi^-6 at U = 0.1 | R(0.001) | R(0.01) | R(0.1) | R(0.01)/R(0.1) | R(0.001)/R(0.01) |"
echo "|---|---|---|---|---|---|---|---|---|"
jq -r '.[] | "| \(.bg) | \(.m) | +\(.N - 4) | \(."max_U_psi^-6_at_U=0.1" * 10000 | round / 10000) | \(."R(0.001)") | \(."R(0.01)") | \(."R(0.1)") | \(."ratio_0.01_over_0.1" * 100000 | round / 100000) | \(."ratio_0.001_over_0.01" * 100000 | round / 100000) |"' $OUT/c5e_diag.json
} >> $R
echo "appended C5e diagnostic"
