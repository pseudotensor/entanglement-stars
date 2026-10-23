#!/bin/bash
# Strand 2a: fetch run2's outputs from the box (L2 and R4 by gate v2; C8 by gate v1, copied from run1 by launch_strand2a_v2.sh)
# and assemble solve/results/RESULT_strand2a_gate.md (text assembly only).  Version 2 of the assembly: version 1 pointed at run1,
# where the L2 and R4 gates crashed on the vacuum block of the Fock route.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/run2
B1=${QI2_ROOT}/run1
OUT=$W/solve/results/strand2a
mkdir -p $OUT/run1
for f in status.txt pytest.log gate_L2.json gate_R4.json gate_C8.json gate_L2.log gate_R4.log gate_C8.log report.log RESULT_strand2a_tables.md; do
  timeout 120 scp -q compute-host:$B/$f $OUT/ || echo "missing $f"
done
for f in status.txt pytest.log gate_L2.log gate_R4.log; do
  timeout 120 scp -q compute-host:$B1/$f $OUT/run1/ || echo "missing run1/$f"
done
R=$W/solve/results/RESULT_strand2a_gate.md
{
echo "# Strand 2a result (test T1): the lattice realization of H_* on L2, R4 and C8 with the gate of integ17 sec. 7"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand2a_gate.md (lead, 2026-10-14). Numbers, controls and commands only; no"
echo "physics reading. Tables emitted by solve/code/strand2/strand2a_report.py from the JSON files in solve/results/strand2a/."
echo
echo "## Inputs"
echo "- Code (solve/code/strand2/): strand2a_lib.py (lattices, packets, polar transport, seed with regulator and V_B, the"
echo "  algebraic (2,2) projection two_two, q-blocks, dressing), strand2a_gate_v2.py (L2, R4), strand2a_gate.py (C8),"
echo "  strand2a_report.py, tests/strand2a_gate_test_v2.py; strand0_lib.py (strand 0's library, copied unchanged) for the Fock"
echo "  route on L2 and R4; frozen native modules in frozen/."
echo "- Gate versions: version 2 differs from version 1 only in the Fock-route isometry, whose vacuum block (0,0) is Omega"
echo "  itself (version 1 raised on the empty creation tuple). C8 does not use the Fock route, so its run under version 1 is"
echo "  the same computation; its JSON and log are copied from run1 into run2."
echo "- Box compute-host, conda env qi2; one single-thread process per lattice."
echo
echo "## Commands"
echo "solve/code/strand2/launch_strand2a_v1.sh in ${QI2_ROOT}/s2a_gate (run1/: pytest v1 and all three lattices;"
echo "used for C8 only), then solve/code/strand2/launch_strand2a_v2.sh (run2/: pytest v2, L2 and R4 by gate v2, C8 copied"
echo "from run1 when it finished, then the report)."
echo
echo "Run record, run1 (status.txt; its pytest failures and the L2/R4 crashes are the vacuum-block defect fixed in version 2):"
sed 's/^/    /' $OUT/run1/status.txt
echo "run1 pytest: $(grep -E 'passed|failed' $OUT/run1/pytest.log | tail -1)"
echo "run1 L2 and R4 last log lines: $(tail -1 $OUT/run1/gate_L2.log) / $(tail -1 $OUT/run1/gate_R4.log)"
echo
echo "Run record, run2 (status.txt):"
sed 's/^/    /' $OUT/status.txt
echo "run2 pytest: $(grep -E 'passed|failed' $OUT/pytest.log | tail -1)"
echo
echo "## Readings and conventions taken by the builder (questions recorded in solve/lead/INBOX.md)"
echo "- Mode order: j = 2 w + chiral with w = site * ncomp + component (chiral fastest; chiral 0 = u, the first block of h_*)."
echo "- L2: D_W = D2 (x) sigma_1 in the site-major layout (the spec's 'sigma_1 (x) D2' read as the operator, not the layout)."
echo "- R4: the ruler-length kernel uses the ring distance min(|x - y|, 4 - |x - y|); the nonuniform clock is N_x = 1 + 0.3 (-1)^x"
echo "  (the 1d analogue of (-1)^{x_2}); the random background draws psi then N from default_rng(7) uniform on [0.7, 1.3]."
echo "- G7's g_M ratio at g_B = 0 needs a (g_M, g_B) = (0.2, 0) row; it is added for G7 only (marked extra_G7_row in the JSON)."
echo "- G4's K_NN by central differences at h = 0.1 (W_0 and E_vac are affine in N, so no truncation error; rounding ~1e-13); the"
echo "  many-body K_NN on R4 and L2 at the row (0.2, 1, Lambda = 1) and H = 1 only (the heavy packets enter additively through W_0)."
echo "- G1 on C8: S_* is built per (n_p, n_h) block, so [S_*, n_exc] = [S_*, N_c] = 0 holds by construction there; on L2 and R4 the"
echo "  q-block route is checked block by block against the Fock-space route (G1c)."
echo "- G8: the mixed matrix d2W/dpsi dN is not symmetric; its symmetric part's eigenvalues and its largest singular value are given."
echo
sed '1,2d' $OUT/RESULT_strand2a_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
