#!/bin/bash
# Strand 2d, result revision 1: generate the tables on the box (strand2d_report_v4.py for C8run2 + prep31_C8 and B4run1 + prep31_B4),
# fetch them with the library checks, run records and JSON, and assemble solve/results/RESULT_strand2d_jet.md (text assembly only).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2d_jet
OUT=$W/solve/results/strand2d
PY=${QI2_ROOT}/python
mkdir -p $OUT
timeout 600 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 $PY code/strand2d_report_v5.py --run C8run2 --prep prep31_C8 --dressed prep31dr2_C8 --lattice C8 --md RESULT_strand2d_C8_tables.md > report_C8.log 2>&1 && OMP_NUM_THREADS=1 $PY code/strand2d_report_v5.py --run B4run1 --prep prep31_B4 --dressed prep31dr2_C8 --lattice B4 --md RESULT_strand2d_B4_tables.md > report_B4.log 2>&1 && tar czf strand2d_json.tar.gz C8run2/*.json B4run1/*.json prep31_C8/*.json prep31_B4/*.json prep31dr2_C8/*.json check2/check.json check3/check.json"
for f in RESULT_strand2d_C8_tables.md RESULT_strand2d_B4_tables.md strand2d_json.tar.gz report_C8.log report_B4.log; do
  timeout 300 scp -q compute-host:$B/$f $OUT/ || echo "missing $f"
done
for d in C8run2 B4run1 prep31_C8 prep31_B4 prep31dr2_C8; do
  mkdir -p $OUT/$d && timeout 120 scp -q compute-host:$B/$d/status.txt $OUT/$d/ || echo "missing $d/status.txt"
done
timeout 120 scp -q compute-host:$B/check2.log $OUT/ || true
timeout 120 scp -q compute-host:$B/check3.log $OUT/ || true
R=$W/solve/results/RESULT_strand2d_jet.md
{
echo "# Strand 2d result, revision 1 (test T7): the fixed-readout jet of the total energy at the flat point, on C8 and B4"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand2d_jet.md with the lead's rulings of sec. 7. Numbers, controls and commands only;"
echo "no physics reading. Tables emitted by solve/code/strand2/strand2d_report_v4.py."
echo
echo "## Inputs"
echo "- Code (solve/code/strand2/): strand2d_lib_v1.py (the B4 box; the (2,0) block of the light packet in torch with the seed's (2,2)"
echo "  projection built from its Kronecker structure; the common-sea Schur overlaps), strand2d_jet_v1.py (B4 run) and strand2d_jet_v2.py"
echo "  (C8 run; v2 also stores the lowest d_0 + 1 levels at every stencil point), strand2d_prep31_v1.py (sec. 7(2) stage 1: z, the"
echo "  (3,1)' amplitudes, their free evaluation, the fixed-vacuum cost, the Pauli and norm-defect checks), strand2d_prep31_dressed_v2.py"
echo "  (stage 2 on C8: the same terms with the completion through the gate's q-block route; at (0, 0) it reproduces stage 1), strand2d_report_v5.py; the"
echo "  library checks strand2d_check_v2.py (parts a, b on CPU: against strand2b_c8_lib_v3 on C8; part c: B4 timing on GPU 2)."
echo "- Library checks (check2.log, check3.log in solve/results/strand2d/):"
sed 's/^/    /' $OUT/check2.log
sed 's/^/    /' $OUT/check3.log
echo "- Box compute-host, conda env qi2; C8 on CPU; B4 on GPU 2 (torch complex128)."
echo "- Records: solve/results/strand2d/strand2d_json.tar.gz (all jet, prep31 and check JSON)."
echo
echo "## Commands"
echo "C8: launch_strand2d_C8_v2.sh (C8run2) and launch_strand2d_prep31_C8_v1.sh (prep31_C8); B4: launch_strand2d_B4_v1.sh (B4run1) and"
echo "launch_strand2d_prep31_B4_v1.sh (prep31_B4, queued behind B4run1 on GPU 2); then strand2d_report_v4.py on each pair."
for d in C8run2 B4run1 prep31_C8 prep31_B4 prep31dr2_C8; do echo "Run record $d:"; sed 's/^/    /' $OUT/$d/status.txt; done
echo
echo "## Readings and conventions taken by the builder"
echo "- B4: the periodic 4^3 box with the native cube's hopping convention (D[x, x + e_j] += sigma_j/(2i)); the V_B ruler distance is the"
echo "  minimum-image Euclidean distance on the torus; c_q = cos(q x_1) along the first lattice coordinate."
echo "- J^sus is the multiplet average Tr(P_d0 H)/d_0 with d_0 fixed at the flat point by the 1e-9 tolerance (sec. 7(1)); the lowest"
echo "  eigenvalue's jet is beside it, with the multiplet width per unit lam at the stencil points (T4 table)."
echo "- J^prep = J^prep_(2,0)' + J^prep_(3,1)' (sec. 7(2)): the first is the un-renormalized block-restricted functional of the flat"
echo "  multiplet projected by the Schur overlap; the second enters the psi psi entry only and is the free evaluation (stage 1: flat"
echo "  one-body quasiparticle energies), exact at (0, 0); on C8 at (0.2, 1) the dressed value (stage 2) is tabulated beside it; on B4 at"
echo "  (0.2, 1) the free value is the estimate, with the cube's dressed/free ratio beside it as its stated uncertainty (sec. 7(2))."
echo "  J^prep,vac is likewise the free evaluation 2 sum (eps_k + eps_m)|z_km|^2."
echo "- The B4 jet run is version 1: it records the multiplet width and the gap at every point but not the full list of the lowest"
echo "  d_0 + 1 levels (the C8 run, version 2, does)."
echo
sed '1,2d' $OUT/RESULT_strand2d_C8_tables.md
echo
sed '1,2d' $OUT/RESULT_strand2d_B4_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
