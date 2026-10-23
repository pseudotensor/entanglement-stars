#!/bin/bash
# Strand 4, result revision 1 (Part A): fetch partA_run1's tables, status and JSON, and assemble solve/results/RESULT_strand4_controls.md
# (text assembly only).  Part B follows as a revision.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s4_controls
OUT=$W/solve/results/strand4
mkdir -p $OUT/partA_run1
timeout 120 ssh -n -o BatchMode=yes compute-host "cd $B && tar czf partA_run1_json.tar.gz partA_run1/partA_*.json && md5sum full_numerics/hf_closure_3d.py > partA_run1/md5_paper_modules.txt"
for f in status.txt RESULT_strand4_partA_tables.md md5_paper_modules.txt cases.txt; do
  timeout 120 scp -q compute-host:$B/partA_run1/$f $OUT/partA_run1/ || echo "missing $f"
done
timeout 120 scp -q compute-host:$B/partA_run1_json.tar.gz $OUT/partA_run1/ || echo "missing json archive"
R=$W/solve/results/RESULT_strand4_controls.md
{
echo "# Strand 4 result, revision 1 (controls; Part A): the raw vacuum jet of the paper's two-quartic half-filled sea [paper, control]"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand4_controls.md (lead, 2026-10-14). Every number here is a control about the"
echo "paper's model, not a number of H_*. Numbers, controls and commands only; no reading. Part B (the dilation identity on the stored"
echo "executed-star states) follows as a revision."
echo
echo "## Inputs"
echo "- Code (solve/code/strand4/): strand4_sea_lib_v1.py (the layered HS/HF solver with the clock and ruler weights; constants and"
echo "  the uniform solver imported from the paper's full_numerics/hf_closure_3d.py, unmodified: $(cat $OUT/partA_run1/md5_paper_modules.txt)),"
echo "  strand4_partA_v1.py (one (L, m0, dictionary) per call: flat-point check, stencil, jets, controls, target, xi),"
echo "  strand4_partA_report_v1.py (tables)."
echo "- Box compute-host, conda env qi2, CPU, one thread per run, 4 runs at a time."
echo
echo "## Commands"
echo "solve/code/strand4/launch_strand4_partA_v1.sh in ${QI2_ROOT}/s4_controls (output partA_run1/), then"
echo "strand4_partA_report_v1.py --run partA_run1. Records: solve/results/strand4/partA_run1/ (tables, status, the JSON archive)."
echo
echo "Run record (status.txt):"
sed 's/^/    /' $OUT/partA_run1/status.txt
echo
echo "## Readings and conventions taken by the builder"
echo "- The model is hf_closure_3d.py's: 4 components, TI_i = -i/2 AL_i, G0 = sz (x) 1, b_{x,i} = <psi_{x+i}^dag TI_i psi_x>,"
echo "  E_HS = sum_bonds (-2 t0 Re b - U_B |b|^2) + sum_x (m0 sigma - U_M sigma^2), E_Wick = E_HS + sum_bonds U_B X + sum_x U_M Y;"
echo "  t0 = 1, U_B = 1, U_M = 0.2, half filling, zero temperature, periodic box, twist 0. The clock weights every local term (bonds"
echo "  at the midpoint); the ruler enters the hopping as (s_x s_y)^-1 and the quartics by the dictionary; the mass term carries N only."
echo "- Two open readings (INBOX, for the lead) are both run: the bare mass m0 = 0.06741313 (07_spherical_star lines 110-112, the"
echo "  executed star's flat-matched value for m_*/t_* = 0.1) and m0 = 0.3 (the spec's literal value); the quartic ruler weights"
echo "  'spec' (s^-6 on site, (s_x s_y)^-3 on the bond) and 'p2' (the executed star's p = -2: s^-2 on site, (s_x s_y)^-1 on the bond)."
echo "- The perturbations vary along the first axis only, so the state keeps the translations along the other two: the solver works"
echo "  per (k_2, k_3) on the L layers (exact for these perturbations; the symmetric HF of the spec). The flat point is checked"
echo "  against hf_closure_3d.uniform_solve (table A0)."
echo "- The HS level's 'ground energy' is reported two ways: E_HS (the saddle's reduced energy) and E_Wick (the full <H> in the saddle's"
echo "  Slater state); the preparation cost is <Phi_0|H[N, s]|Phi_0> = E_Wick of the flat state at the perturbed weights (and E_HS of it)."
echo "- The Slater quadratic control agrees to about 1e-7 relative, not the spec's 1e-9: the central difference at 1e-4 sits at its"
echo "  rounding floor (energies of order 10^2 to 10^3); a fourth-order stencil would be the follow-up."
echo "- Target units: the sea's energies are in t0 units; the target uses G_0 = 0.2745414 as given in the spec (t_* units in the paper)."
echo "- The auxiliary-field Hessian and Zhat/What are not exposed by hf_closure_3d.py and are not computed (the paper's values are"
echo "  quoted at the end of the tables)."
echo
sed '1,2d' $OUT/partA_run1/RESULT_strand4_partA_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
