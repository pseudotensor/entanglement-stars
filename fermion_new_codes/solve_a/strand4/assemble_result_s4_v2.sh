#!/bin/bash
# Strand 4, result revision 2: Part A's rerun of the primary reading with the code's midpoint hopping weight (partA_run2, dictionary
# 'p2mid'; tables by strand4_partA_report_v1.py on the box) beside revision 1's tables (partA_run1, now the sensitivity rows), and Part B
# (partB1.json, partB3.json; tables by strand4_partB_report_v1.py).  Fetches the tables, run records and JSON, and assembles
# solve/results/RESULT_strand4_controls.md (text assembly only; revision 1's text is kept in git history).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s4_controls
OUT=$W/solve/results/strand4
PY=${QI2_ROOT}/python
mkdir -p $OUT/partA_run2 $OUT/partB_run1
timeout 600 ssh -n -o BatchMode=yes compute-host "cd $B && export OMP_NUM_THREADS=1 && nice -n 10 $PY code/strand4_partA_report_v1.py --run partA_run2 --md partA_run2/RESULT_strand4_partA_tables.md > partA_run2/report.log 2>&1; tar czf partA_run2_json.tar.gz partA_run2/partA_*.json; nice -n 10 $PY code/strand4_partB_report_v1.py --b1 partB_run1/partB1.json --b3 partB_run1/partB3.json --md partB_run1/RESULT_strand4_partB_tables.md > partB_run1/report.log 2>&1"
for f in status.txt RESULT_strand4_partA_tables.md report.log partA_L6.log partA_L8.log; do
  timeout 120 scp -q compute-host:$B/partA_run2/$f $OUT/partA_run2/ || echo "missing partA_run2/$f"
done
timeout 120 scp -q compute-host:$B/partA_run2_json.tar.gz $OUT/partA_run2/ || echo "missing partA_run2 json archive"
for f in partB3.json partB3.log RESULT_strand4_partB_tables.md report.log; do
  timeout 120 scp -q compute-host:$B/partB_run1/$f $OUT/partB_run1/ || echo "missing partB_run1/$f"
done
R=$W/solve/results/RESULT_strand4_controls.md
{
echo "# Strand 4 result, revision 2 (controls): Part A with the code's midpoint hopping weight, and Part B, the dilation identity's pieces on the stored executed-star states [paper, control]"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand4_controls.md with the lead's rulings after Part A (the readings: m0 = 0.06741313"
echo "primary, the executed code's p = -2 dictionary; one rerun with the code's hopping weight Nbar_b/psibar_b^2). Every number here is a"
echo "control about the paper's model, not a number of H_*. Numbers, controls and commands only; no reading. Revision 1 (Part A, both"
echo "readings, hopping weight (s_x s_y)^-1) is in git history; its tables are repeated at the end as the sensitivity rows."
echo
echo "## Inputs"
echo "- Part A rerun: strand4_sea_lib_v2.py and strand4_partA_v2.py (dictionary 'p2mid': N psi^-2 on both quartics as 'p2', and the"
echo "  hopping weight Nbar_b/psibar_b^2 at the bond midpoint, psibar_b = (s_x + s_y)/2, in place of (s_x s_y)^-1); the paper's"
echo "  hf_closure_3d.py unmodified ($(cat $OUT/partA_run1/md5_paper_modules.txt)); tables by strand4_partA_report_v1.py."
echo "- Part B: strand4_partB1_v1.py (B.1, B.2) and strand4_partB3_v1.py (B.3 to B.5 pieces) on the paper lane's stored profiles and"
echo "  modules (copied with md5s: solve/results/strand4/partB_run1/md5_copied.txt); tables by strand4_partB_report_v1.py."
echo "- Box compute-host, conda env qi2, CPU; Part A one thread per run, sequential; Part B two threads."
echo
echo "## Commands and run records"
echo "Part A rerun: launch_strand4_partA_mid_v1.sh (partA_run2/), called from launch_small_after_prep31B4_v1.sh. Part B.1/B.2:"
echo "launch_strand4_partB1_v1.sh; B.3 to B.5: launch_strand4_partB3_v1.sh (partB_run1/, after the 2b jacobian-gap check)."
echo "Run record partA_run2 (status.txt):"
sed 's/^/    /' $OUT/partA_run2/status.txt
echo "Part B.3 log tail:"
tail -3 $OUT/partB_run1/partB3.log | cut -c1-400 | sed 's/^/    /'
echo
echo "## Readings and conventions taken by the builder"
echo "- Part A: as revision 1 (the model, the HS and Wick levels, the layered solver, the units), with the hopping ruler weight now the"
echo "  code's Nbar_b/psibar_b^2; the flat point is unchanged (psibar = s at psi = 1)."
echo "- Part B: the executed selected functional is E_sel = F_g^a[n, s; s_inf] + E_mat,ret[X; n, s]: F_g^a = -k (s - s_inf 1)^T K"
echo "  (chi - s_inf 1), chi = n s, K = 4 pi Gr^-1 with Gr_nk = 1/max(n, k) (the inverse of lattice_star_shell.green; the exterior and"
echo "  its boundary term are inside K), k = 1/(2 pi G_0); E_mat,ret = sum_n md sz + sum_b Qc Z + sum_n Wc ksy + E2 at fixed"
echo "  covariance (md = m_* N mu_L(psi), Qc = (Nbar/psibar^2) kappa_L(psibar), Wc = (N/(psi^2 n)) kappa_L(psi); E2 from the paper's"
echo "  CorrChain.e2_terms), as reconstructed from the build scripts; at lam = 1 it equals B.1's E_exc,ret (table B3)."
echo "- The family: n_lam(r_n) = n(lam r_n), s_lam(r_n) = lam^(1/2) s(lam r_n) by cubic splines with the exact exterior 1 + A/r,"
echo "  1 - B/r beyond N_s and the even continuation at the centre; orbitals transported per component (u(0) = 0, zero beyond N_s) and"
echo "  orthonormalized per channel by QR, so N_B and S_occ do not move (table B3). The pieces are reported as computed; the builder's"
echo "  question on the identity's form and normalization (INBOX 06:35 UTC) is open, so no combination is formed beyond"
echo "  Total = R_stat + S_inf, which holds by the family's construction."
echo
echo "## Part A, primary reading with the midpoint hopping weight (partA_run2)"
sed '1,2d' $OUT/partA_run2/RESULT_strand4_partA_tables.md
echo
echo "## Part B"
sed '1,2d' $OUT/partB_run1/RESULT_strand4_partB_tables.md
echo
echo "## Part A revision 1 tables (hopping weight (s_x s_y)^-1; both readings), the sensitivity rows"
sed '1,2d' $OUT/partA_run1/RESULT_strand4_partA_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
