#!/bin/bash
# Strand 5 J2 and J3 result (assembly v1): tables from J2run3 (strand5_kernel_v3.py) and J3run3 (strand5_eos_v4.py) by
# strand5_kernel_report_v1.py and strand5_eos_report_v1.py on the box; fetches the records; writes solve/results/RESULT_strand5_kernel_eos.md.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s5_star
O2=$W/solve/results/strand5/J2run3
O3=$W/solve/results/strand5/J3run3
PY=${QI2_ROOT}/python
mkdir -p $O2 $O3
timeout 300 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 $PY code/strand5_kernel_report_v1.py --json J2run3/kernel_v3.json --kernel ${QI2_ROOT}/kernel_L256.json --md J2run3/RESULT_tables_v1.md > J2run3/report_v1.log 2>&1 && OMP_NUM_THREADS=1 $PY code/strand5_eos_report_v1.py --json J3run3/eos_v4.json --md J3run3/RESULT_tables_v1.md > J3run3/report_v1.log 2>&1"
for f in status.txt run.log kernel_v3.json kernel_v3_K.npz RESULT_tables_v1.md; do timeout 120 scp -q compute-host:$B/J2run3/$f $O2/ || echo "missing J2 $f"; done
for f in status.txt run.log eos_v4.json eos_v4_tables.npz RESULT_tables_v1.md; do timeout 120 scp -q compute-host:$B/J3run3/$f $O3/ || echo "missing J3 $f"; done
R=$W/solve/results/RESULT_strand5_kernel_eos.md
{
echo "# Strand 5 result J2 and J3, revision 1: the kernel ingredients (T3) and the lattice EOS with the matter rows (T4)"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand5_kernel_star.md sec. 3 J2, J3 (lead, 2026-10-14); LEAD.tex sec. 8. Numbers, controls"
echo "and commands only; no reading. Tables emitted by solve/code/strand5/strand5_kernel_report_v1.py (T3) and strand5_eos_report_v1.py (T4)."
echo
echo "## Inputs and commands"
echo "- J2: solve/code/strand5/strand5_kernel_v3.py on the 2c part A table s2c_mismatch/kernel1/kernel_L256.json (the same file as"
echo "  solve/results/strand2c_addendum/kernel1/kernel_L256.json); box compute-host, CPU, seconds; records solve/results/strand5/J2run3/."
echo "  Earlier versions kept: v1 (used the table's zone-edge entry as tabulated, which left a non-decaying alternating line kernel; box run"
echo "  J2run1), v2 (the entry halved; box run J2run2, whose items v3 reproduces and extends)."
echo "- J3: solve/code/strand5/strand5_eos_v4.py, launcher launch_strand5_J3_v3.sh (box, CPU, 2 threads, about 7.5 minutes); records"
echo "  solve/results/strand5/J3run3/ (eos_v4.json, eos_v4_tables.npz). Earlier versions kept: v1 (its smoke run showed the strain"
echo "  difference at fixed n limited by rounding at small filling, step 1e-4), v2 (step 1e-3; its secant root fell through to a brentq call"
echo "  that scipy rejects; box run J3run1 stopped), v3 (step 1e-3 pushed the Fermi ball past the cones at theta_F = 1.55; box run J3run2"
echo "  stopped); v4 takes the step min(1e-3, (1 - kappa)/4)."
echo
echo "## Readings and conventions taken by the builder"
echo "J2."
echo "1. The 2c table's zone-edge entry (n = 128, q = pi) is twice the symbol: strand2c_kernel_v1.py defines k_a = -2 W^(2)/(q_lat^2 L^3)"
echo "   with the cosine-mode norm L^3/2, while cos(pi x_1) = (-1)^x_1 has norm L^3 (T3c: the entry is twice its neighbour). J2 halves it."
echo "   The 2c addendum's table (i), row n = 128, carries the doubled value."
echo "2. The spec's fit range q <= aH/4 holds at most one table point for aH <= 0.1 (the table's spacing is 2 pi/256). The line kernel's"
echo "   moments give k_a(0) and c_4 at every aH (columns beside the spec's); they rely on K decaying within the 256-ring, which T3c shows"
echo "   for every equal-rectangle aH. The physical row (light packet massless) does not decay on the ring; its moments are not reliable."
echo "3. For aH <= 0.1 the 2c addendum's k_a(0) is the smallest-q value; T3a compares it with the second moment and gives both ratios to"
echo "   k_lat."
echo "4. P14's prediction is evaluated with k := the fitted k_a(0); with the moment's k instead the prediction changes by less than 1e-4"
echo "   of itself at every aH."
echo "5. The spec's Green-function integral is cut at the zone edge q = pi, the end of the [100] table. Its deviation from the Newtonian"
echo "   tail falls as 1/r and alternates at integer r, the form of the cut's boundary term"
echo "   (2/pi) k_a(0) [(1/k_a(pi) - 1/k_a(0))/pi] cos(pi r)/r; the column with k_a(q) -> k_a(0) isolates the cut for the local row (T3d)."
echo "   T3e gives the Green function on the radial lattice instead (a smeared source solved with the line kernel)."
echo "6. The radial operator is built on a uniform radial grid of spacing a, not on J1's map: J1's spacing near R_max is about r_s, far"
echo "   larger than the kernel's range for J4's stars, where a convolution would not be resolved. The J4 grid is the lead's (for example a"
echo "   uniform inner grid to beyond the matter plus the range, then the exact exterior A/r). The operator's Gauss charge is the second"
echo "   moment's k (T3f)."
echo "7. J4's members (m a, aH) = (0.03, 0.3) and (0.01, 0.1) need kernels at those light masses: the 2c table has m = 0.1 only, and"
echo "   aH = 0.3 is not tabulated. At m a = 0.01 the light Compton length is 100 sites. Per q value strand2c_kernel_v1.py sums over L"
echo "   times about L^2/8 transverse classes, and the table needs L/2 values of q."
echo "J3."
echo "8. theta_F = k_F b is the axis Fermi momentum and kappa = sin(theta_F) the radius of the Fermi ball in s_i = sin(theta_i). The table"
echo "   stops at theta_F = 1.55: at kappa = 1 the cones' Fermi surfaces touch, a band critical value outside the regular branch."
echo "9. e_1 uses the continuum-equivalent k_F = (3 pi^2 n/8)^(1/3) as the cone radius (the resolved-cone forms), in lattice units of"
echo "   b = a at the flat point with Lambda = 1, l = 1, g_M = 0.2, g_B = 1. In these forms e_1 depends on n, m, Lambda, l and the couplings"
echo "   only, so b d_b e_1 = 0 at fixed n; the exact finite-zone strain coefficient (r3's (R16)) is not evaluated."
echo "10. The spec's control 'the sign of e_M (attractive, -g_M)': e_M is attractive where 3 n_R^2 > n_0^2, i.e. for k_F up to about m,"
echo "    and repulsive in the relativistic part of the table (T4c, last column)."
echo "11. 'P_b = n e_b' - e_b against the lattice pressure by direct differentiation' is read as e_b' by central differences of e(n) at"
echo "    fixed b against the Fermi level (P = n mu - e by thermodynamics). At the most dilute filling with m b = 1, P and b d_b e are small"
echo "    differences of large terms and the finite-difference checks are limited by rounding there (T4b gives the medians)."
echo
sed '1,2d' $O2/RESULT_tables_v1.md
echo
sed '1,2d' $O3/RESULT_tables_v1.md
} > $R
echo "wrote $R"
