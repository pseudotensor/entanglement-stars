#!/bin/bash
# Strand 5 J1 result (assembly v1): tables from J1run3 (strand5_star_v4.py) by strand5_star_report_v2.py on the box; fetches the records;
# writes solve/results/RESULT_strand5_continuum.md.
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s5_star
OUT=$W/solve/results/strand5/J1run3
PY=${QI2_ROOT}/python
mkdir -p $OUT
timeout 300 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 $PY code/strand5_star_report_v2.py --json J1run3/star_v4.json --md J1run3/RESULT_tables_v2.md > J1run3/report_v2.log 2>&1"
for f in status.txt run.log star_v4.json star_v4_profiles.npz RESULT_tables_v2.md; do timeout 120 scp -q compute-host:$B/J1run3/$f $OUT/ || echo "missing $f"; done
R=$W/solve/results/RESULT_strand5_continuum.md
{
echo "# Strand 5 result J1, revision 1: the continuum control (T1 the family, T2 its controls)"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand5_kernel_star.md sec. 3 J1 and sec. 6 (lead, 2026-10-14); LEAD.tex sec. 8. Numbers,"
echo "controls and commands only; no reading. Tables emitted by solve/code/strand5/strand5_star_report_v2.py."
echo
echo "## Inputs and commands"
echo "- Code: solve/code/strand5/strand5_star_v4.py (the equations, both solvers and every check are in its docstring), the report"
echo "  strand5_star_report_v2.py, the launcher launch_strand5_J1_v3.sh (box compute-host, conda env qi2, CPU, 2 threads, about 80 s)."
echo "  Records: solve/results/strand5/J1run3/ (star_v4.json, star_v4_profiles.npz with the profiles of a few members, run.log)."
echo "- Earlier versions, kept and superseded: v1 (its grid Newton did not converge: the first face's geometric-mean area vanished at r = 0 and"
echo "  decoupled the centre node); v2 (unknowns psi, chi: the tail readout of the charges amplified the Newton floor, and its 8000-node"
echo "  continuation stalled after one member; box run J1run1); v3 (deviation unknowns; no fold indicator; box run J1run2, whose numbers v4"
echo "  reproduces)."
echo
echo "## Readings and conventions taken by the builder"
echo "1. Units G = m = 1, k = 1/(2 pi), g = 16. P16's units: M_0 = R_0 = (g/2)^(-1/2) in these units (P16's nu = g/2 = 8, as the spec"
echo "   states); T2b gives the g = 2 maximum directly (the OV normalization, M_0 = R_0 = 1) and the ratio of the two maxima."
echo "2. Two solvers. The reference integrates the same equations as ODEs from the centre and uses the isotropic chart's two exact scalings"
echo "   to impose psi, N psi -> 1; it carries the integrals as ODE components. The spec's solver is Newton on the grid (finite volumes, Robin"
echo "   exterior at R_max, continuation in z_c); its first member starts from the reference profile, every later member from the previous one."
echo "3. Grid: R_max = 50 r_s of the previous member (T1b lists R_max/r_s per member), r_s placed at s = 0.6 of the uniform map variable."
echo "4. Charges: A, B from the tail at R_max (the spec's definition) and from the volume integrals (1/(4 pi k)) int psi^5 e and"
echo "   (1/(4 pi k)) int N psi^5 (e + 6P); the discrete Gauss law makes them equal at convergence (T1b)."
echo "5. C5: at fixed z_c (the family parameter) no singularity is expected at the mass maximum; at fixed N_B the turning point is"
echo "   singular. The specified sigma_min of the row-scaled Jacobian is set at both rows by the Laplacian's long-range mode on [0, 50 r_s]"
echo "   (compare the vacuum value in T2c), so it cannot show the fold; the determinant's sign at fixed z_c and at fixed N_B is added as"
echo "   the fold indicator, beside the sign changes of dN_B/dz_c and dM/dz_c."
echo "6. Trapping (the spec's r_s > 2A): in the isotropic chart the exterior lapse f = (r - A)/(r + A) vanishes at r = A, the areal"
echo "   R = 2GM; T2c gives the family's smallest r_s/A, r_s/(2A) and R_s/(2GM)."
echo "7. 2GM/R_s uses the areal surface radius R_s = r_s psi(r_s)^2: P16's printed 2GM/R_s equals 2 M_max/R_s with its own printed R_s,"
echo "   so its R_s is the areal radius."
echo "8. The first law by centred steps in z_c: 1e-5 z_c on the reference, 1e-4 z_c on the grid (on the member's own grid)."
echo
sed '1,2d' $OUT/RESULT_tables_v2.md
} > $R
echo "wrote $R"
