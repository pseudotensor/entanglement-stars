#!/bin/bash
# Strand 2c addendum result (SPEC_strand2c_addendum.md), revision 1: generate the tables on the box (part A: strand2c_kernel_report_v2.py
# over kernel1; parts B and C: strand2c_addendum_BC_report_v1.py over addBC1; the cube's P2 profiles: strand2c_report_v3.py over run2 with
# --p2 run4c, keeping only the cube P2 rows), fetch them with the run records and JSON, and assemble
# solve/results/RESULT_strand2c_addendum.md (text assembly only).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2c_mismatch
OUT=$W/solve/results/strand2c_addendum
PY=${QI2_ROOT}/python
mkdir -p $OUT/kernel1 $OUT/addBC1 $OUT/run4c
timeout 900 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 nice -n 10 $PY code/strand2c_kernel_report_v2.py --json kernel1/kernel_L256.json --md RESULT_strand2c_addA_tables.md > report_addA.log 2>&1; OMP_NUM_THREADS=1 nice -n 10 $PY code/strand2c_addendum_BC_report_v1.py --json addBC1/addBC.json --md RESULT_strand2c_addBC_tables.md > report_addBC.log 2>&1; OMP_NUM_THREADS=1 nice -n 10 $PY code/strand2c_report_v3.py --run run2 --p2 run4c --md RESULT_strand2c_tables_cubeP2.md > report_cubeP2.log 2>&1; tar czf strand2c_addendum_fields.tar.gz run4c/w_L*.npz 2>/dev/null || true"
for f in RESULT_strand2c_addA_tables.md RESULT_strand2c_addBC_tables.md RESULT_strand2c_tables_cubeP2.md report_addA.log report_addBC.log report_cubeP2.log strand2c_addendum_fields.tar.gz; do
  timeout 300 scp -q compute-host:$B/$f $OUT/ || echo "missing $f"
done
for f in status.txt kernel_L256.json kernel.log; do timeout 300 scp -q compute-host:$B/kernel1/$f $OUT/kernel1/ || echo "missing kernel1/$f"; done
for f in status.txt addBC.json addBC.log; do timeout 300 scp -q compute-host:$B/addBC1/$f $OUT/addBC1/ || echo "missing addBC1/$f"; done
for f in status.txt p2_profiles_C8.json profiles.log mismatch_P2_L8.json mismatch_P2_L12.json mismatch_P2_L16.json P2_L8.log P2_L12.log P2_L16.log; do
  timeout 120 scp -q compute-host:$B/run4c/$f $OUT/run4c/ || echo "missing run4c/$f"
done
# keep the cube P2 rows of the regenerated 2c tables (headings, table headers, rows naming P2_C8); drop the sections without P2 rows
python3 - "$OUT/RESULT_strand2c_tables_cubeP2.md" "$OUT/cubeP2_rows.md" <<'EOF'
import sys
src, dst = sys.argv[1], sys.argv[2]
try:
    lines = open(src).read().split("\n")
except OSError:
    open(dst, "w").write("(the cube P2 tables are missing)\n")
    sys.exit(0)
keep_sections = ("## T2 Interior profiles", "### T2 exterior profile", "## C5 Box convergence", "## T3 Delta C_x", "## T4 Scaling")
out, on, hdr = [], False, 0
for ln in lines:
    if ln.startswith("## ") or ln.startswith("### "):
        on = ln.startswith(keep_sections)
        hdr = 0
        if on:
            out += ["", "#" + ln]
        continue
    if not on:
        continue
    if ln.startswith("|"):
        hdr += 1
        if hdr <= 2 or "P2_C8" in ln:
            out.append(ln)
    elif ln.strip():
        out.append(ln)
    else:
        hdr = 0
open(dst, "w").write("\n".join(out) + "\n")
EOF
R=$W/solve/results/RESULT_strand2c_addendum.md
{
echo "# Strand 2c addendum result, revision 1: the vacuum kernel in momentum space, the harmonic-exterior profiles, the bump width; and the cube's P2 profiles"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand2c_addendum.md (lead, 2026-10-14 06:19 UTC); the cube's P2 profiles complete"
echo "RESULT_strand2c_mismatch.md (revision 1 had the ring4 ones). Numbers, controls and commands only; no physics reading. Tables emitted by"
echo "solve/code/strand2/strand2c_kernel_report_v2.py (A), strand2c_addendum_BC_report_v1.py (B, C) and strand2c_report_v3.py (cube P2)."
echo
echo "## Inputs"
echo "- Code (solve/code/strand2/): strand2c_kernel_v1.py (part A: W_0 at second order in eps in momentum space, per packet and grouped as"
echo "  the real-space code defines W_0; the dense real-space control at L 8 and 12 with eps 2e-2, 1e-2, 5e-3 extrapolated twice), the"
echo "  report strand2c_kernel_report_v2.py; strand2c_addendum_BC_v1.py (parts B and C with 2c's packet routine strand2c_mismatch_v2,"
echo "  GPU) and its report strand2c_addendum_BC_report_v1.py; for the cube P2: strand2c_p2_profiles_v1.py over the cube pass 1"
echo "  (s2b_row/c8pass1v5) and strand2c_mismatch_v3.py --only-p2 at L 8, 12, 16."
echo "- Box compute-host, conda env qi2; part A on CPU (two threads); B, C and the cube P2 on GPU 2."
echo "- Records: solve/results/strand2c_addendum/ (kernel1/, addBC1/, run4c/: JSON, logs, run records; the cube P2 w fields in"
echo "  strand2c_addendum_fields.tar.gz)."
echo
echo "## Commands"
echo "Part A: launch_small_after_prep31B4_v1.sh (kernel1: strand2c_kernel_v1.py --out kernel1 --control --L 256). Parts B and C:"
echo "launch_queue_slotC_v5.sh (addBC1: strand2c_addendum_BC_v1.py --profiles run3/p2_profiles_R4.json --kernel kernel1/kernel_L256.json"
echo "on a GPU free at launch). The cube P2: run4c: strand2c_p2_profiles_v1.py --c8 s2b_row/c8pass1v5, then strand2c_mismatch_v3.py --only-p2"
echo "--L 8, 12, 16 on a GPU free at launch; launch_strand2c_P2_C8_v3.sh, after slot C v5 ran the step from the wrong working"
echo "directory into run4b, computing nothing)."
for d in kernel1 addBC1 run4c; do echo "Run record $d:"; sed 's/^/    /' $OUT/$d/status.txt 2>/dev/null || echo "    (missing)"; done
echo
echo "## Readings and conventions taken by the builder"
echo "- Part A: k_a(q) = -2 W^(2)/(q_lat^2 L^3) and mixed = -W^(11)/(q_lat^2 L^3), with W^(2) the coefficient of eps^2 and W^(11) that of"
echo "  eps eps_N in W_0 (2c's T1 definitions at eps -> 0), q_lat^2 = 2 (1 - cos q). k_a(0; aH) is the a + b q^2 fit over"
echo "  q <= aH/4 when at least 4 grid points lie there, else the value at the smallest q (stated per row). The physical row takes the"
echo "  light packet massless with the twist named in its heading."
echo "- Part B: the open box has no wrapping links; the boundary layer is every site with a coordinate 0 or L - 1; the harmonic"
echo "  extension solves the 6-neighbour Laplace equation on the remaining sites with I and the boundary layer as Dirichlet data; the"
echo "  Gauss charge uses the open-box Laplacian. Only the ring4 cases are in B (the spec's P2 cases); the cube's P2 is on the periodic"
echo "  boxes as in 2c."
echo "- Part C: the periodic L = 16 box; the bump centred at (8, 8, 8) with minimum-image distances; the row +k psi Lap(psi - 1) (the"
echo "  lead's sign)."
echo
sed '1,2d' $OUT/RESULT_strand2c_addA_tables.md 2>/dev/null || echo "(part A tables missing)"
echo
sed '1,2d' $OUT/RESULT_strand2c_addBC_tables.md 2>/dev/null || echo "(parts B and C tables missing)"
echo
echo "## The cube's P2 profiles (rows of the 2c tables for the cube's 2b solutions; ring4 rows are in RESULT_strand2c_mismatch.md)"
cat $OUT/cubeP2_rows.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
