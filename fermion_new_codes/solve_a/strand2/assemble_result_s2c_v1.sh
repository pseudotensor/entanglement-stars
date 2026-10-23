#!/bin/bash
# Strand 2c, result revision 1: generate the tables on the box (strand2c_report_v3.py over run2 and run3), fetch them with the run
# records and the field archive, and assemble solve/results/RESULT_strand2c_mismatch.md (text assembly only).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2c_mismatch
OUT=$W/solve/results/strand2c
PY=${QI2_ROOT}/python
mkdir -p $OUT/run2 $OUT/run3
timeout 600 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 nice -n 10 $PY code/strand2c_report_v3.py --run run2 --p2 run3 --md RESULT_strand2c_tables.md > report_v3.log 2>&1 && tar czf strand2c_fields.tar.gz run2/w_L*.npz run3/w_L*.npz"
for f in RESULT_strand2c_tables.md report_v3.log strand2c_fields.tar.gz; do
  timeout 300 scp -q compute-host:$B/$f $OUT/ || echo "missing $f"
done
for f in status.txt mismatch_L8.json mismatch_L12.json mismatch_L16.json L8.log L12.log L16.log check_gpu.log check_cpu.log; do
  timeout 120 scp -q compute-host:$B/run2/$f $OUT/run2/ || echo "missing run2/$f"
done
for d in check_gpu check_cpu; do
  mkdir -p $OUT/run2/$d && timeout 120 scp -q compute-host:$B/run2/$d/mismatch_L4.json $OUT/run2/$d/ || echo "missing run2/$d"
done
for f in status.txt p2_profiles_R4.json profiles.log mismatch_P2_L8.json mismatch_P2_L12.json mismatch_P2_L16.json P2_L8.log P2_L12.log P2_L16.log; do
  timeout 120 scp -q compute-host:$B/run3/$f $OUT/run3/ || echo "missing run3/$f"
done
R=$W/solve/results/RESULT_strand2c_mismatch.md
{
echo "# Strand 2c result, revision 1 (test T5): the mismatch Delta C_x between the exact lattice vacuum row of W_{0,a} and the installed two-derivative row"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand2c_mismatch.md (lead, 2026-10-14). Numbers, controls and commands only; no"
echo "physics reading. Revision 1: P1, P3 and k_a(q) at L = 8, 12, 16; P2 on the ring4 2b solutions at L = 8, 12, 16; the cube's P2"
echo "profiles follow when 2b's C8 (2,0) cases exist. Tables emitted by solve/code/strand2/strand2c_report_v3.py."
echo
echo "## Inputs"
echo "- Code (solve/code/strand2/): strand2c_mismatch_v2.py (run2: controls, k_a(q), P1, P3; version 2 is version 1 with the GPU"
echo "  path's dtypes fixed), strand2c_mismatch_v3.py (run3: --only-p2, the P2 profiles alone, each record carrying its 2b case and"
echo "  matter source), strand2c_p2_profiles_v1.py (picks the 2b solutions), strand2c_report_v3.py (tables; version 3 adds the"
echo "  field-level control C3'; version 2 was a failed patch and was removed); frozen native modules in frozen/."
echo "- Box compute-host, conda env qi2; torch eigh on GPU 0 (L = 16, run2 and run3) and GPU 2 (L = 8, 12, run2); the GPU against"
echo "  CPU check at L = 4 is in the tables."
echo "- Records: solve/results/strand2c/run2/ and run3/ (JSON per box size and logs); the w fields (npz) in strand2c_fields.tar.gz."
echo
echo "## Commands"
echo "solve/code/strand2/launch_strand2c_v2.sh in ${QI2_ROOT}/s2c_mismatch (run2/), then launch_strand2c_P2_R4_v2.sh"
echo "(run3/; it waited for run2 and for GPU 0 to have no compute process), then strand2c_report_v3.py --run run2 --p2 run3."
echo
echo "Run record, run2 (status.txt):"
sed 's/^/    /' $OUT/run2/status.txt
echo "Run record, run3 (status.txt):"
sed 's/^/    /' $OUT/run3/status.txt
echo
echo "## Readings and conventions taken by the builder"
echo "- The interior set I (ring4's 4 positions or the cube's 8) is placed at the box centre: site (L/2, L/2, L/2) plus the"
echo "  positions; K_I = native_core.K_matrix(positions of I)."
echo "- k_a(q) and the mixed coefficient use psi = 1 + eps cos(q x_1), N = 1 + eps_N cos(q x_1) along the first coordinate, eps ="
echo "  eps_N = 1e-3, q = 2 pi/L; the mixed coefficient is W_0(N_q, psi_q) - W_0(1, psi_q), the term bilinear in eps and eps_N."
echo "- P3 (smooth bump over the box): the installed row with K_I replaced by the periodic Laplacian is read literally as"
echo "  -k_lat psi Lap(psi - 1); since K_I is a positive operator and Lap a negative one, the other sign, +k_lat psi Lap(psi - 1)"
echo "  (K_I <-> -Lap), is tabulated beside it, flagged: the lead decides which is meant."
echo "- P2 profiles: 2b's R4 solutions at m 0.1, (0.2, 1), Lambda 1, k 1 and 10, blocks (2,1) and (1,1) (the dressed"
echo "  stellar-type states of SPEC_strand2b_row sec. 9) and (1,0) as the one-particle reference; psi and N on I, 1 outside."
echo "- C3 (isospectrality per packet) is computed from the eigenvalues at L <= 12 on the flat background; at every L the"
echo "  field-level form sum_x w_x/psi_x = sum_s eta_s Tr(E_s1 - E_s0) = 0 is checked on every saved field (C3'). For P1 it fixes"
echo "  the leak at -(sum of w on I)/(1 + eps), so the leak column and the interior sum are not independent there."
echo
sed '1,2d' $OUT/RESULT_strand2c_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
