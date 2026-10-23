#!/bin/bash
# Strand 2d, result revision 2 (assembly v5: report v9 with SPEC secs. 9 and 10: the B4 (0.2, 1) rerun B4run2, the 2b cross-check on C8,
# and the sec. 10 items (sec10/: shells, heavy, lowest31, c8dressed31; jnn1/: J_NN by the sum over states).
# Assembly v4 (not run): report v8 with sec. 9 only.
# Assembly v3 (revision 1): report v7; the vacuum run's device stated as run. generate the tables on the box (strand2d_report_v8.py for C8run2 + prep31_C8 and B4run1 +
# prep31_B4, with the cube's stage 2 (prep31dr2_C8), the dressed (1,1) vacuum cost (vac11: the C8 check and B4) and control C7 (c7)),
# fetch them with the library checks, run records and JSON, and assemble solve/results/RESULT_strand2d_jet.md (text assembly only).
set -eu
W=${QI2_ROOT}/.wt-solve_a
B=${QI2_ROOT}/s2d_jet
OUT=$W/solve/results/strand2d
PY=${QI2_ROOT}/python
mkdir -p $OUT
timeout 600 ssh -n -o BatchMode=yes compute-host "cd $B && OMP_NUM_THREADS=1 $PY code/strand2d_report_v9.py --run C8run2 --prep prep31_C8 --dressed prep31dr2_C8 --vac11 vac11 --c7 c7 --b2c8 ../s2b_row/c8pass1v5 --sec10 sec10 --lattice C8 --md RESULT_strand2d_C8_tables.md > report_C8.log 2>&1 && OMP_NUM_THREADS=1 $PY code/strand2d_report_v9.py --run B4run1 --prep prep31_B4 --dressed prep31dr2_C8 --vac11 vac11 --c7 c7 --sec9 B4run2 --sec10 sec10 --jnn jnn1 --lattice B4 --md RESULT_strand2d_B4_tables.md > report_B4.log 2>&1 && tar czf strand2d_json.tar.gz C8run2/*.json B4run1/*.json prep31_C8/*.json prep31_B4/*.json prep31dr2_C8/*.json vac11/*.json c7/*.json B4run2/*.json sec10/*.json jnn1/*.json check2/check.json check3/check.json"
for f in RESULT_strand2d_C8_tables.md RESULT_strand2d_B4_tables.md strand2d_json.tar.gz report_C8.log report_B4.log; do
  timeout 300 scp -q compute-host:$B/$f $OUT/ || echo "missing $f"
done
for d in C8run2 B4run1 prep31_C8 prep31_B4 prep31dr2_C8 vac11 B4run2 sec10 jnn1; do
  mkdir -p $OUT/$d && timeout 120 scp -q compute-host:$B/$d/status.txt $OUT/$d/ || echo "missing $d/status.txt"
done
timeout 120 scp -q compute-host:$B/check2.log $OUT/ || true
timeout 120 scp -q compute-host:$B/check3.log $OUT/ || true
mkdir -p $OUT/c7
for f in vac11/vac11_C8.log vac11/vac11_B4.log c7/c7_C8_early.log c7/c7_B4_early.log c7/status_early.txt; do
  timeout 120 scp -q compute-host:$B/$f $OUT/$f || echo "missing $f"
done
R=$W/solve/results/RESULT_strand2d_jet.md
{
echo "# Strand 2d result, revision 2 (test T7): the fixed-readout jet of the total energy at the flat point, on C8 and B4"
echo
echo "Builder: analyst. Spec: solve/specs/SPEC_strand2d_jet.md with the lead's rulings of secs. 7 to 9. Numbers, controls and commands only;"
echo "revision 2 adds sec. 9 (the B4 (0.2, 1) jets rerun at smaller steps with the cluster average, B4run2; 2b's cube dilation derivatives"
echo "beside the C8 jet) and sec. 10 (the C3 table with the dressed (3,1)' term, T3's finite-q columns, the first-order mode change, the"
echo "lowest (3,1)' levels and the ordering condition, J_NN by the exact sum over states, the shell closed forms, the heavy packets' fixed"
echo "and relaxed vacuum costs, the rerun's norm defect; item 6 not computed). Revision 1's tables are unchanged below except where secs. 9"
echo "and 10 add tables; no physics reading. Tables emitted by solve/code/strand2/strand2d_report_v9.py."
echo
echo "## Inputs"
echo "- Code (solve/code/strand2/): strand2d_lib_v1.py (the B4 box; the (2,0) block of the light packet in torch with the seed's (2,2)"
echo "  projection built from its Kronecker structure; the common-sea Schur overlaps), strand2d_jet_v1.py (B4 run) and strand2d_jet_v2.py"
echo "  (C8 run; v2 also stores the lowest d_0 + 1 levels at every stencil point), strand2d_prep31_v1.py (sec. 7(2) stage 1: z, the"
echo "  (3,1)' amplitudes, their free evaluation, the fixed-vacuum cost, the Pauli and norm-defect checks), strand2d_prep31_dressed_v2.py"
echo "  (stage 2 on C8: the same terms with the completion through the gate's q-block route; at (0, 0) it reproduces stage 1),"
echo "  strand2d_vac11_dressed_v1.py (sec. 8(2): the dressed (1,1) fixed-vacuum cost; the completion's particle-hole block built from the"
echo "  seed's Kronecker terms; checked on C8 against the q-block route, run on B4 on the CPU), strand2d_c7_v1.py (control C7, sec. 8(1) and"
echo "  the momentum support of sec. 8(3)), strand2d_sec10_small_v1.py (sec. 10 items 2, 4, 5 and the pp check), strand2d_jnn_v1.py (sec. 10
  item 3, GPU), strand2d_jet_v4.py (the sec. 9 rerun), strand2d_report_v9.py; the"
echo "  library checks strand2d_check_v2.py (parts a, b on CPU: against strand2b_c8_lib_v3 on C8; part c: B4 timing on GPU 2)."
echo "- Library checks (check2.log, check3.log in solve/results/strand2d/):"
sed 's/^/    /' $OUT/check2.log
sed 's/^/    /' $OUT/check3.log
echo "- Box compute-host, conda env qi2; C8 on CPU; the B4 jets and stage 1 on GPU 2 (torch complex128); the B4 dressed vacuum cost and C7 on the CPU."
echo "- Records: solve/results/strand2d/strand2d_json.tar.gz (all jet, prep31, vac11, C7 and check JSON)."
echo
echo "## Commands"
echo "C8: launch_strand2d_C8_v2.sh (C8run2) and launch_strand2d_prep31_C8_v1.sh (prep31_C8); B4: launch_strand2d_B4_v1.sh (B4run1) and"
echo "launch_strand2d_prep31_B4_v1.sh (prep31_B4, queued behind B4run1 on GPU 2); C8 stage 2: launch_strand2d_prep31dr_C8_v2.sh (prep31dr2_C8);"
echo "Sec. 9 rerun: launch_strand2d_B4_sec9_v1.sh (B4run2: strand2d_jet_v4.py --lattice B4 --couplings 0.2,1 --hs 1e-4,1e-5,1e-6"
echo "--qs 0,pi/2,pi --nlevels 125 --d-cluster 120 --skip-vacuum, on a GPU free at launch)."
echo "Sec. 10: launch_strand2d_sec10_small_v1.sh (sec10/: the parts shells, heavy, lowest31 and c8dressed31 on the CPU) and"
echo "launch_strand2d_jnn_v1.sh (jnn1/, queued behind B4run2, on a GPU free at launch)."
echo "C7: launch_strand2d_c7_early_v1.sh (c7/, in a spare thread; the small chain's own C7 call later rewrites the same JSON); the dressed"
echo "(1,1) vacuum cost: launch_queue_slotC_v2.sh (the C8 check) and launch_queue_slotC_v3.sh (B4 on the CPU: GPU 2 had been taken by"
echo "another lane's process); then strand2d_report_v8.py."
for d in C8run2 B4run1 prep31_C8 prep31_B4 prep31dr2_C8 vac11 B4run2 sec10 jnn1; do echo "Run record $d:"; sed 's/^/    /' $OUT/$d/status.txt; done
echo "Run record c7 (status_early.txt):"; sed 's/^/    /' $OUT/c7/status_early.txt
echo "B4 dressed vacuum cost, resources (time -v):"; grep "Elapsed\|Maximum resident" $OUT/vac11/vac11_B4.log | sed 's/^/    /' || true
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
echo "  J^prep,vac free is 2 sum (eps_k + eps_m)|z_km|^2; dressed (sec. 8(2)) it is 2 sum (eps_k + eps_m)|u_km|^2 with u = e^S z, S the"
echo "  completion on the (1,1) block (the dressed annihilators act as the identity on the one-excitation blocks, where S vanishes); on B4"
echo "  this is computed directly, with the cube's ratio beside it."
echo "- The B4 jet run is version 1: it records the multiplet width and the gap at every point but not the full list of the lowest"
echo "  d_0 + 1 levels (the C8 run, version 2, does); the sec. 9 rerun (version 4) records the lowest 125 levels at every point."
echo "- Sec. 10 item 1(a) on B4 uses the free (3,1)' term times the cube's dressed/free ratio (an estimate, not a bound, as sec. 10 states);"
echo "  item 1(b) takes c = 1/2 for a cosine mode and c = 1 at q = pi; item 2 on B4 uses the free (3,1) block's lowest level as the stated"
echo "  proxy; item 7's z step agreement is the stage-1 column (the rerun stores no (3,1) amplitudes); item 6 was not computed."
echo
sed '1,2d' $OUT/RESULT_strand2d_C8_tables.md
echo
sed '1,2d' $OUT/RESULT_strand2d_B4_tables.md
} > $R
echo "wrote $R ($(wc -l < $R) lines)"
