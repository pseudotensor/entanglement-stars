#!/usr/bin/env python
"""engine_collapse_v31.py (builder analyst; a diagnostic for engine spec 30.3's sigma/a 2 member, not a rule change: version 30's
damped test (DAMP30, step 0.5) shows the iteration's residual grow by a factor 1.16 per iteration after its minimum, so the undamped map has
an eigenvalue near +1.3 that no damping removes, beside the plain map's period-2 overshoot; whether a self-consistent solution exists at
the 2-site kernel is then a question for a Newton-type solve): version 30 with --mix newton: per record, the fixed point of the same map
x -> G(x) (the eigendecompositions at the applied fields x = (N, v), the continuation, the sources, one field solve; version 30's loop
body, unchanged) is solved by scipy.optimize.newton_krylov on F(x) = G(x) - x (LGMRES, --nk-inner Krylov vectors, at most --nk-maxiter
Newton steps, f_tol the stop rule's field tolerance in the max norm), started from the first solve's output (the plain iteration's first
applied fields); every evaluation of F enters the record's iteration_history (finite-difference evaluations included) and the
smallest-residual evaluation is kept, as in version 30; one plain evaluation at that iterate's output then gives |dE/E| and a second
residual, and the record is converged when the kept residual is below --stop-fields and that |dE/E| below --stop-E. --mix plain, anderson
and damped are version 30's (the defaults are version 30 bitwise). The record line and the head name the mixer. Nothing else changes;
version 30's docstring follows.
engine_collapse_v30.py (builder analyst; a diagnostic for engine spec 30.3's sigma/a 2 member, not a rule change: the member's
iteration history (results/engine/LAD27/residual_history_sigma2_pod1copy_v1.md; GATE27 lad_e2) shows the undamped plain iteration reach
its smallest residual at iterations 1 to 3 and then grow, at T/rs 0 as a period-2 alternation whose even and odd iterates each double
every two iterations, the signature of an overshooting iteration): version 29 with --mix damped and --mix-alpha a: the applied fields of
iteration k + 1 are N_app + a (N_out - N_app) and v_app + a (v_out - v_app) (a = 1 is the plain iteration up to rounding); --mix plain and
anderson are version 29's, so the defaults are version 29 bitwise; the record line and the head name the mixer and a. Nothing else
changes; version 29's docstring follows.
engine_collapse_v29.py (builder analyst; engine spec 30.3: "on the box CPU, re-solves the member's first 5 records with --stop-fields 1e-5
and the field solver's tolerance tightened tenfold (about 15 minutes) to say whether the floor moves with the solver"): version 27 with
the field solve's integration tolerances as options (--field-rtol, --field-atol; defaults 1e-12 and 1e-16, version 27's, so the defaults
are version 27 bitwise); the record line names them. Nothing else changes; version 27's docstring follows.
engine_collapse_v27.py (builder analyst; the goal of 2026-10-16, task 2, inbox #3: "The averaging-scale control of this task is a
fixed-preparation ladder: one initial covariance C_0 (prepared once, at one averaging scale ...) evolved at 3 averaging scales a factor 2
apart on one engine version with the same stepping and the same continuation convention, the sources and fields re-averaged at each scale
from that common C_0 ... The ladder's kernel is stated by its shape and its per-axis variance sigma^2 (this lane's Gaussian of standard
deviation ell has sigma^2 = ell^2, plus the cell's a^2/12 where the cell average enters) ... the ladder runs at sigma/a 2, 4 and 8"):
version 25 with --prep-ell P: the initial data (the levitating well's filling, the C-init fields and the dust orbitals, version 5's
preparation) solved at the averaging scale P, the run's own --ell used for every solve after it, so runs with one P and different --ell
start from one C_0 (the T = 0 record keeps the C-init fields as its applied fields, as in version 25; the first step re-solves the fields
at --ell from the propagated state); --prep-ell absent or equal to --ell is version 25 (the gate: bitwise on stored records). The record
line names the kernel by its per-axis variance: "kernel: the Gaussian of standard deviation ell, sigma^2 = ell^2 (plus a^2/12 where the
cell average enters); prepared at ell P". Nothing else changes; version 25's docstring follows.
engine_collapse_v25.py (builder analyst; engine spec 25.1 and 25.7: "every new launch runs on v25"): version 24 on engine_ops_v3 (the
reference continued in one step from the previous accepted record past the Weyl test, the sliding window retired; a partly occupied
block carries a Weyl gap of zero; whole-block widths; straddling blocks sign free; ties at the cut by the predecessor or flagged;
inversions by block order), the record line "engine_fields_v16; v25 (ops v3); continuation: the LZ-inspired maximum-overlap
convention (c_LZ = ..., the lane's declaration); rows: raw, floor in the stress's trace (25.3)", and --resume-from accepting an
engine_collapse_v19 or v24 run (resume_state_v25): its own engine_collapse_v17 parent replayed by v17's rule to its resume record (the
version-19 path), then its native records replayed with its own rule (engine_ops_v2.continue_sea_v2 with its c_LZ and window) on its
stored applied fields to the requested record, the masks compared bit for bit with its stored masks and the sources with its stored
shell totals (acceptance 1e-8), the run continuing from the next record on ops v3 with this run's c_LZ. Nothing else changes.
Version 24's docstring: engine_collapse_v24.py (builder analyst; engine spec 24.1 and 24.6: "every new launch runs on v16"): version 19 on engine_fields_v16 (the
zeros of U by the s = sqrt(U) stretch, no step-failure rule; bitwise version 14 on the plain path, the gate) and engine_stress_v2 (shell 0's
regular S_r, i_2's series to z^8 below 0.5), with the record line stating the lane's declarations (22.0, 24.1 (2)): "engine_fields_v16;
v24; tau = ...; continuation: verified (Weyl) | LZ convention (c_LZ = ..., the lane's declaration); stresses: the lane's declared stress
(the Noether flux of Q at unit clock and zero flow); rows: flat-slice ADM, K^ADM = -K, raw (the static floor 24.2); J_ell = <P_r> = -J_r of
the goal's display; gauge: ...". Nothing else changes. Version 19's docstring: (builder analyst; engine spec 21.2, 21.3 and 21.6 with the items 20.1, 20.2 and 20.6 assigned to a version 18
folded in here: the name engine_collapse_v18.py holds the builder's draft on engine_fields_v13 (one test record, C3eq/v17 t18), withdrawn
with the floor rule (20.2), so the lead's v18 items enter this version). Version 19 = version 17 with:
(a) the fields from engine_fields_v14 (20.2: the momentum equation kept everywhere, the series to r^5, N_centre = exp(-ln N_c), the regime
diagnostic, the flat-slice gauge continued and labelled where U < -1e-12 max U, never rejected; padded_collar_cost checks its integration);
the r_0 -> r_0/2 check on every --delta-rtol-every-th record beside delta_rtol;
(b) R2's continuation by engine_ops_v2.continue_sea_v2 (21.2, the lane's declaration, the owner's confirmation asked): Landau-Zener
matched, by maximum overlap with degenerate blocks rotated to their weight eigenbasis, the count exact, never rejected; no halving and no
verification through subdivisions (version 12's midpoint check compared the endpoint continuation with the halving one, which is no longer
the rule); per channel the reference windows of the last --window-max (4) accepted records; per record and channel the mode (sub-steps or
window), nsub, m, s_T, tau_LZ, the window's consistency, the Weyl verification and its margin, the intermediate weights, the smallest
occupied and largest empty weight, the inversions, the pockets, the block tolerance; the sticky labels "continuation: verified (Weyl)" |
"LZ convention (c_LZ = ...) from T/rs ..." and "reference degraded" (smallest occupied weight below 0.5); whether the occupation mask equals
the negative-energy set (bitwise);
(c) the self-consistency iteration (version 17's, the lagged first iterate, plain or Anderson) stops at --maxit (20) with the
smallest-residual iterate returned, flagged "unconverged (residual r)", sticky, the run continuing (20.2);
(d) per record the handoff quantities (20.1) on the accepted state, computed from the traces in hand (no extra eigendecomposition): the
stresses of engine_stress_v1 (the Noether flux of the lane's declaration) for the dust and the pairs apart (the face flux F, the
centrifugal force f, the mass-term density b, the kappa-weighted sigma_y for the continuum form of S_t), the balance identity's largest
difference, the virial check, the smoothed S_r,ell and S_t,ell of the total at the coarse nodes with the trace identity's error, the K_0
averages <N^2>_n of the dense lapse, the shell energies of the propagated sea and of R2 apart (the Gauss balance's evolution and reset
parts per interval); the evolution rows are computed afterwards from these records by engine_rows_v1; per channel the occupation mask
(packed bits) and the eigenvalues at the accepted fields; the orbitals at the slice times (--slices-c128, --slices-c64 in T/rs: the sea's
and the dust's columns, R2's, the eigenvalues and the mask per channel, the applied fields) into --slice-dir;
(e) --resume-from dir/tag --resume-T-over-rs x (21.3): the stored engine_collapse_v17 run is replayed exactly by its own rule
(engine_collapse_v17.continue_v17 with the stored applied fields, engine_replay_v1's loop) to its record at T/rs = x, the sources at every
replayed record compared with the stored shell totals (the run stops when the largest relative difference exceeds 1e-8), the state
propagated past that record, the stored records and fields up to it kept in this run's files, and the run continued from the next record
under (a) to (d). The record line reads "engine_fields_v14; tau = ...; stresses: Noether (lane's declaration); rows: flat-slice ADM,
K^ADM = -K; J_ell = <P_r> = -J_r of the goal's display; gauge: intact | continued on [r1, r2]; continuation: ...". Version 17's
docstring follows.
engine_collapse_v17.py (builder analyst; the economy engine of SPEC_final4b_engine.md 18.5 with the continuation items 18.1 assigned to a
version 16, folded in here: no separate version 16 was built, the r_s 64 primary having moved to the H100 on version 15 by the owner's
addendum of 20:34 UTC. Naming: the lead's "engine_fields_v10" is this lane's engine_fields_v12 (solve_constraints_ivp with the series below
r_0; the file engine_fields_v10.py is the earlier shooting and averaging module). Version 17 = version 15 with:
(b) the constraints solved by engine_fields_v12.solve_constraints_ivp (U' = 8 pi g r^2 (rho_ell + q J_ell), (ln N)' = -4 pi g r J_ell/q,
q = -sqrt(U/r), one outward integration from the centre's series start, N normalized at the collar by one division: no sweep, no secant,
no iteration limit), in the initial data's preparation as in every re-solve;
(c) the self-consistency iteration as R2's dependence on the fields alone (the source operators are field independent): iteration 0 takes
the previous applied fields and their R2 (no eigendecomposition) and gives fields_1 by one solve with the propagated state; iteration
k >= 1 diagonalizes the channels at fields_k, continues R2 from the previous accepted configuration to fields_k and gives fields_{k+1} by
one solve; stop when |E_{k+1} - E_k|/E < --stop-E (1e-7) and max |N_{k+1} - N_k|, max |v_{k+1} - v_k| < --stop-fields (1e-6) over the
sites; the accepted configuration is (fields_k, R2_k), propagated with the eigendecomposition computed for R2_k; the record's N, v and E
are the solve from its sources and N_applied, v_applied the accepted fields (version 15's convention); --mix plain (the lead's iteration)
or anderson (type II with beta 1, whose first step is the plain one; labelled); the sea's and the dust's densities once per re-solve (the
reference's alone change between iterations); per record the iteration history (|dE/E|, max dN, max dv per iteration), the number of
eigendecompositions and the timing of the parts; the trial's continuation: the Weyl test at nsub = 1 first (the previous accepted
configuration's gap between occupied and empty blocks against 2 ||h_m[Delta N, Delta v]||, the Gershgorin row sum): passing, the direct
endpoint continuation (version 12's fast path, else version 14's block step at the endpoint) is the adiabatic one; failing, version 15's
subdivisions nsub = 2, 4, ..., 64 with the test per sub-segment and, at the finest, the sampled continuation kept and flagged (an
unresolved split still rejects); per channel the count of (occupied, empty) level pairs in different blocks whose Weyl bound fails at the
finest subdivision (the largest count over its sub-segments) and the smallest margin gap - 2 ||Delta h_sub|| (negative where a crossing
cannot be excluded); "halvings" counts log2 nsub (0 for the endpoint, 1 for the midpoint); ||[h_m, R2]||_F per channel on the accepted
configuration (sqrt(2) ||(1 - R2) h_m V||_F with V R2's columns: one sparse product); version 12's midpoint verification (the endpoint
continuations repeated through the subdivisions and compared bitwise; a difference takes the subdivided reference and continues the
iteration through the subdivisions) on every --verify-every-th (10) re-solve;
(d) --site-fields averaged | dense: averaged, the fields that enter h_m are engine_fields_v12.site_fields_kernel_average's
N_n = int 4 pi r^2 K_0(r, R_n) N(r) dr and v_n = int 4 pi r^2 K_1(r, R_n) v(r) dr of the solved dense pair (the two kernel identities, K_0 on
constants and K_1 on v = c r, checked at the start and printed); dense, version 15's point sampling (the control);
streams: --streams 1 runs the channels' eigendecompositions from threads on separate CUDA streams (adopted only where
engine_eigh_streams_bench_v1 shows a gain above 1.3);
the solve's indicator (18.1): delta_rtol, the dense pair's change under rtol 1e-10 against 1e-12 (one extra solve on every
--delta-rtol-every-th record, the others carrying the latest value labelled with its record), plus the source splines' error:
B_estimate = delta_rtol + source-spline error, "indicator pass | fail" (B_estimate < 1e-9), sticky from the first failure; the collar on
every record: the padded cost (engine_fields_v12.padded_collar_cost; "lifted (padded cost)" when |delta E| < 1e-9 E and |ln N_b(a)| <
1e-9, else "flagged"; "clean" when the collar lies inside the chain) with version 15's tail-sum screen beside it; M1's Simpson check (18.1)
on 3 records (--m1-simpson-steps): the cut current at the state propagated by tau/2 under the same frozen h_m and reference, the next
record's balance printing the Simpson value of the evolution term beside the trapezoid. The record line reads "engine_fields_v12 (lead:
v10); tau = ...; indicator pass | fail; collar: clean | lifted (padded cost) | flagged". Version 15's docstring follows.
engine_collapse_v15.py (builder analyst; version 15 = version 14 plus (1) the goal of 19:57 UTC (the lead's review, binding): every record
carries the interior fields' change per self-consistency iteration (iteration_history: per iteration |dE/E|, max |N_out - N_in| and
max |v_out - v_in| over the sites, the solve's output against the iteration's applied fields; the stop rule tests |dE/E| < 1e-9 only),
the last iteration's two printed in the record line; (2) BUILDER, labelled, until the lead rules: when the Weyl test still fails at the
finest subdivision (64) but the sampled continuation there is resolved, the re-solve is kept with the continuation flagged
(intersample unverified, sticky from the first failure, printed in the record line) instead of rejected; an unresolved sampled split still
rejects. Nothing else changes. Version 14's docstring follows.
engine_collapse_v14.py (builder analyst; version 14 = version 13 with R25a's defects 3 and 4 in the continuation (the builder's proposal
in the lead's turn 26; labelled until ruled): (3) intersample control: per segment of the continuation path (previous configuration to
the trial, sub-steps k/nsub, nsub = 2 first, the fields interpolated linearly) the Gershgorin bound ||h_m(cur) - h_m(prev)||_2 <= max row
sum of |Delta h| (h_m is linear in N, v, so Delta h = h_m[Delta N, Delta v] on the host CSR) divided by nsub must satisfy
2 ||Delta h_sub|| < the occupied-to-empty gap between blocks at the sub-segment's start (Weyl: no occupied level can meet an empty one
inside the sub-segment), else nsub doubles (to 64) and then the re-solve is rejected; the gap at the start of the first sub-segment is the
previous accepted configuration's; (4) the rank of a block is the number of its weight-matrix eigenvalues within tol of 1, its trace
checked against that count (round(trace) could select rank 9 of ten weights 0.905); a partly occupied degenerate block declares gap 0
(the record's gap) while the Weyl test uses the gap between distinct blocks with occupied and empty members (a degenerate block's internal
rotation is fixed by the overlap, not by energy). Per record and channel: the norm, nsub and the smallest gap/(2 ||Delta h_sub||) ratio.
Nothing else changes. Version 13's docstring follows.
engine_collapse_v13.py (builder analyst; version 13 = version 12 with M1's operators sparse: the collar sits at the chain's last node in
every record (collar policy flag; r_collar = nsites), so the cut is at nsites - 10 and version 12's band restriction of the inside energies
saved nothing: five dense n x n products per channel and re-solve (about 4 s at r_s 32, eight times that at r_s 64). Here h_m enters M1 and
M9 as the host CSR 0.5{N, h} + 0.5{v, Q} (the same operator) moved to the device: <H_in> from the rows of P_in by one sparse product, the
cut current's (h_m^2)_W from the host sparse product of the window's rows and columns. M1 and M9 are diagnostics (nothing in them feeds the
evolution), so the trajectory is version 12's bitwise and the M1 parts agree to rounding. Version 12's docstring follows.
engine_collapse_v12.py (builder analyst; version 12 = version 11 made faster and leaner for the owner's request of 15 October (the GPU
jobs optimized for speed and memory); the accepted state of every re-solve is computed exactly as version 11 computes it (the same
continuation through the midpoint, the same solve, the same certificate formulas); what changes is the work spent on the trial iterations
and on diagnostics (BUILDER, labelled; each part switchable): (1) --reuse-first 1: the first iteration of a re-solve starts from the
previous re-solve's applied fields, whose h_m is the reference configuration itself, so the continuation is the identity and no
eigendecomposition is done (version 11 diagonalized an h_m equal to it to 1e-10); the first iteration is never accepted (its dE is
undefined), so its sources only start the iteration; (2) --verify-at-end 1: the trial iterations continue the reference from the
previous configuration to the trial endpoint directly (when the negative-energy eigenvectors number n_sea, no degenerate block straddles
zero and their summed weight exceeds n_sea - 0.1, the block algorithm provably selects exactly these columns, so they are taken from one
half-size overlap; otherwise the block algorithm, and through the midpoint when the endpoint is unresolved); when a trial converges
(dE < 1e-9) its configuration is continued through the midpoint exactly as in version 11 (engine 17.3 (3)) and the reference compared
bitwise with the trial's; equal: accepted with the midpoint continuation's record entries; different: every later iteration of the
re-solve continues through the midpoint (the fallback, counted); (3) h_m is assembled on the GPU from the channel's sparse h and Q
(clocked_gpu: no dense host array, no transfer) and rebuilt when needed, never stored (the midpoint from the linearly interpolated fields,
which is the interpolated matrix since h_m is linear in N and v); (4) the diagnostics of the reference (ambiguous blocks, weights, pockets,
refills, R2 - R0) once per re-solve on the accepted state; (5) M1's inside energies on the rows of P_in and the band of h_m only (the same
number as the full product); (6) eps_map (the extra solve at rtol 1e-10) on every --eps-map-every-th (10) re-solve and on every flagged one
(converged at the floor or at the iteration limit), as engine 17.3 (1) specifies; the other records use the latest measured value, labelled
with its record; (7) per record the timing of the parts (eigendecompositions and their count, continuation, densities, solves and their
inner iterations, certificate, M1). Version 11's docstring follows.
engine_collapse_v11.py (builder analyst; version 11 = version 10 on engine_fields_v9 with on_maxit = "return": a re-solve whose
iteration reaches the limit (200) with the larger of its last changes below 1e-8 no longer stops the run; the last iterate is the pair,
the record carries solver.not_converged = {dN, dW, it, stall_cap}, the line prints "NOT CONVERGED (dN, dW)", and engine 17.3 (1)'s
certificate decides the label as before (B with q_c over the last three changes; B_prefloor printed beside it, not used for the label);
a change at the limit above 1e-8 still rejects and stops. Cause: the v9 primary (r_s 64, ell 4, tau 8, R2) stopped at T/rs 5.88 with
dN 1.77e-11, dW 1.31e-10 after 200 iterations (version 7's floor rule needs three changes below 1e-11). Nothing else changes.
Version 10's docstring follows.
engine_collapse_v10.py (builder analyst; engine 17.3 (3) to (5) and the certificates of (1) and (2) inline; version 10 = version 9 on
engine_fields_v8 plus: (a) per record the certificate of engine_fields_v8.certify (q_c and its pre-floor value, eta, eps_map at rtol 1e-10 on
every record, B or "uncertified", the U = W/N^2 and ln N routes' defects, the residual ladder and the matching region's residuals, the collar's
tail sums against 1e-9) and the label "engine_fields_v8; tau = ...; certified | uncertified", sticky from the first failing re-solve;
(b) R2 at avoided crossings: no early return; every continuation passes the midpoint configuration (the fields interpolated linearly), and
deeper halvings (to 64 sub-steps) when needed; a block is accepted only when its weight on the previous reference is within 0.1 of an integer
and every eigenvalue of its weight matrix lies within 0.1 of 0 or 1 (the resolved split); otherwise the re-solve is rejected and the run stops
with the record; the smallest occupied-to-empty gap per channel at the start, the midpoint and the end is recorded (a partly occupied
degenerate block counts as gap 0, flagged); (c) M1's balance per interval: with X = Tr[C A_k], Tr[R2 A_k] at the record and Y = Tr[C A_(k-1)],
Tr[R2_k A_(k-1)] at the old configuration, the exact split d Tr(D A) = evolution under the frozen h + field work + reference reset (their sum
closing to rounding), the evolution term against the interval's trapezoid of I_E; (d) per-jet errors: two Richardson estimates (h = 1/4, 1/8 and
1/8, 1/16) of kappa_T and of A, B, C of v and v/N, their differences recorded; v/N's jets in z = kappa' (R - R_t) with kappa' = (v/N)' at R_t
(observables 13.1; kappa_T/N_t printed against it; the kappa_T-normalized v/N jets kept beside, labelled); (e) the fields npz written every 5 records (certification can
run during the run), with N_applied and v_applied: the fields of the last self-consistency iteration's h_m, the Hamiltonian that continued the
reference and propagated the sea (the stored N, v are that iteration's solve; they differ at the convergence level), so C5.3' can replay exactly; (f) every 10 re-solves the projection of the end and engine 12.6's decision printed. Version 9's docstring follows.
engine_collapse_v9.py (builder analyst; version 9: engine_collapse_v8 plus the pair block's own source exported per step: the fields npz
carries eps_pairs and J_pairs (the shell totals of the continued sea against its reference, without the dust) beside eps_tot and J_tot. The
spec's no-dust control (sec. 5.3, C5.3: "the vacuum under the primary run's stored fields: the created-pair source alone") evolves the same
initial sea under the same stored N, v; by linearity it reproduces this block exactly (the engine red team's remark: bookkeeping, not an
independent evolution), so the block is stored by the run itself instead of a second run. Nothing else changes. Version 8's docstring follows.
engine_collapse_v8.py (builder analyst; version 8: engine_collapse_v7 on engine_fields_v7, whose iteration accepts a stall at the
integrator's floor (both changes below 1e-11 for three iterations; flagged): version 7's r_s 64 ell 4 primary was rejected at T/rs 0.25 with
dN 7.6e-13 and dW 1.46e-12 after 200 iterations. Each record now carries the solver's converged_at_floor entry (None when converged below tol).
Nothing else changes. Version 7's docstring follows.
engine_collapse_v7.py (builder analyst; version 7: engine_collapse_v6 on engine_fields_v6, whose trapping-radius scan reaches the chain's end
(version 5 of the fields stopped at the collar). Nothing else changes. Version 6's docstring follows.
engine_collapse_v6.py (builder analyst; SPEC_final4b_engine.md sec. 16.2 items (1) to (9) and the restart at tau = 8; version 6 of
engine_collapse_v5.py, whose docstring and helpers (the channel chains, the dust's preparation, the energy cut current) it imports).

Version 6: (a) the fields from engine_fields_v5 (the lead's "engine_fields_v4" of 16.2: dense-function convergence and rejection, parity-extended
sources, the support of both sources for the collar, refined trapping zeros, the exterior W as the field, the amended dictionary line); the
collar policy --collar-policy flag (BUILDER, labelled: the continued sea polarizes the whole chain, so the sources' support reaches the chain's
end and "fail" would stop every re-solve; the margin's deficit and the sources' share in the last 8 ell are recorded per re-solve) or fail;
(b) B1 to the fourth derivative of N + v at R_t by Richardson in h (1/4 and 1/8, central differences of the dense functions), with P25's jets
A, B, C of v and of v/N (z = kappa (R - R_t), kappa = d(N + v)/dR) and the exact profile's at the solved E; (c) M1 corrected: I_E =
Tr[(C - R2) i[h_m, H_in]] is +d<H_in>/dt (the rate of the inside Killing energy at the collar cut), printed beside dE/dT and d<H_in>/dT (from
successive re-solves) and the balance d(E - <H_in>)/dT; M9 stores the matched cut currents at 0.5 R_sh, R_t and the collar beside the rates of
W/(2g) there; (d) R2's continuation degeneracy aware: the current eigenvalues within 1e-8 of the band scale form blocks; a block's summed weight
on the previous reference fixes its occupied dimension (an integer within 0.1, else ambiguous), a partly occupied block takes the top
eigenvectors of its weight matrix; the total must equal n_sea; an ambiguous step is retried through intermediate configurations
h(s) = h_prev + s (h_cur - h_prev) with up to 6 halvings (64 sub-steps), and an ambiguity that survives is a rejected re-solve (the run stops
with the record written). R2 is this lane's declared reference: the adiabatically continued, never refilled sea (the literal propagated sea R1
sources nothing); (e) the export: the fields npz stores per slice the shell totals eps_tot and J_tot (dust plus pairs), with g, ell and the
solver's parameters, so a consumer re-solves the dense functions (engine_fields_v5.resolve_dense); the unit-site arrays stay; (f) the record's
labels: the collar charge W/(2g) and the Misner-Sharp W/(2 g N^2) at R_t, late_surface by the total source's vacuum condition (|rho_ell| and
|J_ell| over [R_t - 5 ell, R_t + 5 ell] below 1e-6 of their maxima; the ratios printed), v at R_c from the dense flow; the residuals, the source
splines' error, the centre series and the tail diagnostic per re-solve; "engine_fields_v6; tau = %g" in the label.
Usage: python engine_collapse_v6.py --m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --ell 4 --tau 8 --T-over-rs 40
--kappa-sea 10 --reference R2 --out <dir> --tag <tag> [--gpu 0] [--collar-policy flag] [--benchmark]"""
import argparse
import json
import math
import os
import threading
import time

import numpy as np
import scipy.sparse as sp

import engine_collapse_v5 as C5
import engine_fields_v16 as FLD
import engine_mixing_v1 as MIX
import engine_ops_v2 as OPS2
import engine_ops_v3 as OPS3
import engine_stress_v2 as STR

BK = C5.BK
DICT24 = FLD.DICTIONARY.replace("the owner's confirmation pending", "the lane's declaration (engine 12.2)")
T_ = C5.T_
eigT = C5.eigT
dens = C5.dens
Chan = C5.Chan
logistic = C5.logistic
fill_lowest = C5.fill_lowest
threshold_radii = C5.threshold_radii
energy_cut_current = C5.energy_cut_current
OPT = {"collar_policy": "flag", "support_rel": 1e-14, "site_fields": "dense"}


SOLVE_TOL = {"rtol": 1e-12, "atol": 1e-16}   # version 29: set from --field-rtol, --field-atol


def solve(eps, J, g, R, ell, N0, rtol=None, atol=None):
    rtol = SOLVE_TOL["rtol"] if rtol is None else rtol; atol = SOLVE_TOL["atol"] if atol is None else atol
    """the two constraints as one outward initial-value problem (engine_fields_v16.solve_constraints_ivp); the site fields point sampled
    (dense) or kernel averaged (averaged; info carries the point values as N_point, v_point)."""
    Nn, vn, Wn, info = FLD.solve_constraints_ivp(eps, J, g, R, ell, N0=N0, rtol=rtol, atol=atol, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"])
    if info.get("failed") or OPT["site_fields"] == "dense":
        return Nn, vn, Wn, info
    Na, va, norms = FLD.site_fields_kernel_average(info, R, ell)
    info["site_fields"] = "averaged"
    info["N_point"], info["v_point"] = Nn, vn
    info["kernel_norms"] = (float(norms.min()), float(norms.max()))
    return Na, va, Wn, info


def kernel_identities(R, ell, c=0.37):
    """engine 18.5 (d): site_fields_kernel_average on N = 1 and v = c r (K_0 preserves constants, K_1 preserves v = c r)."""
    one = lambda r: np.ones_like(np.atleast_1d(np.asarray(r, float)))
    lin = lambda r: c * np.atleast_1d(np.asarray(r, float))
    Na, va, norms = FLD.site_fields_kernel_average({"N_dense": one, "v_dense": lin}, R, ell)
    R = np.asarray(R, float)
    dv = float(np.max(np.abs(va - c * R)))
    return {"K0_constant_max_abs": float(np.max(np.abs(Na - 1.0))), "K1_linear_max_abs": dv, "K1_linear_max_rel": dv / (c * float(R.max())),
            "K0_weight_sums": [float(norms.min()), float(norms.max())]}


def tail_screen(eps_tot, J_tot, g, R, ell, N_sites, v_sites, info):
    """version 15's collar screen (engine_fields_v9.certify's tail sums over the chain's last 8 ell): a conservative screen, not the cost."""
    R = np.asarray(R, float); eps_tot = np.asarray(eps_tot, float); J_tot = np.asarray(J_tot, float)
    E = float(info["E_collar"])
    reg = R > R[-1] - 8.0 * ell
    t1 = np.asarray(N_sites, float)[reg] ** 2 * eps_tot[reg]
    vv = np.asarray(v_sites, float)[reg]
    ok = np.abs(vv) > 0
    t2 = np.where(ok, g * J_tot[reg] / (R[reg] * np.where(ok, vv, 1.0)), 0.0)
    return {"flagged": bool(info.get("collar_margin", {}).get("deficit_ell", 0) > 0), "delta_E_signed": float(t1.sum()), "delta_E_rel_abs": float(np.abs(t1).sum()) / max(abs(E), 1e-300),
            "delta_invN_signed": float(t2.sum()), "delta_invN_abs": float(np.abs(t2).sum())}


def indicator(eps, J, g, R, ell, info, N_sites, v_sites, fresh, last):
    """engine 18.1 for the initial-value solve: delta_rtol (fresh: the dense pair's change under rtol 1e-10, atol 1e-14 against the run's
    1e-12, 1e-16; else the latest value, labelled), the source splines' error, B_estimate = their sum, the pass (B_estimate < 1e-9); the
    collar's padded cost and label, the tail-sum screen beside it."""
    out = {}
    sse = info.get("source_spline_error")
    s_val = max(float(x) for x in sse.values()) if isinstance(sse, dict) else float(sse or 0.0)
    out["source_spline_error"] = s_val
    if fresh:
        _, _, _, ia = FLD.solve_constraints_ivp(eps, J, g, R, ell, rtol=1e-10, atol=1e-14, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"])
        if ia.get("failed"):
            out["delta_rtol"] = None
            out["delta_rtol_failed"] = ia["failed"]
        else:
            rf = info["rfine"]
            Wret = info["W_field"](rf)
            Wmax = max(float(np.max(np.abs(Wret))), 1e-300)
            out["delta_rtol"] = max(float(np.max(np.abs(ia["N_dense"](rf) - info["N_dense"](rf)))), float(np.max(np.abs(ia["W_field"](rf) - Wret))) / Wmax)
            out["delta_rtol_E"] = abs(float(ia["E_collar"]) - float(info["E_collar"])) / abs(float(info["E_collar"]))
        out["delta_rtol_source"] = "this record"
    else:
        out["delta_rtol"] = last["value"]
        out["delta_rtol_source"] = "the record at T/rs %.2f (every --delta-rtol-every-th record)" % last["T_over_rs"]
    out["B_estimate"] = (out["delta_rtol"] + s_val) if out["delta_rtol"] is not None else None
    out["indicator_pass"] = bool(out["B_estimate"] is not None and out["B_estimate"] < 1e-9)
    out["collar_screen"] = tail_screen(eps, J, g, R, ell, N_sites, v_sites, info)
    if out["collar_screen"]["flagged"]:
        pc = FLD.padded_collar_cost(eps, J, g, R, ell, info)
        out["collar_padded"] = pc
        out["collar_label"] = "lifted (padded cost)" if pc["passes"] else "flagged"
    else:
        out["collar_padded"] = None
        out["collar_label"] = "clean"
    return out


def prepare_dust(chs, g, R, ell, NB, R0, m, reading, shell, passes=200, tol=1e-10):
    """version 5's preparation (the levitating well, the zero-temperature filling, the projection); version 17: each solve the initial-value problem, the site fields by OPT."""
    torch = BK["torch"]
    nsites = len(R)
    N = np.ones(nsites); v = np.zeros(nsites)
    mixer = MIX.Anderson()
    U0 = 0.5 * m
    for ip in range(passes):
        Ns = np.sqrt(np.maximum(N * N - v * v, 1e-12)) if reading == "A" else N.copy()
        if shell:
            s_in, s_out = logistic(R, shell[0]), logistic(R, shell[1])
            Uw = U0 * (1 - s_in) + U0 * s_out + m * (1 - Ns) * s_in * (1 - s_out)
        else:
            s = logistic(R, R0)
            Uw = U0 * s + m * (1 - Ns) * (1 - s)
        Ud = T_(np.diag(np.repeat(Uw, 2)))
        spec, levels = {}, []
        for ic, ch in enumerate(chs):
            hw = ch.clocked_dense(N, v if reading == "A" else 0 * v) + Ud
            ew, Uwv = eigT(hw)
            em, Um = eigT(ch.clocked_dense(N, v))
            pos = ew > 0
            ewp = ew[pos].cpu().numpy()
            spec[ch.kappa] = (ewp, Uwv[:, pos], Um[:, em > 0], hw)
            levels += [(float(x), ch.w, ic, j) for j, x in enumerate(ewp[:400])]
        occ = fill_lowest(levels, NB)
        eps = np.zeros(nsites); J = np.zeros(nsites); dust = {}; NBs = 0.0
        diag = {"discarded_norm_max": 0.0, "gram_cond_max": 1.0, "stationarity_max": 0.0}
        for ic, ch in enumerate(chs):
            ewp, Uwv, Pplus, hw = spec[ch.kappa]
            keep = [j for j in range(min(400, len(ewp))) if occ.get((ic, j), 0.0) > 0]
            if not keep:
                continue
            fk = np.array([occ[(ic, j)] for j in keep])
            raw = Pplus @ (Pplus.conj().T @ Uwv[:, keep])
            diag["discarded_norm_max"] = max(diag["discarded_norm_max"], float((1 - (raw.abs() ** 2).sum(dim=0)).max()))
            S = raw.conj().T @ raw
            sv, sU = torch.linalg.eigh(S)
            diag["gram_cond_max"] = max(diag["gram_cond_max"], float(sv.max() / sv.min().clamp_min(1e-300)))
            Phi = raw @ (sU * sv.clamp_min(1e-300).rsqrt()[None, :].to(sU.dtype)) @ sU.conj().T
            Ew = (Phi.conj() * (hw @ Phi)).sum(dim=0).real
            res = (hw @ Phi - Phi * Ew[None, :].to(Phi.dtype)).norm(dim=0)
            diag["stationarity_max"] = max(diag["stationarity_max"], float(res.max()))
            eps += ch.w * dens(ch.h, Phi, fk); J += ch.w * dens(ch.Q, Phi, fk); NBs += ch.w * fk.sum()
            dust[ch.kappa] = (Phi, fk)
        Nn, vn, Wn, info = solve(eps, J, g, R, ell, N)
        if info.get("failed"):
            raise RuntimeError("initial data: the solver failed: %s" % info["failed"])
        d = max(float(np.max(np.abs(Nn - N))), float(np.max(np.abs(vn - v))))
        x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
        N, v = x[:nsites], x[nsites:]
        if d < tol:
            break
    N, v, Wn, info = solve(eps, J, g, R, ell, N)
    if info.get("failed"):
        raise RuntimeError("initial data: the final solve failed: %s" % info["failed"])
    kin = 0.0
    for ch in chs:
        if ch.kappa in dust:
            Phi, fk = dust[ch.kappa]
            hD = (Phi.conj() * (ch.h @ Phi)).sum(dim=0).real.cpu().numpy()
            kin += ch.w * float(fk @ (hD - m))
    u = np.where(np.abs(eps) > 1e-14 * np.abs(eps).max(), J / np.where(np.abs(eps) > 0, eps, 1.0), 0.0)
    bulk = 0.5 * float(np.sum(eps * u * u))
    head = {"N_B": NBs, "passes": ip + 1, "last_change": d, "E0": FLD.charge(info, g), "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
            "site_fields": "%s (engine_fields_v16, solve_constraints_ivp)" % OPT["site_fields"], "f_kin": kin / (NBs * m), "f_kin_bulk_frame": (kin - bulk) / (NBs * m), "dust_momentum_total": float(J.sum()),
            "projection": diag, "reading": reading, "shell": shell, "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"),
            "source_spline_error": info.get("source_spline_error"), "centre_series_ratio": info.get("centre_series_ratio"), "collar_margin": info.get("collar_margin"),
            "solver_iterations": info.get("it"), "dictionary": FLD.DICTIONARY}
    return N, v, dust, info, head


def surface_jets(info, Rt, g, E):
    """B1 to the fourth derivative of f = N + v at R_t by Richardson in h (1/4, 1/8); P25's jets A, B, C of v and of v/N with z = kappa (R - R_t)."""
    if Rt is None:
        return None
    Nf = lambda x: float(info["N_dense"](x)[0]); vf = lambda x: float(info["v_dense"](x)[0])

    def d14(F, x, h):
        f = {q: F(x + q * h) for q in (-2, -1, 0, 1, 2)}
        d1 = (f[1] - f[-1]) / (2 * h)
        d2 = (f[1] - 2 * f[0] + f[-1]) / h ** 2
        d3 = (f[2] - 2 * f[1] + 2 * f[-1] - f[-2]) / (2 * h ** 3)
        d4 = (f[2] - 4 * f[1] + 6 * f[0] - 4 * f[-1] + f[-2]) / h ** 4
        return np.array([d1, d2, d3, d4])

    def rich(F):
        a, b = d14(F, Rt, 0.25), d14(F, Rt, 0.125)
        return (4.0 * b - a) / 3.0, float(np.max(np.abs(b - a)))

    def rich2(F):
        b, c = d14(F, Rt, 0.125), d14(F, Rt, 0.0625)
        return (4.0 * c - b) / 3.0
    fnv = lambda x: Nf(x) + vf(x)
    qv = lambda x: vf(x) / Nf(x)
    dfn, spread = rich(fnv)
    dv, _ = rich(vf)
    dq, _ = rich(qv)
    kap = float(dfn[0])
    dfn2 = rich2(fnv); dv2 = rich2(vf); dq2 = rich2(qv)
    kap2 = float(dfn2[0])
    jets = lambda d, k: [float(d[1] / (2 * k ** 2)), float(d[2] / (6 * k ** 3)), float(d[3] / (24 * k ** 4))]
    ex = [1.0 / (4 * g * E), -3.0 / (16 * (g * E) ** 2), 15.0 / (64 * (g * E) ** 3), -105.0 / (256 * (g * E) ** 4)]
    # v's jets in z = kappa_T (R - R_t); v/N's in z = kappa' (R - R_t), kappa' = (v/N)' = kappa_T/N_t (observables 13.1's lapse mapping; the
    # manufactured variable-lapse fixture showed the kappa_T normalization of v/N's jets off by powers of N_t)
    kq, kq2 = float(dq[0]), float(dq2[0])
    jv1, jv2 = jets(dv, kap), jets(dv2, kap2); jq1, jq2 = jets(dq, kq), jets(dq2, kq2)
    Nt = Nf(Rt)
    return {"d_N_plus_v": [float(x) for x in dfn], "over_exact": [float(dfn[i] / ex[i]) for i in range(4)], "richardson_spread": spread,
            "kappa_T": kap, "kappa_prime": kq, "N_t": Nt, "kappa_prime_minus_kappa_T_over_N_t": kq - kap / Nt,
            "jets_v": jv1, "jets_v_over_N": jq1, "jets_v_over_N_in_kappa_T": jets(dq, kap), "jets_exact": [-1.5, 2.5, -4.375],
            "kappa_T_second_estimate": kap2, "kappa_T_err": abs(kap - kap2), "jets_v_err": [abs(x - y) for x, y in zip(jv1, jv2)],
            "jets_v_over_N_err": [abs(x - y) for x, y in zip(jq1, jq2)]}


def inside_energy(hmT, cols, f, cut):
    """<H_in> = Tr[D H_in], H_in = (1/2){P_in, h_m}: Re sum_j f_j (P_in c_j)^dagger (h_m c_j)."""
    hc = hmT @ cols
    P = slice(0, 2 * cut)
    w = (cols[P, :].conj() * hc[P, :]).sum(dim=0).real.cpu().numpy()
    return float(np.dot(np.asarray(f, float), w))


def blocks_of(e, tol_rel=1e-8):
    e = np.asarray(e, float)
    scale = max(float(np.max(np.abs(e))), 1e-300)
    out, i = [], 0
    while i < len(e):
        j = i + 1
        while j < len(e) and abs(e[j] - e[j - 1]) < tol_rel * scale:
            j += 1
        out.append((i, j))
        i = j
    return out


def continue_reference_aware(ref_prev, e, U, n_sea):
    """one continuation step: (ref_new columns, info) or (None, info) when ambiguous."""
    torch = BK["torch"]
    A = ref_prev.conj().T @ U                                       # n_prev x n
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy()
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    cols, n_occ, amb, npocket, wmin, wmax_empty = [], 0, 0, 0, 1.0, 0.0
    for (i, j) in blocks_of(e_np):
        WB = float(w[i:j].sum()); nB = int(round(WB))
        if abs(WB - nB) >= 0.1:
            amb += 1
            continue
        size = j - i
        if nB == size:
            cols.append(U[:, i:j]); wmin = min(wmin, WB / size)
        elif nB > 0:
            M = A[:, i:j].conj().T @ A[:, i:j]
            ev, V = torch.linalg.eigh(M)
            top = V[:, -nB:]
            cols.append(U[:, i:j] @ top); wmin = min(wmin, float(ev[-nB:].real.min()))
            wmax_empty = max(wmax_empty, float(ev[:-nB].real.max()) if size > nB else 0.0)
        else:
            wmax_empty = max(wmax_empty, WB / size)
        n_occ += nB
        eB = float(np.mean(e_np[i:j]))
        npocket += nB if eB > 0 else (size - nB if eB < 0 else 0)
    info = {"ambiguous_blocks": amb, "n_occ": n_occ, "n_sea": n_sea, "n_pocket": npocket, "min_occupied_weight": wmin, "max_empty_weight": wmax_empty,
            "nneg": int((e_np < 0).sum())}
    if amb or n_occ != n_sea or not cols:
        return None, info
    return torch.cat(cols, dim=1), info


def continue_reference_resolved(ref_prev, e, U, n_sea, tol_split=0.1):
    """version 10's step: every block's weight-matrix eigenvalues within tol_split of 0 or 1 and its trace within tol_split of an integer (the
    resolved split); the occupied and empty eigenvalues collected for the gap (a partly occupied degenerate block gives gap 0)."""
    torch = BK["torch"]
    A = ref_prev.conj().T @ U
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy()
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    cols, n_occ, amb, npocket, wmin, wmax_empty, partial = [], 0, 0, 0, 1.0, 0.0, 0
    occ_e, emp_e = [], []
    for (i, j) in blocks_of(e_np):
        size = j - i
        if size == 1:
            lam = np.array([w[i]]); V = None
        else:
            M = A[:, i:j].conj().T @ A[:, i:j]
            lam_t, V = torch.linalg.eigh(M)
            lam = lam_t.real.cpu().numpy()
        WB = float(lam.sum()); nB = int(round(WB))
        if abs(WB - nB) >= tol_split or not bool(np.all((np.abs(lam) < tol_split) | (np.abs(lam - 1.0) < tol_split))):
            amb += 1
            continue
        if nB == size:
            cols.append(U[:, i:j]); wmin = min(wmin, float(lam.min())); occ_e += list(e_np[i:j])
        elif nB > 0:
            cols.append(U[:, i:j] @ V[:, -nB:]); wmin = min(wmin, float(lam[-nB:].min()))
            wmax_empty = max(wmax_empty, float(lam[:-nB].max())); partial += 1
            occ_e += list(e_np[i:j][:nB]); emp_e += list(e_np[i:j][nB:])
        else:
            wmax_empty = max(wmax_empty, float(lam.max())); emp_e += list(e_np[i:j])
        n_occ += nB
        eB = float(np.mean(e_np[i:j]))
        npocket += nB if eB > 0 else (size - nB if eB < 0 else 0)
    gap = None
    if occ_e and emp_e:
        lab = np.concatenate([np.zeros(len(occ_e)), np.ones(len(emp_e))]); vals = np.concatenate([occ_e, emp_e])
        o = np.argsort(vals); vs, ls = vals[o], lab[o]
        dif = np.abs(np.diff(vs))[ls[1:] != ls[:-1]]
        gap = float(dif.min()) if len(dif) else None
    info = {"ambiguous_blocks": amb, "n_occ": n_occ, "n_sea": n_sea, "n_pocket": npocket, "min_occupied_weight": wmin, "max_empty_weight": wmax_empty,
            "nneg": int((e_np < 0).sum()), "gap": gap, "partly_occupied_blocks": partial}
    if amb or n_occ != n_sea or not cols:
        return None, info
    return torch.cat(cols, dim=1), info


def continue_resolved(ref_prev, hm_prev, hm_cur, e_cur, U_cur, n_sea, max_halvings=6):
    """version 10: always through the midpoint (2 sub-steps), then 4, 8, ... 64 when a sub-step is unresolved; the gaps at the midpoint and the end."""
    last = None
    for hv in range(1, max_halvings + 1):
        nsub = 2 ** hv
        r_ = ref_prev; ok = True; gap_mid = None
        for k in range(1, nsub + 1):
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                e_s, U_s = eigT(hm_prev + (k / nsub) * (hm_cur - hm_prev))
            r_, inf_ = continue_reference_resolved(r_, e_s, U_s, n_sea)
            last = inf_
            if r_ is None:
                ok = False
                break
            if k == nsub // 2:
                gap_mid = inf_["gap"]
        if ok:
            inf_["halvings"] = hv - 1
            inf_["gap_mid"] = gap_mid; inf_["gap_end"] = inf_["gap"]
            return r_, inf_
    info = dict(last or {}); info["halvings"] = max_halvings; info["rejected"] = True
    return None, info


def continue_with_halving(ref_prev, hm_prev, hm_cur, e_cur, U_cur, n_sea, max_halvings=6):
    """the step from the previous reference to the current eigenbasis; on ambiguity, through intermediate configurations, up to 6 halvings."""
    ref, info = continue_reference_aware(ref_prev, e_cur, U_cur, n_sea)
    info["halvings"] = 0
    if ref is not None or hm_prev is None:
        return ref, info
    for hv in range(1, max_halvings + 1):
        nsub = 2 ** hv
        r_ = ref_prev
        ok = True
        for k in range(1, nsub + 1):
            s = k / nsub
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                e_s, U_s = eigT(hm_prev + s * (hm_cur - hm_prev))
            r_, inf_ = continue_reference_aware(r_, e_s, U_s, n_sea)
            if r_ is None:
                ok = False
                break
        if ok:
            inf_["halvings"] = hv
            return r_, inf_
    info["halvings"] = max_halvings
    info["rejected"] = True
    return None, info


def sync_dev():
    if BK["dev"] is not None and BK["dev"].type == "cuda":
        BK["torch"].cuda.synchronize(BK["dev"])


TM = {}


def tm_reset():
    TM.clear()
    TM.update({"eigh_s": 0.0, "eigh_n": 0, "continuation_s": 0.0, "weyl_s": 0.0, "weyl_pass_unresolved": 0, "verify_s": 0.0, "densities_s": 0.0, "solve_s": 0.0,
               "solve_n": 0, "solve_nfev": 0, "indicator_s": 0.0, "commutator_s": 0.0, "M1_s": 0.0, "simpson_s": 0.0, "propagation_s": 0.0, "lagged": 0, "fast": 0,
               "block": 0, "midpoint": 0, "fallbacks": 0})


def eigT_t(hmT):
    t_ = time.time()
    e, U = eigT(hmT)
    sync_dev()
    TM["eigh_s"] = TM.get("eigh_s", 0.0) + time.time() - t_; TM["eigh_n"] = TM.get("eigh_n", 0) + 1
    return e, U


def clocked_gpu(ch, N, v):
    """h_m = (1/2){N, h} + (1/2){v, Q} (Chan.clocked_dense's operator) assembled on the device from the channel's sparse h and Q."""
    torch = BK["torch"]
    dev = BK["dev"]
    if getattr(ch, "_coo", None) is None:
        hc = ch.hs.tocoo(); qc = ch.Qs.tocoo()
        ch._coo = tuple(torch.as_tensor(x, dtype=dt, device=dev) for x, dt in ((hc.row, torch.int64), (hc.col, torch.int64), (hc.data, torch.complex128),
                                                                            (qc.row, torch.int64), (qc.col, torch.int64), (qc.data, torch.complex128)))
    hr, hcl, hv, qr, qcl, qv = ch._coo
    n = ch.hs.shape[0]
    Nd = torch.as_tensor(np.repeat(np.asarray(N, float), 2), dtype=torch.float64, device=dev)
    Vd = torch.as_tensor(np.repeat(np.asarray(v, float), 2), dtype=torch.float64, device=dev)
    M = torch.zeros((n, n), dtype=torch.complex128, device=dev)
    M.index_put_((hr, hcl), hv * (0.5 * (Nd[hr] + Nd[hcl])).to(torch.complex128), accumulate=True)
    M.index_put_((qr, qcl), qv * (0.5 * (Vd[qr] + Vd[qcl])).to(torch.complex128), accumulate=True)
    return M


def continue_fast(ref_prev, e, U, n_sea, tol_split=0.1):
    """the trial iterations' endpoint continuation: when the negative-energy eigenvectors number n_sea, no degenerate block (blocks_of's
    tolerance) straddles zero and their summed weight in the previous reference exceeds n_sea - tol_split, each occupied weight exceeds
    1 - tol_split and the empty ones sum below tol_split (the weights lie in [0, 1] and sum to n_sea; for a degenerate block the same bound
    holds for the eigenvalues of its weight matrix), so continue_reference_resolved selects exactly these columns, in this order: the same
    reference from half the overlap. Otherwise (None, reason)."""
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    nneg = int((e_np < 0).sum())
    if nneg != n_sea or nneg == 0 or nneg == len(e_np):
        return None, "nneg %d against n_sea %d" % (nneg, n_sea)
    scale = max(float(np.max(np.abs(e_np))), 1e-300)
    if abs(e_np[nneg] - e_np[nneg - 1]) < 1e-8 * scale:
        return None, "a degenerate block straddles zero"
    Uneg = U[:, :nneg]
    wsum = float((ref_prev.conj().T @ Uneg).abs().pow(2).sum())
    if wsum <= n_sea - tol_split:
        return None, "summed weight %.4f" % wsum
    return Uneg, None


def continue_resolved_f(ch, ref_prev, f_prev, f_cur, e_cur, U_cur, n_sea, max_halvings=6):
    """continue_resolved (version 10) with the sub-step configurations built from the linearly interpolated fields (h_m is linear in N, v)."""
    last = None
    for hv in range(1, max_halvings + 1):
        nsub = 2 ** hv
        r_ = ref_prev; ok = True; gap_mid = None
        for k in range(1, nsub + 1):
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                s_ = k / nsub
                e_s, U_s = eigT_t(clocked_gpu(ch, f_prev[0] + s_ * (f_cur[0] - f_prev[0]), f_prev[1] + s_ * (f_cur[1] - f_prev[1])))
            r_, inf_ = continue_reference_resolved(r_, e_s, U_s, n_sea)
            last = inf_
            if r_ is None:
                ok = False
                break
            if k == nsub // 2:
                gap_mid = inf_["gap"]
        if ok:
            inf_["halvings"] = hv - 1
            inf_["gap_mid"] = gap_mid; inf_["gap_end"] = inf_["gap"]
            return r_, inf_
    info = dict(last or {}); info["halvings"] = max_halvings; info["rejected"] = True
    return None, info


def continue_reference_resolved_v14(ref_prev, e, U, n_sea, tol_split=0.1):
    """version 14's step (R25a defect 4): a block's rank is the number of its weight-matrix eigenvalues within tol_split of 1, the trace
    checked against that count; every eigenvalue within tol_split of 0 or 1; a partly occupied block declares gap 0; gap_between_blocks
    is the smallest distance between consecutive blocks one of which has occupied and the other empty members."""
    torch = BK["torch"]
    A = ref_prev.conj().T @ U
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy()
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    cols, n_occ, amb, npocket, wmin, wmax_empty, partial = [], 0, 0, 0, 1.0, 0.0, 0
    occ_e, emp_e, blk = [], [], []
    for (i, j) in blocks_of(e_np):
        size = j - i
        if size == 1:
            lam = np.array([w[i]]); V = None
        else:
            M = A[:, i:j].conj().T @ A[:, i:j]
            lam_t, V = torch.linalg.eigh(M)
            lam = lam_t.real.cpu().numpy()
        near1 = np.abs(lam - 1.0) < tol_split; near0 = np.abs(lam) < tol_split
        WB = float(lam.sum()); nB = int(near1.sum())
        if not bool(np.all(near0 | near1)) or abs(WB - nB) >= tol_split:
            amb += 1
            continue
        if nB == size:
            cols.append(U[:, i:j]); wmin = min(wmin, float(lam.min())); occ_e += list(e_np[i:j])
        elif nB > 0:
            cols.append(U[:, i:j] @ V[:, -nB:]); wmin = min(wmin, float(lam[-nB:].min()))
            wmax_empty = max(wmax_empty, float(lam[:-nB].max())); partial += 1
            occ_e += list(e_np[i:j][:nB]); emp_e += list(e_np[i:j][nB:])
        else:
            wmax_empty = max(wmax_empty, float(lam.max())); emp_e += list(e_np[i:j])
        blk.append((float(e_np[i]), float(e_np[j - 1]), nB > 0, nB < size))
        n_occ += nB
        eB = float(np.mean(e_np[i:j]))
        npocket += nB if eB > 0 else (size - nB if eB < 0 else 0)
    gap = None
    if occ_e and emp_e:
        lab = np.concatenate([np.zeros(len(occ_e)), np.ones(len(emp_e))]); vals = np.concatenate([occ_e, emp_e])
        o = np.argsort(vals); vs, ls = vals[o], lab[o]
        dif = np.abs(np.diff(vs))[ls[1:] != ls[:-1]]
        gap = float(dif.min()) if len(dif) else None
    if partial:
        gap = 0.0
    gb = None
    for a_, b_ in zip(blk[:-1], blk[1:]):
        if (a_[2] and b_[3]) or (a_[3] and b_[2]):
            d_ = b_[0] - a_[1]
            gb = d_ if gb is None else min(gb, d_)
    info = {"ambiguous_blocks": amb, "n_occ": n_occ, "n_sea": n_sea, "n_pocket": npocket, "min_occupied_weight": wmin, "max_empty_weight": wmax_empty,
            "nneg": int((e_np < 0).sum()), "gap": gap, "gap_between_blocks": gb, "partly_occupied_blocks": partial}
    if amb or n_occ != n_sea or not cols:
        return None, info
    return torch.cat(cols, dim=1), info


def continue_resolved_f15(ch, ref_prev, f_prev, f_cur, e_cur, U_cur, n_sea, gap_start, max_halvings=6):
    """continue_resolved_f with R25a defect 3's intersample control (the Weyl test per sub-segment) and version 14's block step."""
    dH = clocked_host_sparse(ch, f_cur[0] - f_prev[0], f_cur[1] - f_prev[1])
    norm_dH = float(np.max(np.asarray(abs(dH).sum(axis=1)).ravel())) if dH.nnz else 0.0
    last = None; why = None
    for hv in range(1, max_halvings + 1):
        nsub = 2 ** hv
        dh = norm_dH / nsub
        finest = (hv == max_halvings)
        r_ = ref_prev; ok = True; gap_mid = None; g0 = gap_start; ratio_min = np.inf; weyl_failed = False
        for k in range(1, nsub + 1):
            if g0 is not None and dh > 0:
                ratio_min = min(ratio_min, g0 / (2.0 * dh))
                if 2.0 * dh >= g0:
                    why = "the Weyl test failed at sub-step %d of %d (gap %.3e, 2 ||dh|| %.3e)" % (k, nsub, g0, 2.0 * dh)
                    if not finest:
                        ok = False
                        break
                    weyl_failed = True                 # version 15: at the finest level the sampled continuation is kept, flagged
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                s_ = k / nsub
                e_s, U_s = eigT_t(clocked_gpu(ch, f_prev[0] + s_ * (f_cur[0] - f_prev[0]), f_prev[1] + s_ * (f_cur[1] - f_prev[1])))
            r_, inf_ = continue_reference_resolved_v14(r_, e_s, U_s, n_sea)
            last = inf_
            if r_ is None:
                ok = False; why = "unresolved split at sub-step %d of %d" % (k, nsub)
                break
            g0 = inf_.get("gap_between_blocks")
            if k == nsub // 2:
                gap_mid = inf_["gap"]
        if ok:
            inf_["halvings"] = hv - 1
            inf_["gap_mid"] = gap_mid; inf_["gap_end"] = inf_["gap"]
            inf_["weyl"] = {"norm_dH": norm_dH, "nsub": nsub, "ratio_min": (ratio_min if np.isfinite(ratio_min) else None), "gap_start": gap_start,
                            "failed": weyl_failed, "why": why if weyl_failed else None}
            return r_, inf_
    info = dict(last or {}); info["halvings"] = max_halvings; info["rejected"] = True; info["why"] = why
    info["weyl"] = {"norm_dH": norm_dH, "nsub": 2 ** max_halvings, "gap_start": gap_start}
    return None, info


def inside_energy_band(hmT, cols, f, cut, band=8):
    """inside_energy on the rows of P_in and the columns h_m reaches from them (its range is 4 rows): the same number."""
    n2 = 2 * cut
    hi = min(hmT.shape[0], n2 + band)
    hc = hmT[:n2, :hi] @ cols[:hi, :]
    w = (cols[:n2, :].conj() * hc).sum(dim=0).real.cpu().numpy()
    return float(np.dot(np.asarray(f, float), w))


def clocked_host_sparse(ch, N, v):
    """h_m = (1/2){N, h} + (1/2){v, Q} as a host CSR (Chan.clocked_dense's operator without the dense array)."""
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ ch.hs + ch.hs @ Nd) + 0.5 * (Vd @ ch.Qs + ch.Qs @ Vd)).tocsr()


def dev_csr(A):
    torch = BK["torch"]
    A = A.tocsr()
    return torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                   torch.as_tensor(A.data, dtype=torch.complex128), size=A.shape, device=BK["dev"])


def inside_energy_sp(Hrows, cols, f, n2):
    """inside_energy with h_m's rows of P_in (Hrows, a device CSR of n2 rows): Re sum_j f_j (P_in c_j)^dagger (h_m c_j)."""
    hc = Hrows @ cols
    w = (cols[:n2, :].conj() * hc).sum(dim=0).real.cpu().numpy()
    return float(np.dot(np.asarray(f, float), w))


def energy_cut_current_sp(Hhost, cols, f, cut, half=8):
    """C5.energy_cut_current with (h_m^2)_W = h_m[W, :] h_m[:, W] from the host CSR (W the window of +-half sites about the cut)."""
    torch = BK["torch"]
    lo, hi = 2 * max(cut - half, 0), 2 * min(cut + half, Hhost.shape[0] // 2)
    h2W = T_((Hhost[lo:hi, :] @ Hhost[:, lo:hi]).toarray())
    cW = cols[lo:hi, :]
    DW = (cW * T_(f)[None, :]) @ cW.conj().T
    P = torch.zeros(hi - lo, dtype=torch.complex128, device=BK["dev"]); P[: 2 * cut - lo] = 1.0
    Cm = h2W * P[None, :] - P[:, None] * h2W
    return float((0.5j * (DW.T * Cm).sum()).real)


def continue_fast_info(ref_prev, e, U, n_sea, tol_split=0.1):
    """continue_fast (version 12's endpoint fast path: the negative levels number n_sea, no degenerate block straddles zero and their summed
    weight in the previous reference exceeds n_sea - tol_split, so the block step selects exactly these columns) with the block step's record
    entries: the occupied weights, the gap at zero (here the gap between the occupied and the empty blocks) and the level lists of the next
    Weyl test. Otherwise (None, reason)."""
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    nneg = int((e_np < 0).sum())
    if nneg != n_sea or nneg == 0 or nneg == len(e_np):
        return None, "nneg %d against n_sea %d" % (nneg, n_sea)
    scale = max(float(np.max(np.abs(e_np))), 1e-300)
    if abs(e_np[nneg] - e_np[nneg - 1]) < 1e-8 * scale:
        return None, "a degenerate block straddles zero"
    Uneg = U[:, :nneg]
    w = (ref_prev.conj().T @ Uneg).abs().pow(2).sum(dim=0).cpu().numpy()
    wsum = float(w.sum())
    if wsum <= n_sea - tol_split:
        return None, "summed weight %.4f" % wsum
    gap = float(e_np[nneg] - e_np[nneg - 1])
    return Uneg, {"ambiguous_blocks": 0, "n_occ": nneg, "n_sea": n_sea, "n_pocket": 0, "min_occupied_weight": float(w.min()),
                  "empty_weight_sum": float(n_sea - wsum), "nneg": nneg, "gap": gap, "gap_between_blocks": gap, "partly_occupied_blocks": 0,
                  "occ_e": np.asarray(e_np[:nneg], float), "emp_e": np.asarray(e_np[nneg:], float), "partial_pairs": 0}


def continue_reference_resolved_v17(ref_prev, e, U, n_sea, tol_split=0.1):
    """version 14's block step (continue_reference_resolved_v14, the same selection) returning also the occupied and empty level lists and
    the number of (occupied, empty) pairs inside partly occupied blocks (excluded from the Weyl pair count: a degenerate block's internal
    rotation is fixed by the overlap, not by energy)."""
    torch = BK["torch"]
    A = ref_prev.conj().T @ U
    w = (A.abs() ** 2).sum(dim=0).cpu().numpy()
    e_np = e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)
    cols, n_occ, amb, npocket, wmin, wmax_empty, partial, ppairs = [], 0, 0, 0, 1.0, 0.0, 0, 0
    occ_e, emp_e, blk = [], [], []
    for (i, j) in blocks_of(e_np):
        size = j - i
        if size == 1:
            lam = np.array([w[i]]); V = None
        else:
            M = A[:, i:j].conj().T @ A[:, i:j]
            lam_t, V = torch.linalg.eigh(M)
            lam = lam_t.real.cpu().numpy()
        near1 = np.abs(lam - 1.0) < tol_split; near0 = np.abs(lam) < tol_split
        WB = float(lam.sum()); nB = int(near1.sum())
        if not bool(np.all(near0 | near1)) or abs(WB - nB) >= tol_split:
            amb += 1
            continue
        if nB == size:
            cols.append(U[:, i:j]); wmin = min(wmin, float(lam.min())); occ_e += list(e_np[i:j])
        elif nB > 0:
            cols.append(U[:, i:j] @ V[:, -nB:]); wmin = min(wmin, float(lam[-nB:].min()))
            wmax_empty = max(wmax_empty, float(lam[:-nB].max())); partial += 1; ppairs += nB * (size - nB)
            occ_e += list(e_np[i:j][:nB]); emp_e += list(e_np[i:j][nB:])
        else:
            wmax_empty = max(wmax_empty, float(lam.max())); emp_e += list(e_np[i:j])
        blk.append((float(e_np[i]), float(e_np[j - 1]), nB > 0, nB < size))
        n_occ += nB
        eB = float(np.mean(e_np[i:j]))
        npocket += nB if eB > 0 else (size - nB if eB < 0 else 0)
    gap = None
    if occ_e and emp_e:
        lab = np.concatenate([np.zeros(len(occ_e)), np.ones(len(emp_e))]); vals = np.concatenate([occ_e, emp_e])
        o = np.argsort(vals); vs, ls = vals[o], lab[o]
        dif = np.abs(np.diff(vs))[ls[1:] != ls[:-1]]
        gap = float(dif.min()) if len(dif) else None
    if partial:
        gap = 0.0
    gb = None
    for a_, b_ in zip(blk[:-1], blk[1:]):
        if (a_[2] and b_[3]) or (a_[3] and b_[2]):
            d_ = b_[0] - a_[1]
            gb = d_ if gb is None else min(gb, d_)
    info = {"ambiguous_blocks": amb, "n_occ": n_occ, "n_sea": n_sea, "n_pocket": npocket, "min_occupied_weight": wmin, "max_empty_weight": wmax_empty,
            "nneg": int((e_np < 0).sum()), "gap": gap, "gap_between_blocks": gb, "partly_occupied_blocks": partial,
            "occ_e": np.asarray(occ_e, float), "emp_e": np.asarray(emp_e, float), "partial_pairs": ppairs}
    if amb or n_occ != n_sea or not cols:
        return None, info
    return torch.cat(cols, dim=1), info


def count_failing_pairs(occ_e, emp_e, partial_pairs, thr):
    """the (occupied, empty) level pairs in different blocks within thr = 2 ||Delta h_sub|| of each other: by Weyl's inequality only these
    can meet inside the sub-segment."""
    if occ_e is None or emp_e is None or len(occ_e) == 0 or len(emp_e) == 0:
        return None
    es = np.sort(np.asarray(emp_e, float))
    o = np.asarray(occ_e, float)
    n = int(np.sum(np.searchsorted(es, o + thr, side="right") - np.searchsorted(es, o - thr, side="left")))
    return max(n - int(partial_pairs), 0)


def continue_v17(ch, ref_prev, f_prev, f_cur, e_cur, U_cur, n_sea, gap_start, levels_start, force_subdivide=False, max_halvings=6):
    """R2 continued from the previous accepted configuration (fields f_prev, reference ref_prev) to the trial fields f_cur (engine 18.1 and
    18.5): the Weyl test at nsub = 1 first (2 ||h_m[Delta N, Delta v]|| below gap_start, the previous configuration's gap between occupied
    and empty blocks): passing, the direct endpoint continuation (the fast path, else the block step at the endpoint); failing (or
    force_subdivide, or no gap known), version 15's subdivisions nsub = 2 ... 2^max_halvings with the test per sub-segment, the sampled
    continuation kept and flagged at the finest level, an unresolved split rejecting. info["weyl"]: norm_dH, nsub, ratio_min, margin_min
    (the smallest gap - 2 ||Delta h_sub||), failed, failing_pairs (at the finest level, the largest count over its sub-segments)."""
    t_w = time.time()
    dH = clocked_host_sparse(ch, f_cur[0] - f_prev[0], f_cur[1] - f_prev[1])
    norm_dH = float(np.max(np.asarray(abs(dH).sum(axis=1)).ravel())) if dH.nnz else 0.0
    TM["weyl_s"] += time.time() - t_w
    occ0, emp0, pp0 = levels_start if levels_start is not None else (None, None, 0)
    if not force_subdivide and gap_start is not None and 2.0 * norm_dH < gap_start:
        ref, inf_ = continue_fast_info(ref_prev, e_cur, U_cur, n_sea)
        mode = "endpoint (fast)"
        if ref is None:
            ref, inf_ = continue_reference_resolved_v17(ref_prev, e_cur, U_cur, n_sea)
            mode = "endpoint (blocks)"
        if ref is not None:
            inf_["mode"] = mode; inf_["halvings"] = 0; inf_["gap_mid"] = None; inf_["gap_end"] = inf_["gap"]
            inf_["weyl"] = {"norm_dH": norm_dH, "nsub": 1, "ratio_min": (gap_start / (2.0 * norm_dH)) if norm_dH > 0 else None,
                            "margin_min": gap_start - 2.0 * norm_dH, "gap_start": gap_start, "failed": False, "failing_pairs": 0}
            TM["fast" if mode == "endpoint (fast)" else "block"] += 1
            return ref, inf_
        TM["weyl_pass_unresolved"] += 1                # the endpoint's split unresolved although the test passed: the subdivisions
    last = None; why = None
    for hv in range(1, max_halvings + 1):
        nsub = 2 ** hv
        dh = norm_dH / nsub
        finest = (hv == max_halvings)
        r_ = ref_prev; ok = True; gap_mid = None; g0 = gap_start; lev = (occ0, emp0, pp0)
        ratio_min = np.inf; margin_min = np.inf; weyl_failed = False; fail_pairs = 0
        for k in range(1, nsub + 1):
            if g0 is not None and dh > 0:
                ratio_min = min(ratio_min, g0 / (2.0 * dh)); margin_min = min(margin_min, g0 - 2.0 * dh)
                if 2.0 * dh >= g0:
                    why = "the Weyl test failed at sub-step %d of %d (gap %.3e, 2 ||dh|| %.3e)" % (k, nsub, g0, 2.0 * dh)
                    if not finest:
                        ok = False
                        break
                    weyl_failed = True                 # version 15: at the finest level the sampled continuation is kept, flagged
                    c_ = count_failing_pairs(lev[0], lev[1], lev[2], 2.0 * dh)
                    if c_ is not None:
                        fail_pairs = max(fail_pairs, c_)
            if k == nsub:
                e_s, U_s = e_cur, U_cur
            else:
                s_ = k / nsub
                e_s, U_s = eigT_t(clocked_gpu(ch, f_prev[0] + s_ * (f_cur[0] - f_prev[0]), f_prev[1] + s_ * (f_cur[1] - f_prev[1])))
            r_, inf_ = continue_reference_resolved_v17(r_, e_s, U_s, n_sea)
            last = inf_
            if r_ is None:
                ok = False; why = "unresolved split at sub-step %d of %d" % (k, nsub)
                break
            g0 = inf_.get("gap_between_blocks"); lev = (inf_["occ_e"], inf_["emp_e"], inf_["partial_pairs"])
            if k == nsub // 2:
                gap_mid = inf_["gap"]
        if ok:
            inf_["mode"] = "subdivided (nsub %d)" % nsub
            inf_["halvings"] = hv
            inf_["gap_mid"] = gap_mid; inf_["gap_end"] = inf_["gap"]
            inf_["weyl"] = {"norm_dH": norm_dH, "nsub": nsub, "ratio_min": (ratio_min if np.isfinite(ratio_min) else None),
                            "margin_min": (margin_min if np.isfinite(margin_min) else None), "gap_start": gap_start, "failed": weyl_failed,
                            "failing_pairs": (fail_pairs if weyl_failed else 0), "why": why if weyl_failed else None}
            TM["midpoint"] += 1
            return r_, inf_
    info = {k_: v_ for k_, v_ in (last or {}).items() if k_ not in ("occ_e", "emp_e")}
    info["halvings"] = max_halvings; info["rejected"] = True; info["why"] = why
    info["weyl"] = {"norm_dH": norm_dH, "nsub": 2 ** max_halvings, "gap_start": gap_start}
    return None, info


def eigh_all(chs, N, v, streams=False):
    """the channels' eigendecompositions at the fields (N, v): one after another, or (streams) from threads, each on its own CUDA stream."""
    torch = BK["torch"]
    out = {}
    if not streams or BK["dev"] is None or BK["dev"].type != "cuda":
        for ch in chs:
            out[ch.kappa] = eigT_t(clocked_gpu(ch, N, v))
        return out
    t_ = time.time()
    hms = {ch.kappa: clocked_gpu(ch, N, v) for ch in chs}
    sync_dev()
    ss = {ch.kappa: torch.cuda.Stream(device=BK["dev"]) for ch in chs}

    def work(k):
        with torch.cuda.stream(ss[k]):
            e, U = torch.linalg.eigh(hms[k])
            out[k] = (e.real, U)
        ss[k].synchronize()
    th = [threading.Thread(target=work, args=(ch.kappa,)) for ch in chs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    sync_dev()
    hms.clear()
    TM["eigh_s"] += time.time() - t_; TM["eigh_n"] += len(chs)
    return out


def commutator_norm(ch, N, v, V):
    """||[h_m, R2]||_F for R2 = V V^dagger (V orthonormal columns): [h, P] = (1 - P) h P - P h (1 - P), the two blocks orthogonal, so the
    norm is sqrt(2) ||(1 - V V^dagger) h_m V||_F (one sparse product; no squared cancellation)."""
    torch = BK["torch"]
    H = dev_csr(clocked_host_sparse(ch, N, v))
    X = H @ V
    Y = X - V @ (V.conj().T @ X)
    return float(math.sqrt(2.0) * float(torch.linalg.norm(Y)))


def eig_at(ch, N, v):
    """the eigendecomposition of h_m at (N, v) (the continuation's intermediate configurations)."""
    return eigT_t(clocked_gpu(ch, N, v))


def _num_equal(x, y):
    try:
        return abs(float(x) - float(y)) <= 1e-12 * max(1.0, abs(float(x)))
    except (TypeError, ValueError):
        return str(x) == str(y)


def resume_state(a, chs, R, shell):
    """docstring (e): the stored run replayed by its own rule to the record at T/rs = a.resume_T_over_rs."""
    import engine_collapse_v17 as E17
    torch = BK["torch"]
    d, tag = os.path.split(a.resume_from)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    ra = J["args"]
    for key in ("m", "g", "rs", "R0_over_rs", "reading", "nsites", "ell", "kappa_sea", "tau", "half_offset", "site_fields", "shell", "reference"):
        if not _num_equal(ra.get(key), getattr(a, key)):
            raise RuntimeError("resume: the stored run's %s = %r differs from this run's %r" % (key, ra.get(key), getattr(a, key)))
    E17.OPT["collar_policy"] = ra.get("collar_policy", "flag"); E17.OPT["support_rel"] = ra.get("support_rel", 1e-14)
    E17.OPT["site_fields"] = ra.get("site_fields", "dense")
    rs = a.rs
    t0 = time.time()
    Nc, vc, dust, info0, head0 = E17.prepare_dust(chs, a.g, R, a.ell, rs / (2.0 * a.g * a.m), a.R0_over_rs * rs, a.m, a.reading, shell)
    Ncs = np.asarray(Z["N_cinit"], float); vcs = np.asarray(Z["v_cinit"], float)
    cinit = {"dN": float(np.max(np.abs(Nc - Ncs))), "dv": float(np.max(np.abs(vc - vcs)))}
    E17.tm_reset()
    st = {}
    for ch in chs:
        e, U = E17.eigT(E17.clocked_gpu(ch, Ncs, vcs))
        neg = e < 0
        Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), np.zeros(0)))
        en = e.real.cpu().numpy(); kk = int(neg.sum())
        gap = float(en[kk] - en[kk - 1]) if 0 < kk < len(en) else None
        ref0 = U[:, neg].clone()
        st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": kk, "gap_between_prev": gap,
                        "levels_prev": (np.asarray(en[:kk], float), np.asarray(en[kk:], float), 0), "window": [{"ref": ref0, "f": (Ncs.copy(), vcs.copy()), "gap": gap}]}
        del e, U
    f_ref = (Ncs, vcs)
    Tz = np.asarray(Z["T"], float)
    hits = np.where(np.abs(Tz / rs - a.resume_T_over_rs) < 1e-9)[0]
    if not len(hits):
        raise RuntimeError("resume: no stored record at T/rs %g" % a.resume_T_over_rs)
    k_res = int(hits[0])
    recs_by_T = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    worst_e = worst_j = 0.0
    rows = []
    for k in range(k_res + 1):
        T = float(Tz[k]); rec = recs_by_T.get(round(T, 6), {})
        N_app = np.asarray(Z["N_applied"][k], float); v_app = np.asarray(Z["v_applied"][k], float)
        force = bool(rec.get("verification_differs"))
        eig = E17.eigh_all(chs, N_app, v_app)
        cache = {}
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            s_ = st[ch.kappa]
            ref, ci = E17.continue_v17(ch, s_["ref"], f_ref, (N_app, v_app), e, U, s_["n_sea"], s_.get("gap_between_prev"), s_.get("levels_prev"), force_subdivide=force)
            if ref is None:
                raise RuntimeError("resume: the replay's continuation failed at T/rs %.3f, kappa %d (%s)" % (T / rs, ch.kappa, ci.get("why")))
            cache[ch.kappa] = (e, U, ref, ci)
        eps = np.zeros(a.nsites); Jt = np.zeros(a.nsites)
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            eps += ch.w * (dens(ch.h, s_["sea"]) - dens(ch.h, ref) + dens(ch.h, s_["dust"], s_["f"]))
            Jt += ch.w * (dens(ch.Q, s_["sea"]) - dens(ch.Q, ref) + dens(ch.Q, s_["dust"], s_["f"]))
        es = np.asarray(Z["eps_tot"][k], float); Js = np.asarray(Z["J_tot"][k], float)
        de = float(np.max(np.abs(eps - es))) / max(float(np.max(np.abs(es))), 1e-300)
        dj = float(np.max(np.abs(Jt - Js))) / max(float(np.max(np.abs(Js))), 1e-300)
        worst_e = max(worst_e, de); worst_j = max(worst_j, dj)
        rows.append([T / rs, de, dj])
        if max(de, dj) > 1e-8:
            raise RuntimeError("resume: the replayed sources differ from the stored ones at T/rs %.3f (eps %.2e, J %.2e; acceptance 1e-8)" % (T / rs, de, dj))
        for ch in chs:
            e, U, ref, ci = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            if "occ_e" in ci:
                s_["levels_prev"] = (ci.pop("occ_e"), ci.pop("emp_e"), ci.get("partial_pairs", 0))
            if ci.get("gap_between_blocks") is not None or "weyl" in ci:
                s_["gap_between_prev"] = ci.get("gap_between_blocks")
            s_["window"].append({"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap_between_blocks")})
            s_["window"] = s_["window"][-a.window_max:]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            prop = lambda X, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
        del cache, eig
        f_ref = (N_app, v_app)
        if k % 10 == 0 or k == k_res:
            print("resume: replayed T/rs %.3f (eps %.2e, J %.2e; %.0f s)" % (T / rs, de, dj, time.time() - t0), flush=True)
    fields = {}
    for key in ("T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied"):
        fields[key] = [np.asarray(x, float) if key != "T" else float(x) for x in Z[key][:k_res + 1]]
    for key in ("r_nodes", "N_nodes", "v_nodes"):
        fields[key] = [np.asarray(x, float) for x in Z[key][:k_res + 1]]
    recs = [r for r in J["records"] if r.get("E") is not None and r["T"] <= float(Tz[k_res]) + 1e-9]
    head = dict(J["head"])
    resume = {"from": a.resume_from, "script": J.get("script"), "T_over_rs": float(Tz[k_res] / rs), "records_replayed": k_res + 1,
              "max_eps_rel": worst_e, "max_J_rel": worst_j, "cinit": cinit, "seconds": time.time() - t0, "rows": rows,
              "superseded_records_after": [r["T"] / rs for r in J["records"] if r["T"] > float(Tz[k_res]) + 1e-9]}
    print("resume: %d records replayed to T/rs %.3f in %.0f s; the largest source differences eps %.2e, J %.2e (acceptance 1e-8); the initial data re-prepared to %.2e, %.2e" % (
        k_res + 1, Tz[k_res] / rs, resume["seconds"], worst_e, worst_j, cinit["dN"], cinit["dv"]), flush=True)
    return st, f_ref, {"records": recs, "fields": fields, "N_cinit": Ncs, "v_cinit": vcs, "k_res": k_res, "T_next": float(Tz[k_res]) + a.tau}, head, resume


def resume_state_v25(a, chs, R, shell):
    """docstring: --resume-from an engine_collapse_v17 run (resume_state, the version-19 path) or an engine_collapse_v19/v24 run (its v17
    parent replayed to its own resume record, then its native records by its own rule to a.resume_T_over_rs)."""
    from types import SimpleNamespace
    torch = BK["torch"]
    d, tag = os.path.split(a.resume_from)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    script = J.get("script") or ""
    if "engine_collapse_v17" in script:
        return resume_state(a, chs, R, shell)
    if not any(v_ in script for v_ in ("engine_collapse_v19", "engine_collapse_v24")):
        raise RuntimeError("resume: the stored run %s (%s) is neither a v17 nor a v19/v24 run" % (a.resume_from, script))
    ra = J["args"]
    for key in ("m", "g", "rs", "R0_over_rs", "reading", "nsites", "ell", "kappa_sea", "tau", "half_offset", "site_fields", "shell", "reference"):
        if not _num_equal(ra.get(key), getattr(a, key)):
            raise RuntimeError("resume: the stored run's %s = %r differs from this run's %r" % (key, ra.get(key), getattr(a, key)))
    if not ra.get("resume_from"):
        raise RuntimeError("resume: a v19/v24 parent must itself be resumed from a v17 run (this one starts at T = 0; not implemented)")
    pa = SimpleNamespace(**dict(vars(a)))
    pa.resume_from = getattr(a, "resume_grandparent", None) or ra["resume_from"]
    pa.resume_T_over_rs = float(ra["resume_T_over_rs"])
    st, f_ref, stored0, head, resume0 = resume_state(pa, chs, R, shell)
    for ch in chs:
        st[ch.kappa]["window"] = st[ch.kappa]["window"][-int(ra.get("window_max", 4)):]
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
    rs = a.rs
    Tz = np.asarray(Z["T"], float); TS = np.asarray(S["T"], float)
    t0 = time.time()
    worst_e = worst_j = 0.0; n_rec = 0; mask_eq = 0; mask_pairs = 0
    rows = []
    T_res = float(a.resume_T_over_rs) * rs
    T_last = None
    for i in range(len(TS)):
        T = float(TS[i])
        if T > T_res + 1e-9:
            break
        T_last = T
        kz = int(np.argmin(np.abs(Tz - T)))
        if abs(Tz[kz] - T) > 1e-9:
            raise RuntimeError("resume: the parent's fields hold no record at T/rs %.3f" % (T / rs))
        N_app = np.asarray(Z["N_applied"][kz], float); v_app = np.asarray(Z["v_applied"][kz], float)
        eig = eigh_all(chs, N_app, v_app)
        cache = {}
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            s_ = st[ch.kappa]
            ref, ci, mask = OPS2.continue_sea_v2(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], a.tau, float(ra.get("c_LZ", 1.0)), eig_at, clocked_host_sparse, int(ra.get("window_max", 4)))
            mask = np.asarray(mask, bool)
            smask = np.unpackbits(np.asarray(S["mask_%d" % ch.kappa][i], np.uint8))[:mask.size].astype(bool)
            mask_pairs += 1; mask_eq += int(np.array_equal(mask, smask))
            cache[ch.kappa] = (e, U, ref, ci, mask)
        del eig
        eps = np.zeros(a.nsites); Jt = np.zeros(a.nsites)
        for ch in chs:
            e, U, ref, ci, mask = cache[ch.kappa]; s_ = st[ch.kappa]
            eps += ch.w * (dens(ch.h, s_["sea"]) - dens(ch.h, ref) + dens(ch.h, s_["dust"], s_["f"]))
            Jt += ch.w * (dens(ch.Q, s_["sea"]) - dens(ch.Q, ref) + dens(ch.Q, s_["dust"], s_["f"]))
        es = np.asarray(Z["eps_tot"][kz], float); Js = np.asarray(Z["J_tot"][kz], float)
        de = float(np.max(np.abs(eps - es))) / max(float(np.max(np.abs(es))), 1e-300)
        dj = float(np.max(np.abs(Jt - Js))) / max(float(np.max(np.abs(Js))), 1e-300)
        worst_e = max(worst_e, de); worst_j = max(worst_j, dj); n_rec += 1
        rows.append([T / rs, de, dj])
        if max(de, dj) > 1e-8:
            raise RuntimeError("resume: the parent's replayed sources differ from its stored ones at T/rs %.3f (eps %.2e, J %.2e; acceptance 1e-8)" % (T / rs, de, dj))
        for ch in chs:
            e, U, ref, ci, mask = cache[ch.kappa]; s_ = st[ch.kappa]
            s_["ref"] = ref
            s_["window"].append({"ref": ref, "f": (N_app.copy(), v_app.copy()), "gap": ci.get("gap_between_blocks"),
                                 "weyl_gap": (0.0 if ci.get("partly_occupied_blocks") else ci.get("gap_between_blocks")), "mask": mask})
            s_["window"] = s_["window"][-int(ra.get("window_max", 4)):]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            prop = lambda X, U=U, ph=ph: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
        del cache
        f_ref = (N_app, v_app)
        if n_rec % 10 == 1:
            print("resume (v19 parent): replayed its native record T/rs %.3f (eps %.2e, J %.2e; masks equal %d of %d; %.0f s)" % (T / rs, de, dj, mask_eq, mask_pairs, time.time() - t0), flush=True)
    if T_last is None or abs(T_last - T_res) > 1e-9:
        raise RuntimeError("resume: no native record of the parent at T/rs %g" % a.resume_T_over_rs)
    for ch in chs:
        st[ch.kappa]["window"] = st[ch.kappa]["window"][-1:]
    k_res = int(np.argmin(np.abs(Tz - T_res)))
    fields = {}
    for key in ("T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied"):
        fields[key] = [np.asarray(x_, float) if key != "T" else float(x_) for x_ in Z[key][:k_res + 1]]
    for key in ("r_nodes", "N_nodes", "v_nodes"):
        fields[key] = [np.asarray(x_, float) for x_ in Z[key][:k_res + 1]]
    recs = [r for r in J["records"] if r.get("E") is not None and r["T"] <= T_res + 1e-9]
    head = dict(J["head"])
    resume = {"from": a.resume_from, "script": script, "T_over_rs": T_res / rs, "parent_resume": resume0, "native_records_replayed": n_rec,
              "native_max_eps_rel": worst_e, "native_max_J_rel": worst_j, "native_masks_equal": [mask_eq, mask_pairs], "parent_c_LZ": ra.get("c_LZ"),
              "parent_window_max": ra.get("window_max"), "seconds": time.time() - t0, "rows": rows,
              "superseded_records_after": [r["T"] / rs for r in J["records"] if r["T"] > T_res + 1e-9]}
    print("resume (v19 parent): %d native records replayed to T/rs %.3f; sources within eps %.2e, J %.2e; masks equal to the stored ones in %d of %d channel-records" % (
        n_rec, T_res / rs, worst_e, worst_j, mask_eq, mask_pairs), flush=True)
    return st, f_ref, {"records": recs, "fields": fields, "N_cinit": stored0["N_cinit"], "v_cinit": stored0["v_cinit"], "k_res": k_res, "T_next": T_res + a.tau}, head, resume


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)
    label = ("quasi-static (tau = %g); ell = %d; reference: %s (%s); site fields: %s; engine_fields_v16 (one outward initial-value problem, the momentum "
             "equation kept everywhere, the flat-slice gauge continued and labelled); economy: lagged first iterate, stop |dE/E| < %g and fields < %g, "
             "%s iteration, at most %d iterations (the smallest-residual iterate returned, flagged); tau = %g; collar policy %s; %s") % (
        a.tau, a.ell, a.reference, ("the continued sea, the Landau-Zener-inspired maximum-overlap convention (c_LZ = %g; ops v3: one step, no window)" % a.c_LZ) if a.reference == "R2" else "the propagated sea",
        a.site_fields, a.stop_E, a.stop_fields, a.mix, a.maxit, a.tau, OPT["collar_policy"], DICT24)
    rs = a.rs
    NB = rs / (2.0 * a.g * a.m)
    chs = [Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]
    R = chs[0].R
    shell = [float(x) * rs for x in a.shell.split(",")] if a.shell else None
    kid = kernel_identities(R, a.ell) if a.site_fields == "averaged" else None
    if kid:
        print("kernel identities (site fields averaged): max |K_0 average of 1 - 1| %.2e; max |K_1 average of c r - c R_n| %.2e (relative %.2e); K_0 weight sums %.15f to %.15f" % (
            kid["K0_constant_max_abs"], kid["K1_linear_max_abs"], kid["K1_linear_max_rel"], kid["K0_weight_sums"][0], kid["K0_weight_sums"][1]), flush=True)
    t0 = time.time()
    resume = None
    if a.resume_from:
        st, f_ref, stored, head, resume = resume_state_v25(a, chs, R, shell)
        head = dict(head); head["resume"] = resume; head["label_v19"] = label; head["kernel_identities_v19"] = kid
        E0 = head["E0"]
        N_cinit, v_cinit = stored["N_cinit"], stored["v_cinit"]
        recs = stored["records"]; fields = stored["fields"]
        step0 = stored["k_res"] + 1; T = stored["T_next"]
    else:
        prep_ell = a.prep_ell if a.prep_ell is not None else a.ell
        N, v, dust, info, head = prepare_dust(chs, a.g, R, prep_ell, NB, a.R0_over_rs * rs, a.m, a.reading, shell)
        head["prep_ell"] = prep_ell; head["evolution_ell"] = a.ell
        head["kernel"] = "the Gaussian of standard deviation ell (per-axis variance sigma^2 = ell^2, plus a^2/12 where the cell average enters); prepared at ell %d, evolved at ell %d" % (prep_ell, a.ell)
        N_cinit, v_cinit = N.copy(), v.copy()
        head.update({"label": label, "rs_rest_label": rs, "R0": a.R0_over_rs * rs, "seconds": time.time() - t0, "kernel_identities": kid})
        print("initial data (%s): N_B %.2f E0 %.6f passes %d f_kin %.4f (bulk frame %.4f) discarded %.2e Gram cond %.2e stationarity %.2e; the spline's distance from the dense fields at the sites N %.3e v %.3e; residuals W %.1e 1/N %.1e; source splines %s; centre series %.6f; collar %s (%.0f s)" % (
            label, head["N_B"], head["E0"], head["passes"], head["f_kin"], head["f_kin_bulk_frame"], head["projection"]["discarded_norm_max"],
            head["projection"]["gram_cond_max"], head["projection"]["stationarity_max"], head["interp_floor_N"] or float("nan"), head["interp_floor_v"] or float("nan"),
            head["residual_W"] or float("nan"), head["residual_invN"] or float("nan"), head["source_spline_error"], head["centre_series_ratio"] or float("nan"),
            head["collar_margin"], head["seconds"]), flush=True)
        E0 = head["E0"]
        st = {}
        for ch in chs:
            e, U = eigT(clocked_gpu(ch, N, v))
            neg = e < 0
            Phi_d, f_d = dust.get(ch.kappa, (torch.zeros((2 * a.nsites, 0), dtype=torch.complex128, device=BK["dev"]), np.zeros(0)))
            _en = e.real.cpu().numpy(); _k = int(neg.sum())
            gap = float(_en[_k] - _en[_k - 1]) if 0 < _k < len(_en) else None
            ref0 = U[:, neg].clone()
            st[ch.kappa] = {"sea": U[:, neg].clone(), "dust": Phi_d, "f": f_d, "ref": ref0, "n_sea": _k, "window": [{"ref": ref0, "f": (N.copy(), v.copy()), "gap": gap}]}
            del e, U
        f_ref = (N.copy(), v.copy())
        recs = []
        fields = {"T": [], "N": [], "v": [], "eps_tot": [], "J_tot": [], "eps_pairs": [], "J_pairs": [], "N_applied": [], "v_applied": [], "r_nodes": [], "N_nodes": [], "v_nodes": []}
        step0 = 0; T = 0.0
    cstr = {ch.kappa: STR.ChanStress(ch, torch, BK["dev"]) for ch in chs}
    r_nodes_fixed = FLD.coarse_grid(R, a.ell)
    drtol_last = {"value": None, "T_over_rs": None}
    for r_ in reversed(recs):
        ind_ = r_.get("indicator") or {}
        if ind_.get("delta_rtol_source") == "this record" and ind_.get("delta_rtol") is not None:
            drtol_last = {"value": ind_["delta_rtol"], "T_over_rs": r_["T"] / rs}
            break
    nsteps = int(round(a.T_over_rs * rs / a.tau))
    if a.m1_simpson_steps == "auto":
        q8 = max(1, nsteps // 8)
        simpson_steps = sorted({min(q8, nsteps - 1), nsteps // 2, max(nsteps - q8, 1)}) if nsteps >= 2 else []
    else:
        simpson_steps = [int(x) for x in a.m1_simpson_steps.split(",") if x.strip()]
    sl128 = [float(x) for x in a.slices_c128.split(",") if x.strip()]
    sl64 = [float(x) for x in a.slices_c64.split(",") if x.strip()]
    if (sl128 or sl64) and a.slice_dir:
        os.makedirs(a.slice_dir, exist_ok=True)
    path = os.path.join(a.out, "collapse_%s.json" % a.tag)
    prevW = None
    ind_state = {"first_failure": None}
    for r_ in recs:
        if (r_.get("indicator") or {}).get("indicator_pass") is False:
            ind_state["first_failure"] = r_["T"] / rs
            break
    lz_state = {}
    deg_state = {}
    unconv_state = {"first": None}
    t_run0 = time.time()
    solver_meta = {"rtol": 1e-12, "atol": 1e-16, "collar_policy": OPT["collar_policy"], "support_rel": OPT["support_rel"], "module": "engine_fields_v16",
                   "solver": "solve_constraints_ivp", "site_fields": a.site_fields, "engine": "engine_collapse_v25", "stop_E": a.stop_E, "stop_fields": a.stop_fields,
                   "mix": a.mix, "mix_alpha": (a.mix_alpha if a.mix == "damped" else None), "maxit": a.maxit, "c_LZ": a.c_LZ, "window_max": a.window_max, "delta_rtol_every": a.delta_rtol_every, "streams": int(a.streams),
                   "resumed": resume is not None}
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "S_r_ell": [], "S_t_ell": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc", "b", "sy_k"):
            hand["%s_%s" % (q, part)] = []
    spectra = {"T": []}
    for ch in chs:
        spectra["mask_%d" % ch.kappa] = []; spectra["e_%d" % ch.kappa] = []

    def save_fields():
        np.savez_compressed(os.path.join(a.out, "collapse_%s_fields.npz" % a.tag), R=R, T=np.array(fields["T"]), N=np.array(fields["N"]), v=np.array(fields["v"]),
                            eps_tot=np.array(fields["eps_tot"]), J_tot=np.array(fields["J_tot"]), eps_pairs=np.array(fields["eps_pairs"]), J_pairs=np.array(fields["J_pairs"]),
                            N_applied=np.array(fields["N_applied"]), v_applied=np.array(fields["v_applied"]), N_cinit=N_cinit, v_cinit=v_cinit,
                            g=a.g, ell=a.ell, tau=a.tau, solver=json.dumps(solver_meta),
                            r_nodes=np.array(fields["r_nodes"], dtype=object), N_nodes=np.array(fields["N_nodes"], dtype=object),
                            v_nodes=np.array(fields["v_nodes"], dtype=object), label=label)
        if hand["T"]:
            np.savez_compressed(os.path.join(a.out, "collapse_%s_handoff.npz" % a.tag), R=R, r_nodes=r_nodes_fixed, g=a.g, ell=a.ell, m=a.m, tau=a.tau,
                                kappas=np.array([ch.kappa for ch in chs]), weights=np.array([ch.w for ch in chs]), **{k: np.array(v_) for k, v_ in hand.items()})
        if spectra["T"]:
            np.savez(os.path.join(a.out, "collapse_%s_spectra.npz" % a.tag), **{k: np.array(v_) for k, v_ in spectra.items()})
    last_prop_s = 0.0
    zero = lambda r: np.zeros_like(np.atleast_1d(np.asarray(r, float)))
    for step in range(step0, nsteps + 1):
        tr = time.time()
        tm_reset()
        TM["propagation_prev_s"] = last_prop_s
        mixer = MIX.Anderson(beta=1.0) if a.mix == "anderson" else None
        converged = False
        it_hist = []
        t_d = time.time()
        base = {}
        for ch in chs:
            s_ = st[ch.kappa]
            base[ch.kappa] = (dens(ch.h, s_["sea"]), dens(ch.Q, s_["sea"]), dens(ch.h, s_["dust"], s_["f"]), dens(ch.Q, s_["dust"], s_["f"]))
        TM["densities_s"] += time.time() - t_d

        def sources(cache_):
            t_d = time.time()
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            for ch in chs:
                ref = cache_[ch.kappa][3]
                hs_, Qs_, dd, Jd = base[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        def do_solve(srcs_):
            t_s = time.time()
            out = solve(srcs_[0], srcs_[1], a.g, R, a.ell, None)
            TM["solve_s"] += time.time() - t_s; TM["solve_n"] += 1
            if out[3].get("ivp"):
                TM["solve_nfev"] += int(out[3]["ivp"]["nfev"])
            return out
        N_app, v_app = f_ref[0].copy(), f_ref[1].copy()
        cache = {ch.kappa: (None, None, None, (st[ch.kappa]["ref"] if a.reference == "R2" else st[ch.kappa]["sea"]), {"mode": "lagged"}) for ch in chs}
        TM["lagged"] += len(chs)
        srcs = sources(cache)
        Nn, vn, Wn, info = do_solve(srcs)
        E_prev = FLD.charge(info, a.g) if not info.get("failed") else None
        if not info.get("failed"):
            it_hist.append([None, float(np.max(np.abs(Nn - N_app))), float(np.max(np.abs(vn - v_app)))])
        k_it = 0
        best = None
        if a.mix == "newton" and not info.get("failed"):
            k_it, converged, best, cache, srcs, Nn, vn, Wn, info, N_app, v_app = newton_record(a, chs, st, cache, sources, do_solve, Nn, vn, E_prev, it_hist, T)
        for k_it in (range(1, a.maxit + 1) if (not info.get("failed") and a.mix != "newton") else ()):
            if mixer is not None:
                x = mixer.step(np.concatenate([N_app, v_app]), np.concatenate([Nn, vn]))
                N_app, v_app = x[:a.nsites].copy(), x[a.nsites:].copy()
            elif a.mix == "damped":
                N_app, v_app = N_app + a.mix_alpha * (Nn - N_app), v_app + a.mix_alpha * (vn - v_app)
            else:
                N_app, v_app = Nn.copy(), vn.copy()
            eig = eigh_all(chs, N_app, v_app, bool(a.streams))
            cache = dict(cache)
            for ch in chs:
                e, U = eig.pop(ch.kappa)
                if a.reference == "R2":
                    t_c = time.time()
                    s_ = st[ch.kappa]
                    ref, cinfo, mask = OPS3.continue_sea_v3(ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"], a.tau, a.c_LZ, eig_at, clocked_host_sparse)
                    cinfo["_mask"] = mask
                    TM["continuation_s"] += time.time() - t_c
                    TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
                else:
                    ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
                cache[ch.kappa] = (None, e, U, ref, cinfo)
            del eig
            srcs = sources(cache)
            Nn, vn, Wn, info = do_solve(srcs)
            if info.get("failed"):
                break
            E = FLD.charge(info, a.g)
            dE = abs(E - E_prev) / abs(E)
            E_prev = E
            dN = float(np.max(np.abs(Nn - N_app))); dv = float(np.max(np.abs(vn - v_app)))
            it_hist.append([float(dE), dN, dv])
            resid = max(dN, dv)
            if best is None or resid < best["resid"]:
                best = {"resid": resid, "k_it": k_it, "cache": cache, "srcs": srcs, "sol": (Nn, vn, Wn, info), "app": (N_app.copy(), v_app.copy()), "dE": dE}
            if dE < a.stop_E and dN < a.stop_fields and dv < a.stop_fields:
                converged = True
                break
        if info.get("failed"):
            recs.append({"T": T, "failed": info["failed"], "label": label, "iteration_history": it_hist})
            print("T/rs %.2f: the re-solve is REJECTED (%s); the run stops with the record written" % (T / rs, info["failed"]), flush=True)
            break
        if k_it == 0 or cache[chs[0].kappa][1] is None:
            recs.append({"T": T, "failed": "no eigendecomposition at the accepted fields", "label": label})
            print("T/rs %.2f: no eigendecomposition at the accepted fields; the run stops" % (T / rs), flush=True)
            break
        unconv = None
        if not converged and best is not None:
            cache, srcs = best["cache"], best["srcs"]
            Nn, vn, Wn, info = best["sol"]
            N_app, v_app = best["app"]
            unconv = {"residual": best["resid"], "iteration": best["k_it"], "dE": best["dE"]}
            if unconv_state["first"] is None:
                unconv_state["first"] = T / rs
        best = None
        eps, J, eps_pairs, eps_dust, J_pairs = srcs
        N, v = Nn, vn
        E = FLD.charge(info, a.g)
        # the reference's diagnostics on the accepted state; ||[h_m, R2]||_F per channel; the continuation's record
        wmin, npocket, nrefill, dR2R0, nsub_max = 1.0, {}, {}, 0.0, 1
        comm = {}
        cont = {}
        n_inter = n_inv = 0
        mask_neg_equal = True
        masks = {}
        if a.reference == "R2":
            for ch in chs:
                hm_, e, U, ref, cinfo = cache[ch.kappa]
                s_ = st[ch.kappa]
                masks[ch.kappa] = np.asarray(cinfo.pop("_mask"), bool)
                eneg = (e < 0).cpu().numpy()
                same = bool(np.array_equal(masks[ch.kappa], eneg))
                mask_neg_equal = mask_neg_equal and same
                if cinfo.get("min_occupied_weight") is not None:
                    wmin = min(wmin, cinfo["min_occupied_weight"])
                npocket[ch.kappa] = cinfo["n_pocket"]
                nrefill[ch.kappa] = cinfo["nneg"] - s_["n_sea"]
                nsub_max = max(nsub_max, int(cinfo.get("nsub") or 1))
                n_inter += int(cinfo.get("n_intermediate") or 0); n_inv += int(cinfo.get("inversions") or 0)
                ov = float(((U[:, e < 0].conj().T @ ref).abs() ** 2).sum())
                dR2R0 += math.sqrt(max(cinfo["nneg"] + s_["n_sea"] - 2 * ov, 0.0))
                cont[str(ch.kappa)] = {k_: cinfo.get(k_) for k_ in ("mode", "nsub", "m", "m_eff", "verified", "margin_min", "s_T", "tau_LZ", "norm_dH_G", "window_consistency",
                                                                     "n_intermediate", "min_occupied_weight", "max_empty_weight", "inversions", "n_pocket",
                                                                     "pockets_occupied_positive", "holes_empty_negative", "tol_block", "rotated_blocks",
                                                                     "gap_between_blocks", "partly_occupied_blocks")}
                cont[str(ch.kappa)]["mask_equals_negative_set"] = same
                if not cinfo.get("verified") and ch.kappa not in lz_state:
                    lz_state[ch.kappa] = T / rs
                if cinfo.get("min_occupied_weight") is not None and cinfo["min_occupied_weight"] < 0.5 and ch.kappa not in deg_state:
                    deg_state[ch.kappa] = T / rs
            if a.commutator_every > 0 and step % a.commutator_every == 0:
                t_q = time.time()
                for ch in chs:
                    comm[str(ch.kappa)] = commutator_norm(ch, N_app, v_app, cache[ch.kappa][3])
                TM["commutator_s"] += time.time() - t_q
        else:
            for ch in chs:
                masks[ch.kappa] = np.asarray(cache[ch.kappa][4].pop("_mask"), bool)
        if lz_state:
            first = min(lz_state.values())
            cont_label = "LZ convention (c_LZ = %g) from T/rs %.3f (channels %s); inversions %d; n_pocket %d; intermediate weights %d" % (
                a.c_LZ, first, ",".join(str(k_) for k_ in sorted(lz_state)), n_inv, sum(npocket.values()) if npocket else 0, n_inter)
        else:
            cont_label = "verified (Weyl)"
        if deg_state:
            cont_label += "; reference degraded from T/rs %.3f (channels %s)" % (min(deg_state.values()), ",".join(str(k_) for k_ in sorted(deg_state)))
        unconv_label = ("unconverged (residual %.2e at iteration %d) from T/rs %.3f" % (unconv["residual"], unconv["iteration"], unconv_state["first"])) if unconv else (
            ("converged (unconverged records from T/rs %.3f)" % unconv_state["first"]) if unconv_state["first"] is not None else "converged")
        # the indicator, the r_0 check and the collar (engine 18.1, 20.2)
        t_ce = time.time()
        fresh = bool(step % max(1, a.delta_rtol_every) == 0 or drtol_last["value"] is None)
        ind = indicator(eps, J, a.g, R, a.ell, info, N, v, fresh, drtol_last)
        if fresh and ind["delta_rtol"] is not None:
            drtol_last.update({"value": ind["delta_rtol"], "T_over_rs": T / rs})
        r0c = FLD.r0_check(eps, J, a.g, R, a.ell, info, collar_policy=OPT["collar_policy"], support_rel=OPT["support_rel"]) if fresh else None
        TM["indicator_s"] += time.time() - t_ce
        if not ind["indicator_pass"] and ind_state["first_failure"] is None:
            ind_state["first_failure"] = T / rs
        ind_label = "indicator pass" if ind_state["first_failure"] is None else ("indicator fail from T/rs %.2f" % ind_state["first_failure"])
        gauge = info.get("gauge") or {}
        gauge_label = "intact" if not gauge.get("intervals") else ("continued on %s" % ", ".join("[%.2f, %.2f]" % (x_, y_) for x_, y_ in gauge["intervals"]))
        # M1's balance (17.3 (4)): the old configuration's terms before the reference and h are replaced
        cut = int(min(max(round(info["r_collar"]) - 1, 2), a.nsites - 10))
        Y1 = Y2 = IEend = 0.0
        t_m1 = time.time()
        if recs and recs[-1].get("E") is not None:
            for ch in chs:
                s_ = st[ch.kappa]; H_old = clocked_host_sparse(ch, f_ref[0], f_ref[1]); Hr_old = dev_csr(H_old[:2 * cut, :]); ref_old = s_["ref"]; ref_new = cache[ch.kappa][3]
                Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
                Y1 += ch.w * inside_energy_sp(Hr_old, Ccols, fC, 2 * cut)
                Y2 += ch.w * inside_energy_sp(Hr_old, ref_new, np.ones(ref_new.shape[1]), 2 * cut)
                colsE = torch.cat([s_["sea"], s_["dust"], ref_old], dim=1)
                fE = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref_old.shape[1])])
                IEend += ch.w * energy_cut_current_sp(H_old, colsE, fE, cut)
                del Hr_old, colsE
        TM["M1_s"] += time.time() - t_m1
        for ch in chs:
            ci_ = cache[ch.kappa][4]
            st[ch.kappa]["ref"] = cache[ch.kappa][3]
            st[ch.kappa]["window"].append({"ref": cache[ch.kappa][3], "f": (N_app.copy(), v_app.copy()), "gap": ci_.get("gap_between_blocks"),
                                           "weyl_gap": ci_.get("weyl_gap", ci_.get("gap_between_blocks")), "mask": masks.get(ch.kappa)})
            st[ch.kappa]["window"] = st[ch.kappa]["window"][-1:]
        f_ref = (N_app.copy(), v_app.copy())
        tz = FLD.trapping_radius(N, v, R, info)
        Rt = tz["zeros"][-1] if tz["zeros"] else None
        r_in = tz["zeros"][0] if len(tz["zeros"]) >= 2 else None
        jets = surface_jets(info, Rt, a.g, E)
        kT = jets["kappa_T"] if jets else None
        M_wg = FLD.mass_function(info, a.g, Rt) if Rt else None
        M_ms = FLD.misner_sharp(info, a.g, Rt) if Rt else None
        Rc = Rsh = None
        tot = eps_dust.sum()
        if tot > 0:
            cum = np.cumsum(eps_dust); Rsh = float(R[min(int(np.searchsorted(cum, 0.99 * tot)), a.nsites - 1)]); Rc = float(eps_dust @ R / tot)
        late = None
        if Rt:
            rr = np.linspace(max(Rt - 5 * a.ell, 0.0), Rt + 5 * a.ell, 101)
            rho_l = np.abs(info["rho_ell"](rr)); J_l = np.abs(info["J_ell"](rr))
            rmax = max(float(np.max(np.abs(info["rho_ell"](np.linspace(0, info["r_collar"], 2001))))), 1e-300)
            jmax = max(float(np.max(np.abs(info["J_ell"](np.linspace(0, info["r_collar"], 2001))))), 1e-300)
            late = {"rho_ratio": float(rho_l.max() / rmax), "J_ratio": float(J_l.max() / jmax)}
            late["flag"] = bool(late["rho_ratio"] < 1e-6 and late["J_ratio"] < 1e-6)
        IE = Hin = 0.0
        X1 = X2 = 0.0
        t_m1 = time.time()
        for ch in chs:
            _hm, e, U, ref, _ci = cache[ch.kappa]
            H_new = clocked_host_sparse(ch, N_app, v_app); Hr = dev_csr(H_new[:2 * cut, :])
            s_ = st[ch.kappa]
            cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
            fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
            IE += ch.w * energy_cut_current_sp(H_new, cols, fs, cut)
            Hin += ch.w * inside_energy_sp(Hr, cols, fs, 2 * cut)
            Ccols = torch.cat([s_["sea"], s_["dust"]], dim=1); fC = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"]])
            X1 += ch.w * inside_energy_sp(Hr, Ccols, fC, 2 * cut)
            X2 += ch.w * inside_energy_sp(Hr, ref, np.ones(ref.shape[1]), 2 * cut)
            del Hr, cols
        TM["M1_s"] += time.time() - t_m1
        IE_mid = None
        if step in simpson_steps and step < nsteps and not a.benchmark:
            t_sm = time.time()
            IE_mid = 0.0
            for ch in chs:
                _hm, e, U, ref, _ci = cache[ch.kappa]
                ph2 = torch.exp(-1j * e.to(torch.complex128) * (0.5 * a.tau))
                s_ = st[ch.kappa]
                halfp = lambda X: U @ (ph2[:, None] * (U.conj().T @ X))
                cols = torch.cat([halfp(s_["sea"]), halfp(s_["dust"]) if s_["dust"].shape[1] else s_["dust"], ref], dim=1)
                fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
                IE_mid += ch.w * energy_cut_current_sp(clocked_host_sparse(ch, N_app, v_app), cols, fs, cut)
                del cols
            TM["simpson_s"] += time.time() - t_sm
        # the handoff quantities (20.1): the stresses (dust and pairs apart), <N^2>_n, the sea's and R2's shell energies
        t_h = time.time()
        Pd = None; Pp = None
        eps_sea = np.zeros(a.nsites); eps_R2 = np.zeros(a.nsites)
        for ch in chs:
            s_ = st[ch.kappa]; ref = cache[ch.kappa][3]; cs = cstr[ch.kappa]
            pd_ = STR.stress_parts(cs, s_["dust"], s_["f"])
            pp_ = STR.stress_parts(cs, torch.cat([s_["sea"], ref], dim=1), np.concatenate([np.ones(s_["sea"].shape[1]), -np.ones(ref.shape[1])]))
            for P_, p_ in ((0, pd_), (1, pp_)):
                p_ = dict(p_); p_["sy_k"] = ch.kappa * p_["sy"]
                if P_ == 0:
                    Pd = {k_: ch.w * np.asarray(v_, float) for k_, v_ in p_.items()} if Pd is None else {k_: Pd[k_] + ch.w * np.asarray(p_[k_], float) for k_ in Pd}
                else:
                    Pp = {k_: ch.w * np.asarray(v_, float) for k_, v_ in p_.items()} if Pp is None else {k_: Pp[k_] + ch.w * np.asarray(p_[k_], float) for k_ in Pp}
            eps_sea += ch.w * base[ch.kappa][0]; eps_R2 += ch.w * dens(ch.h, ref)
        SVd = STR.shell_values(Pd, R, a.m); SVp = STR.shell_values(Pp, R, a.m)
        Sr_tot = SVd["S_r"] + SVp["S_r"]; St_tot = SVd["S_t"] + SVp["S_t"]
        Sr_l, St_l, tr_err = STR.smooth_stress(Sr_tot, St_tot, R, a.ell, r_nodes_fixed)
        N2avg, _vz, _nn = FLD.site_fields_kernel_average({"N_dense": (lambda r_: info["N_dense"](r_) ** 2), "v_dense": zero}, R, a.ell)
        vir = {}
        for lab_, SV in (("dust", SVd), ("pairs", SVp)):
            sc = max(float(np.max(np.abs(SV["virial_rhs"]))), 1e-300)
            vir[lab_] = float(np.max(np.abs(SV["virial_lhs"] - SV["virial_rhs"]))) / sc
        stress_rec = {"identity_max_abs": {"dust": SVd["identity_max_abs"], "pairs": SVp["identity_max_abs"]}, "virial_rel": vir, "trace_identity_rel": tr_err,
                      "F_total_at_last_face": float(SVd["F_face"][-1] + SVp["F_face"][-1]), "seconds": time.time() - t_h}
        hand["T"].append(T); hand["N2avg"].append(np.asarray(N2avg, float)); hand["eps_sea"].append(eps_sea); hand["eps_R2"].append(eps_R2); hand["eps_dust"].append(np.asarray(eps_dust, float))
        hand["S_r_ell"].append(Sr_l); hand["S_t_ell"].append(St_l)
        for part, SV, P_ in (("dust", SVd, Pd), ("pairs", SVp, Pp)):
            hand["F_face_%s" % part].append(SV["F_face"]); hand["fc_%s" % part].append(np.asarray(P_["fc"], float))
            hand["b_%s" % part].append(np.asarray(P_["b"], float)); hand["sy_k_%s" % part].append(np.asarray(P_["sy_k"], float))
        spectra["T"].append(T)
        for ch in chs:
            spectra["mask_%d" % ch.kappa].append(np.packbits(masks[ch.kappa]))
            spectra["e_%d" % ch.kappa].append(cache[ch.kappa][1].cpu().numpy().astype(np.float64))
        TM["handoff_s"] = time.time() - t_h
        saved = None
        tslice = [x for x in sl128 + sl64 if abs(T / rs - x) < 1e-6]
        if tslice and a.slice_dir:
            dt_ = np.complex128 if any(abs(T / rs - x) < 1e-6 for x in sl128) else np.complex64
            arrs = {"T": T, "rs": rs, "N_applied": N_app, "v_applied": v_app, "R": R, "kappas": np.array([ch.kappa for ch in chs]), "weights": np.array([ch.w for ch in chs]),
                    "dtype": str(np.dtype(dt_))}
            for ch in chs:
                _hm, e, U, ref, _ci = cache[ch.kappa]; s_ = st[ch.kappa]
                arrs["sea_%d" % ch.kappa] = s_["sea"].cpu().numpy().astype(dt_)
                arrs["dust_%d" % ch.kappa] = s_["dust"].cpu().numpy().astype(dt_)
                arrs["f_%d" % ch.kappa] = np.asarray(s_["f"], float)
                arrs["R2_%d" % ch.kappa] = ref.cpu().numpy().astype(dt_)
                arrs["e_%d" % ch.kappa] = e.cpu().numpy()
                arrs["mask_%d" % ch.kappa] = np.packbits(masks[ch.kappa])
            saved = os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, T / rs))
            np.savez(saved, **arrs)
            del arrs
        rec = {"T": T, "T_over_rs": T / rs, "E": E, "E_over_E0": E / E0, "E_dust_shells": float(eps_dust.sum()), "E_pairs_shells": float(eps_pairs.sum()),
               "pairs_inside_Rt": (float(eps_pairs[R < Rt].sum()) if Rt else None), "pairs_outside_Rt": (float(eps_pairs[R >= Rt].sum()) if Rt else float(eps_pairs.sum())),
               "R_t": Rt, "R_t_over_2gE": (Rt / (2 * a.g * E) if Rt else None), "W_over_2g_at_Rt": M_wg, "misner_sharp_at_Rt": M_ms,
               "R_t_over_2g_MS": (Rt / (2 * a.g * M_ms) if (Rt and M_ms) else None), "kappa_T": kT, "four_gE_kappa_T": (4.0 * a.g * E * kT if kT is not None else None),
               "surface_jets": jets, "r_in": r_in, "trapping_zeros": tz["zeros"], "tangencies": tz["tangencies"], "min_N_minus_absv": tz["min"], "r_min_N_minus_absv": tz["r_min"],
               "late_surface": late, "N_min": float(N.min()), "R_sh": Rsh, "R_c": Rc, "v_at_Rc": (float(info["v_dense"](Rc)[0]) if Rc else None),
               "threshold_radius": threshold_radii(N, v, R, a.m, [1, 2, a.kappa_sea]), "n_pocket": npocket, "n_refill_R0": nrefill, "R2_minus_R0_F": dR2R0,
               "min_occupied_weight": wmin, "nsub_max": nsub_max, "M1_I_E": IE, "H_in": Hin,
               "iterations": k_it + 1, "eigendecompositions": TM["eigh_n"], "converged": converged, "unconverged": unconv, "iteration_history": it_hist,
               "continuation": cont, "continuation_label": cont_label, "mask_equals_negative_set": mask_neg_equal, "inversions_total": n_inv, "intermediate_weights_total": n_inter,
               "commutator_h_R2_F": comm, "gauge": {k_: gauge.get(k_) for k_ in ("label", "intervals", "min_U_over_U_ref", "delta_L_bound", "delta_L_bound_two_resolutions", "lapse_unresolved", "segments", "zeros", "stretches", "near_tangencies")},
               "regime": info.get("regime"), "r0_check": r0c, "stress": stress_rec, "slice_saved": saved,
               "solver": {"module": "engine_fields_v16.solve_constraints_ivp", "nfev": (info.get("ivp") or {}).get("nfev"), "N_centre": (info.get("ivp") or {}).get("N_centre"),
                          "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"), "residual_matching_region": info.get("residual_matching_region"),
                          "residual_ladder": info.get("residual_ladder"), "source_spline_error": info.get("source_spline_error"), "centre_series_ratio": info.get("centre_series_ratio"),
                          "centre_series_check": info.get("centre_series_check"), "collar_margin": info.get("collar_margin"), "W_tail": info.get("W_tail"),
                          "W_roundoff_negative_points": info.get("W_roundoff_negative_points"), "W_negative": info.get("W_negative"), "site_fields": a.site_fields, "kernel_norms": info.get("kernel_norms")},
               "spectrum": info["spectrum"], "r_collar": info["r_collar"], "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
               "label": label, "seconds": time.time() - tr, "indicator": ind,
               "record_label": "engine_fields_v16 (rtol %g, atol %g); v31 (v30 with the Newton-Krylov solve as an option; mixer %s%s; ops v3);" % (SOLVE_TOL["rtol"], SOLVE_TOL["atol"], a.mix, (" a = %g" % a.mix_alpha) if a.mix == "damped" else "") + " kernel: the Gaussian of standard deviation ell, sigma^2 = ell^2 (plus a^2/12 where the cell average enters), evolved at ell %d, prepared at ell %d; tau = %g; continuation: %s; stresses: the lane's declared stress (the Noether flux of Q at unit clock and zero flow); rows: flat-slice ADM, K^ADM = -K, raw, floor in the stress's trace (25.3); J_ell = <P_r> = -J_r of the goal's display; gauge: %s; %s; %s; collar: %s" % (
                   a.ell, (a.prep_ell if a.prep_ell is not None else a.ell), a.tau, cont_label.replace("LZ convention (c_LZ = %g)" % a.c_LZ, "the LZ-inspired maximum-overlap convention (c_LZ = %g, the lane's declaration)" % a.c_LZ), gauge_label, unconv_label, ind_label, ind["collar_label"]),
               "timing": dict(TM),
               "M1_parts": {"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2, "I_E_end_prev_interval": IEend, "I_E_start": IE, "cut": cut, "I_E_mid": IE_mid}}
        if resume is not None and not any(q.get("engine") in ("engine_collapse_v19", "engine_collapse_v24", "engine_collapse_v25") for q in recs):
            rec["resumed_from"] = {"run": resume["from"], "T_over_rs": resume["T_over_rs"]}
        rec["engine"] = "engine_collapse_v25"
        prevM = recs[-1].get("M1_parts") if (recs and recs[-1].get("E") is not None) else None
        if prevM is not None:
            dT = T - recs[-1]["T"]
            a_exact = Y1 - prevM["X1"]; b_work = (X1 - X2) - (Y1 - Y2); c_reset = -(Y2 - prevM["X2"]); dH = (X1 - X2) - (prevM["X1"] - prevM["X2"])
            rec["M1_balance"] = {"dHin": dH, "evolution_exact": a_exact, "evolution_trapezoid": 0.5 * dT * (prevM["I_E_start"] + IEend), "field_work": b_work,
                                 "reference_reset": c_reset, "closure": a_exact + b_work + c_reset - dH, "cut_changed": bool(prevM.get("cut") != cut),
                                 "per_unit_time": {"dHin": dH / dT, "evolution": a_exact / dT, "field_work": b_work / dT, "reference_reset": c_reset / dT}}
            if prevM.get("I_E_mid") is not None:
                rec["M1_balance"]["evolution_simpson"] = dT / 6.0 * (prevM["I_E_start"] + 4.0 * prevM["I_E_mid"] + IEend)
                rec["M1_balance"]["trapezoid_minus_exact"] = rec["M1_balance"]["evolution_trapezoid"] - a_exact
                rec["M1_balance"]["simpson_minus_exact"] = rec["M1_balance"]["evolution_simpson"] - a_exact
        if recs and recs[-1].get("E") is not None:
            dT = T - recs[-1]["T"]
            rec["dE_dT"] = (E - recs[-1]["E"]) / dT
            rec["dHin_dT"] = (Hin - recs[-1]["H_in"]) / dT
            rec["balance_d_E_minus_Hin_dT"] = rec["dE_dT"] - rec["dHin_dT"]
            if Rc is not None and recs[-1].get("R_c") is not None:
                rec["dRc_dT"] = (Rc - recs[-1]["R_c"]) / dT
        if step % 10 == 0 and prevW is not None and T > prevW[1]:
            rec["M9"] = {}
            for lab, rr_ in (("0.5 R_sh", 0.5 * Rsh if Rsh else None), ("R_t", Rt), ("collar", float(info["r_collar"]) - 1.0)):
                if rr_:
                    ci = int(min(max(round(rr_), 2), a.nsites - 10))
                    I_c = 0.0
                    for ch in chs:
                        _hm, e, U, ref, _ci = cache[ch.kappa]
                        H_new = clocked_host_sparse(ch, N_app, v_app)
                        s_ = st[ch.kappa]
                        cols = torch.cat([s_["sea"], s_["dust"], ref], dim=1)
                        fs = np.concatenate([np.ones(s_["sea"].shape[1]), s_["f"], -np.ones(ref.shape[1])])
                        I_c += ch.w * energy_cut_current_sp(H_new, cols, fs, ci)
                        del cols
                    rec["M9"][lab] = {"r": rr_, "cut_site": ci, "dW_over_2g_dT": float((info["W_field"](rr_)[0] - prevW[0](rr_)[0]) / (2 * a.g * (T - prevW[1]))),
                                      "I_E_cut": I_c}
        if step % 10 == 0:
            prevW = (info["W_field"], T)
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy()); fields["eps_tot"].append(eps.copy()); fields["J_tot"].append(J.copy())
        fields["eps_pairs"].append(eps_pairs.copy()); fields["J_pairs"].append(J_pairs.copy())
        fields["N_applied"].append(N_app.copy()); fields["v_applied"].append(v_app.copy())
        fields["r_nodes"].append(info["r_nodes"]); fields["N_nodes"].append(info["N_nodes"]); fields["v_nodes"].append(info["v_nodes"])
        oe = jets["over_exact"] if jets else None
        mb = rec.get("M1_balance")
        pc = ind.get("collar_padded") or {}
        last_h = it_hist[-1] if it_hist else [None, float("nan"), float("nan")]
        cmin = [c_.get("min_occupied_weight") for c_ in cont.values() if c_.get("min_occupied_weight") is not None]
        cmax = [c_.get("max_empty_weight") for c_ in cont.values() if c_.get("max_empty_weight") is not None]
        tlz = [c_.get("tau_LZ") for c_ in cont.values() if c_.get("tau_LZ") is not None]
        print("T/rs %.3f E/E0 %.7f dust %.4f pairs %.3e R_t %s (2gE %.3f; d1..d4 of N+v over exact %s; r_in %s) min(N-|v|) %.4f R_sh %s R_c %s pockets %s inversions %d intermediate %d mask=neg %s nsub max %d it %d (eigh %d)%s M1 %.3e dE/dT %s res %.1e %.1e; B_estimate %s; r0 check dE %s; gauge %s; regime max %.2e; continuation [%s]; weights min occ %s max empty %s; tau_LZ min %s; [h,R2] max %s; stress identity %.1e %.1e virial %.1e %.1e trace %.1e; M1 balance: evolution %s work %s reset %s closure %s [%s] last iteration dE %s dN %.1e dv %.1e (%.0f s; eigh %d in %.1f s, continuation %.1f s, densities %.1f s, solves %d in %.1f s, indicator %.1f s, handoff %.1f s, M1 %.1f s, propagation %.1f s)" % (
            T / rs, E / E0, eps_dust.sum(), eps_pairs.sum(), ("%.3f" % Rt) if Rt else None, 2 * a.g * E, (", ".join("%.4f" % q for q in oe)) if oe else None,
            ("%.3f" % r_in) if r_in else None, tz["min"], Rsh, ("%.2f" % Rc) if Rc else None, sum(npocket.values()) if npocket else 0, n_inv, n_inter, mask_neg_equal, nsub_max,
            k_it + 1, TM["eigh_n"], ("" if converged else " UNCONVERGED (residual %.1e)" % (unconv["residual"] if unconv else float("nan"))), IE,
            ("%.3e" % rec["dE_dT"]) if "dE_dT" in rec else "-", info.get("residual_W") or 0.0, info.get("residual_invN") or 0.0,
            ("%.1e" % ind["B_estimate"]) if ind["B_estimate"] is not None else "none", ("%.1e" % r0c["dE_rel"]) if (r0c and "dE_rel" in r0c) else "-",
            gauge_label, (info.get("regime") or {}).get("max_ratio", float("nan")), cont_label, ("%.3e" % min(cmin)) if cmin else "-", ("%.3e" % max(cmax)) if cmax else "-",
            ("%.2f" % min(tlz)) if tlz else "-", ("%.1e" % max(comm.values())) if comm else "-",
            SVd["identity_max_abs"], SVp["identity_max_abs"], vir["dust"], vir["pairs"], tr_err,
            ("%.3e" % mb["evolution_exact"]) if mb else "-", ("%.3e" % mb["field_work"]) if mb else "-", ("%.3e" % mb["reference_reset"]) if mb else "-",
            ("%.1e" % mb["closure"]) if mb else "-", rec["record_label"],
            ("%.1e" % last_h[0]) if last_h[0] is not None else "-", last_h[1], last_h[2], time.time() - tr, TM["eigh_n"], TM["eigh_s"], TM["continuation_s"],
            TM["densities_s"], TM["solve_n"], TM["solve_s"], TM["indicator_s"], TM.get("handoff_s", 0.0), TM["M1_s"], TM["propagation_prev_s"]), flush=True)
        if step > step0 and (step - step0) % 10 == 0:
            last = [q["seconds"] for q in recs[-10:] if "seconds" in q]
            mean = float(np.mean(last)) if last else 0.0
            remain = (nsteps - step) * mean
            end = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + remain))
            print("projection at T/rs %.2f: %.0f s per re-solve over the last 10; %d re-solves left; the end about %s UTC" % (T / rs, mean, nsteps - step, end), flush=True)
        if step % 5 == 0 or step == nsteps or step == step0:
            with open(path, "w") as fh:
                json.dump({"script": "engine_collapse_v31.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
            save_fields()
        if a.benchmark or step >= nsteps:
            break
        t_p = time.time()
        for ch in chs:
            _hm, e, U, ref, _ci = cache[ch.kappa]
            ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
            s_ = st[ch.kappa]
            prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
            s_["sea"] = prop(s_["sea"])
            if s_["dust"].shape[1]:
                s_["dust"] = prop(s_["dust"])
            cache[ch.kappa] = (None, None, None, ref, _ci)
            del e, U
        sync_dev()
        last_prop_s = time.time() - t_p
        T += a.tau
    with open(path, "w") as fh:
        json.dump({"script": "engine_collapse_v31.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
    save_fields()


def newton_record(a, chs, st, cache0, sources, do_solve, Nn0, vn0, E0, it_hist, T):
    """version 31: the record's fixed point by Newton-Krylov (docstring); returns the loop's variables."""
    from scipy.optimize import newton_krylov, NoConvergence
    n = a.nsites
    nk = {"best": None, "n": 0, "E_prev": E0, "failed": None}

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if a.reference == "R2":
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = OPS3.continue_sea_v3(ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"], a.tau, a.c_LZ, eig_at, clocked_host_sparse)
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        srcs_ = sources(cache_)
        return cache_, srcs_, do_solve(srcs_)

    def F(x):
        N_a = np.array(x[:n], dtype=float); v_a = np.array(x[n:], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        nk["n"] += 1
        if info_.get("failed"):
            nk["failed"] = info_["failed"]
            return np.full(2 * n, 1.0)
        E_ = FLD.charge(info_, a.g)
        dN_ = float(np.max(np.abs(Nn_ - N_a))); dv_ = float(np.max(np.abs(vn_ - v_a)))
        dE_ = abs(E_ - nk["E_prev"]) / abs(E_); nk["E_prev"] = E_
        it_hist.append([float(dE_), dN_, dv_])
        r_ = max(dN_, dv_)
        if nk["best"] is None or r_ < nk["best"]["resid"]:
            nk["best"] = {"resid": r_, "k_it": nk["n"], "cache": cache_, "srcs": srcs_, "sol": sol_, "app": (N_a.copy(), v_a.copy()), "dE": dE_, "E": E_}
        return np.concatenate([Nn_ - N_a, vn_ - v_a])

    t0 = time.time()
    x0 = np.concatenate([np.asarray(Nn0, float), np.asarray(vn0, float)])
    status = "converged"
    try:
        newton_krylov(F, x0, f_tol=a.stop_fields, maxiter=a.nk_maxiter, method="lgmres", inner_maxiter=a.nk_inner, verbose=False)
    except NoConvergence:
        status = "no convergence within %d Newton steps" % a.nk_maxiter
    except Exception as ex:  # a failed solve inside a line search, a non-finite residual
        status = "stopped: %s" % str(ex)[:120]
    b = nk["best"]
    if b is None:
        return nk["n"], False, None, cache0, None, Nn0, vn0, None, {"failed": nk["failed"] or "no evaluation"}, Nn0, vn0
    Nn_b, vn_b, Wn_b, info_b = b["sol"]
    cache_c, srcs_c, sol_c = evalG(np.array(Nn_b, float), np.array(vn_b, float))
    nk["n"] += 1
    dE_c, r_c = None, None
    if not sol_c[3].get("failed"):
        E_c = FLD.charge(sol_c[3], a.g)
        dE_c = abs(E_c - b["E"]) / abs(E_c)
        r_c = max(float(np.max(np.abs(sol_c[0] - Nn_b))), float(np.max(np.abs(sol_c[1] - vn_b))))
        it_hist.append([float(dE_c), float(np.max(np.abs(sol_c[0] - Nn_b))), float(np.max(np.abs(sol_c[1] - vn_b)))])
    conv = b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E
    b["dE"] = dE_c if dE_c is not None else b["dE"]
    print("T/rs %.3f newton: %s; %d evaluations in %.0f s; kept residual %.2e (evaluation %d), the confirming plain step's residual %s and abs dE/E %s; %s" % (
        T / (a.rs), status, nk["n"], time.time() - t0, b["resid"], b["k_it"], "%.2e" % r_c if r_c is not None else "none", "%.1e" % dE_c if dE_c is not None else "none",
        "converged" if conv else "not converged"), flush=True)
    N_app_b, v_app_b = b["app"]
    return nk["n"], conv, b, b["cache"], b["srcs"], Nn_b, vn_b, Wn_b, info_b, N_app_b, v_app_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--reading", default="A", choices=["A", "B"])
    ap.add_argument("--shell", default=None, help="R_in,R_out in units of r_s (S-shell)")
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--prep-ell", dest="prep_ell", type=int, default=None, help="the averaging scale of the initial data (default --ell)")
    ap.add_argument("--field-rtol", type=float, default=1e-12); ap.add_argument("--field-atol", type=float, default=1e-16)
    ap.add_argument("--kappa-sea", type=int, default=10)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--T-over-rs", type=float, default=40.0)
    ap.add_argument("--reference", default="R2", choices=["R2", "R1"])
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--collar-policy", default="flag", choices=["flag", "fail"])
    ap.add_argument("--support-rel", type=float, default=1e-14)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--site-fields", default="averaged", choices=["averaged", "dense"])
    ap.add_argument("--stop-E", type=float, default=1e-7)
    ap.add_argument("--stop-fields", type=float, default=1e-6)
    ap.add_argument("--maxit", type=int, default=20)
    ap.add_argument("--mix", default="plain", choices=["plain", "anderson", "damped", "newton"]); ap.add_argument("--nk-maxiter", type=int, default=8); ap.add_argument("--nk-inner", type=int, default=15); ap.add_argument("--mix-alpha", type=float, default=0.5, help="the damped mixer's step (version 30)")
    ap.add_argument("--delta-rtol-every", type=int, default=10)
    ap.add_argument("--commutator-every", type=int, default=1)
    ap.add_argument("--streams", type=int, default=0)
    ap.add_argument("--m1-simpson-steps", default="auto")
    ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=1.0)
    ap.add_argument("--window-max", type=int, default=4)
    ap.add_argument("--resume-from", default=None, help="dir/tag of a stored engine_collapse_v17 run")
    ap.add_argument("--resume-T-over-rs", type=float, default=None)
    ap.add_argument("--resume-grandparent", default=None, help="the v17 run under a v19/v24 parent (overrides the parent's stored path)")
    ap.add_argument("--slices-c128", default="")
    ap.add_argument("--slices-c64", default="")
    ap.add_argument("--slice-dir", default=None)
    a = ap.parse_args()
    OPT["collar_policy"] = a.collar_policy; OPT["support_rel"] = a.support_rel; OPT["site_fields"] = a.site_fields
    SOLVE_TOL["rtol"] = a.field_rtol; SOLVE_TOL["atol"] = a.field_atol
    C5.backend(a.gpu)
    run(a)


if __name__ == "__main__":
    main()
