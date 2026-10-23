#!/usr/bin/env python
"""engine_collapse_v36d.py (the solve lane's analyst builder 11; engine spec 53.1 (1), (2)): version 36d = version 36c with one change
inside --nk-set resolve (with the option off nothing of it runs, so the defaults are version 36b bitwise as in 36c):
    when round 1 meets the stop rule after refusing a CROSS-LOWER trial (a Newton step of round 1 at which a crossing trial, not taken, had a
    residual below every non-crossing trial of that step: flags_v36c_records_v1's definition), a second round runs from round 1's far-side
    start with that set held (one round; no further round follows it). When two rounds meet the stop rule, the record keeps the set with the
    fewer flips from the previous record's set (that set itself when it is one), ties by the larger smallest gap at the kept iterate; the
    other solve is stored beside it, labelled "two self-consistent sets": its summary in the solve's "v36d" entry and the record's "v36d"
    entry, its fields (solved and applied N, v, its sources) in <out>/collapse_<tag>_twosets_T<T/r_s>.npz. Round 1 keeps the start's set as
    in 36c. At the first record after a slice restart the previous record's set is not known: two meeting rounds are then kept by the larger
    smallest gap (printed). When round 1 is kept after the second round ran, the solve's E_prev is restored to round 1's kept E.
Version 36c's docstring follows.
engine_collapse_v36c.py (the solve lane's analyst builder 9; engine spec 51.1 (1), the owner: "Fix new solver"): version 36c = version
36b with one option, off by default, so that the defaults are version 36b bitwise (the gate: 2 records from the upcast T/r_s 16 slice against
the v32 continuation's 16.125 and 16.25, compare_v36b_gate_v1's form):
--nk-set resolve (needs --nk-jvp exact and --nk-line-search halving): the set-resolving Newton solve. Every evaluation of the record's map
    records its occupied set (the continuation's masks of every channel, a signature of their bits) with its residual and the smallest
    occupied-to-unoccupied eigenvalue gap. A solve holds its start's set (the set of its first evaluation) and runs version 32's Newton loop
    under the exact J.v; a line-search trial whose evaluation lands on another set is not accepted (its residual logged), the first such trial
    of the solve kept as the far-side start, and the halving continues past 2^-H (H = --nk-halvings) to 2^-12 for as long as the trials cross;
    the kept iterate of a round is the best evaluation on its held set. eta is reset to 1e-3 at each round. A round ending by the halving rule
    with a far-side start runs a further round from that start with its set held, at most 3 rounds. The solve keeps the round meeting the stop
    rule; when two do, the previous record's set is preferred and the record is labelled "two self-consistent sets"; when none does, the round
    with the smallest kept residual, and the end rule acts as in version 36b. Printed per Newton step: the round, the held set, the trials'
    sets (crossing or not, flips against the held set) and residuals, the smallest gap at the iterate and at each trial, the dropped pairs of
    the step's J.v products. Each solve's json entry carries a "v36c" entry (rounds, sets, every evaluation's set); the record a "v36c" entry
    (its set, equal or not to the previous record's). With the option off none of this runs and nothing is added.
Version 36b's docstring follows.
engine_collapse_v36b.py (solve builder 5; REVIEWA_gpu_economics_v1 sec. 5, line 4): version 36b = version 36 with one option, off by
default, so that the defaults are version 36 bitwise (the gate: compare_v36b_gate_v1 on the cost run's 3 records, G1's form):
--pair-workers 2: inside the exact J.v of --nk-jvp exact, the 2 field solves of each central-difference product (the source perturbed by
    +t and by -t) are computed at once in 2 persistent worker processes (engine_pairsolve_v1: spawned once at the first product, each running
    this module's own solve() with the same field module, kernel cache, OPT and SOLVE_TOL), not threads and not per-product forks; the
    parent waits for both and uses their N and v exactly as it used its own two serial solves. A worker error or a lost worker makes that
    product's pair run serially in the parent (counted). Every other solve (the F evaluations, the confirming step) is unchanged. The record
    carries a "v36b" entry (per record: products paired, their seconds, serial fallbacks, the workers' cache hits and misses) only with the option on;
    the parent's own kernel-cache counters then omit the workers' solves.
Version 36's docstring follows.
engine_collapse_v36.py (the lane's analyst builder; engine spec 42.1 (3)): version 36 = version 35 with options, each off by default, so
that the defaults are version 35 bitwise (the gate G1v36: compare_v33_gate_v1-form on S11's 3 records and the cost run's):
--field-kernel-cache (42.1 (3) (ii), the profile's finding): the field solve's Gaussian shell kernels (engine_fields_v4.kernels, 3.0 s of a
    4.2 s solve at 2048 sites) cached by the exact bytes of their arguments (engine_fields_kcache_v1): every solve returns bit for bit what
    the uncached solve returns (tested on 4 source sets: bitwise equal, 4.2 s -> 1.06 s); the hits, misses and bytes print in the record.
--v36-fixes (42.1 (3) (iii), #65 item 7, each confirmed at the named lines): (a) in continue_v17 and the v15 continuation, a sub-step whose
    gap is None or not finite with dh > 0 fails the Weyl test (version 35 skipped the test there and could return failed = False) and the
    count of such sub-steps prints; (b) in version 32's own Newton loop, F() moves E_prev only at an iterate that becomes the best (the
    accepted iterate of the residual-decreasing line search; version 35 moved it at every evaluation); (c) a failed field solve returns an
    infinite residual (rejected by the line search) in place of version 35's fabricated residual 1.0.
--field-tangent analytic (42.1 (3) (i)) is NOT in this version: ordered, not yet built (the builder's report).
Version 35's docstring follows.
engine_collapse_v35.py (the lane's analyst builder; engine spec 40.4 (2) (b), inbox #53): version 35 = version 34 with one option, off by
default, so that the defaults are version 34 bitwise (the gate: compare_v33_gate_v1.py on 3 records, G1's form):
--nk-jvp exact: inside version 32's own Newton loop (nk_loop_v32), the Krylov Jacobian's product J.v is the exact linearization of the
    record's map x -> G(x) - x at the Newton iterate instead of scipy's finite difference of F: the first-order change of each channel's
    occupied projector along the trial field direction (dN, dv) in the iterate's eigenbasis with the continuation's assignment held,
    dP = X U_o^dagger + U_o X^dagger, X = U_u M, M_ji = (U_j^dagger dh U_i)/(e_i - e_j) (i in the mask, j not; dh = h_m(dN, dv), h_m linear in
    the fields), the sources' change -sum_kappa w dens(A, dP) for A = h and Q (dens(A, P) = the site sums of Re diag(A P)), and the field
    solve's response to that change by a central difference of the field solve alone (step t = 1e-4 max(1, max|eps|, max|J|)/max|d src|);
    J.v = (dN_out - dN, dv_out - dv). The iterate's eigendecompositions are those of its own evaluation (kept while the option is on), so a
    product costs no eigendecomposition. Per record the products' count and seconds, the pairs with |e_i - e_j| below 1e-12 dropped, and the
    largest |diag(P_ref - P_mask)| (the continued reference against the masked eigenprojector, 0 where the continuation ends on the
    eigenvectors) are printed in the record's jvp_v35 entry. Only the Newton direction changes; every F evaluation, the line search, the stop
    rule and the confirming step are version 34's. Version 34's docstring follows.
engine_collapse_v34.py (builder analyst, the HORIZON version 2 builder of the lead's turn 47; engine spec 39.4 (2): "The decisive test of
the transport: a certified segment of 2 records across the formation, from the stored complex128 slice nearest before T/r_s 1.375 (the
builder names it), with ceil(2 ||dh||/gap) sub-steps per channel so that the Weyl certificate passes on every channel (a channel that still
fails marked), on box GPU 0 when s4cubic ends ...; compared with the LZ run at the same clocks in E/E0, the roots, min(N - |v|) and the
masks"; the lane's approval of 01:16 UTC: the segment from the T/r_s 1.0 slice over the 4 records 1.125 to 1.5). Version 34 = version 33
with the options below, each off by default, so that the defaults are version 33 bitwise (the gate: compare_v33_gate_v1.py on 3 records):
(a) --resume-slice S --resume-run DIR/TAG: the run restarts from the stored complex128 slice S of the run DIR/TAG (a v33 or v34 run with
    the same arguments): the run's records, fields rows, handoff rows, spectra rows and S11 rows to the slice's record copied; the state at
    the slice (the sea's and the ball's columns, occupations, the reference R2, its mask, the applied fields) taken from the slice; the
    record's eigendecomposition recomputed at the slice's applied fields (its eigenvalues against the stored ones printed) and the state
    propagated over the record's step by it, as the run did; the window's entry the slice's reference, fields, mask and the record's stored
    gaps; the previous record's base densities and applied fields kept for version 32's source continuation; the run continues from the next
    record (the M9 diagnostic, every 10th record from a previous W, restarts with the run);
(b) --segment-end-T-over-rs X: the run ends after its record at T/r_s X (written);
(c) --weyl-certified full|eigvals [--weyl-refine-max J]: the reference's continuation by engine_ops_v34_cert.continue_sea_v34 (the lead's
    ceil(2 ||dh||/gap) sub-steps per channel with the Weyl test on every one, a too-small start gap halving its sub-step at most J times,
    a channel that still fails marked); route full: version 3's maximum-overlap assignment at every sub-step on the eigendecomposition there;
    route eigvals: the eigenvalues only at the sub-steps (scipy's banded solver), the continued occupation the same sorted indices where every
    sub-step passes (Weyl's inequality), the endpoint's columns on the trial fields' eigenvectors; each record's continuation entry carries
    the certificate (the lead's count, the pieces, the refinements, the failing pieces, the margin, the calls and seconds);
(d) --weyl-oracle: at every record's accepted fields, per channel, the full route through the same pieces and version 3's one-step
    continuation, against the run's reference (the masks' equality, the subspace residuals) in the record's certificate_oracle.
Version 33's docstring follows.
engine_collapse_v33.py (builder analyst; the quasi-frozen ruling's S11, owner-approved (inbox #36): the massless ball pathfinder at
r_s/a 16; brief lead/BRIEF_D_S11_builder.md). Version 33 = version 32 with the options below, each off by default, so that the defaults are
version 32 bitwise (every new argument off; the module rebinding below only under --fields-v17):
(a) m a = 0 admitted: the rest number N_B = r_s/(2 g m) is computed only for m > 0 (version 32 divided by m before the preparation); m = 0
requires --massless-ball and --massless-ball requires m = 0 (no mass regulator anywhere);
(b) --massless-ball: the initial data are a massless Fermi ball prepared from the chiral Slater seeds re-derived from P28 sec. 5.1 (an idea
file, not a source): per seeded channel (--ball-orbitals kappa:b, ...) the orthonormal family psi_j(x) = (sigma/pi)^(1/2) (sigma - i x)^j /
(sigma + i x)^(j + 1), x = R - x0, j = 0 .. b - 1, on the ingoing chiral spinor (1, -1)/sqrt(2) (on it h(k) = -sin k > 0 and the group
velocity -cos k < 0 for k < 0; the two-step current negative), times the inward boost e^{-i k0 x}; x0 = --ball-x0-over-R0 R0 and sigma =
--ball-sigma-over-R0 R0 with R0 = --ball-R0-over-rs r_s (4); the seeds projected against the physical sea P_0 = 1(h_m(N, v) < 0) at the
current fields, Phi_0 = (1 - P_0) Psi [Psi^dag (1 - P_0) Psi]^(-1/2), the Gram eigenvalues recorded (the rank test); the fields solved from
the ball's sources alone (the sea is P_0 = R2 at T = 0, so the pairs vanish there) and the fixed point iterated (Anderson) to --ball-tol; the
inward momentum parameter k0 tuned after the constraint solve so that the solved charge FLD.charge equals --ball-E-target (r_s/(2g) by
default: E a = 32 at r_s/a 16, G/a^2 1/4), by bisection alternated with false position on [0, --ball-k0-max]; when the charge at k0 = 0
is not below the target (the boost cannot be negative) the seed count is lowered by one in the channel holding the most seeds until it is
(the builder's rule; each step printed and kept in the head's seed_reductions); the bare packet energy is never the charge (it is printed
beside it); the ball's columns are projected at exactly the fields the run starts from;
(c) S11's per-record readouts (with --massless-ball): the averaged sources rho_ell, J_ell (total, the ball's, the pairs' by the run's own
SmoothedSources5) and the positive-branch reconstruction (the field module's W = r v^2 = U_phys = 2 g N^2 M with M the Misner-Sharp mass
W/(2 g N^2), v_pos = -(W/r)^(1/2)) at the coarse nodes in
collapse_<tag>_s11.npz beside the raw shell totals of the fields npz; both constraint residuals; the mass function's minimum outside the
central averaging cell r > ell and the continued intervals; the ball's own weight and energy beyond each --s11-collars radius (its leakage
on the lattice branch); the record line "S11 ..." per record;
(d) the S11 stop (--s11-stop 1): a record with M(r) < 0 (W below -1e-12 of its largest |W|) at some r > ell ends the run at that record
(written; nothing propagated from it), the three cases of P28 sec. 4 told apart in end_s11 (numerical: nonnegative averaged sources with
an inward current inside the negative interval; a source incompatible with the positive branch: rho + w J < 0 there on a Weyl-certified
reference; (iii) or (ii): the same on a reference not Weyl-certified on that record), and the raw state (the sea's and the ball's columns,
R2), the raw and averaged sources, both residuals and the positive-branch reconstruction saved as s11_audit_<tag>_T....npz;
(e) --fields-v17: the field module engine_fields_v17 (engine_fields_v16 plus N' = 0 on an empty interior; bitwise v16 when there is none)
in place of engine_fields_v16 for every solve.
Version 32's docstring follows.
engine_collapse_v32.py (builder analyst; engine spec 32.1 (3) and 32.6 item 3, the lead's turn 40: the sigma/a 2 member's last bounded
solve): version 31 with the options below, each off by default (every new argument None), so that the defaults are version 31 bitwise
(the gate: chk31's three sigma/a 8 records on pod 1, every json number other than wall-clock seconds and every fields array). With every
option off the record solve is version 31's newton_record, unchanged, and the run label, the record label and the solver entry are version
31's; with any option on --mix newton is required and the record solve is newton_record_v32:
(a) --nk-line-search halving [--nk-halvings H; 6 when absent]: the Newton solve runs this file's Newton loop (nk_loop_v32: scipy 1.14.1's
nonlin_solve loop for newton_krylov, the same Krylov Jacobian and its finite-difference step, the forcing terms gamma 0.9, eta_max 0.9999,
eta_treshold 0.1, eta 1e-3 at the start, the stop max|F| <= --stop-fields tested before each Newton step) with scipy's Armijo search on
|F|_2^2 (which takes the full Newton step when it finds no suitable step) replaced by a backtracking search on the fixed-point residual
max(max|N_out - N_app|, max|v_out - v_app|) along the Newton direction: the steps 1, 1/2, ..., 2^-H in turn, the first whose residual falls
below the current point's taken; when none falls the solve ends there ("the residual did not fall in H halvings") and, as in version 31,
the smallest-residual evaluation is the solve's kept iterate;
(b) --nk-krylov K and --nk-restarts R: LGMRES with K Krylov vectors per cycle (scipy's inner_m) and up to R restarts (R + 1 cycles), scipy's
10 augmentation vectors carried across the Newton steps as in version 31; the Newton steps are --nk-maxiter (30 on the members). In
version 31 --nk-inner never reached the solver: scipy's KrylovJacobian sets LGMRES's maxiter to 1 over inner_maxiter (its outer iteration
replaced by the Newton steps), so every version 31 Newton step ran one LGMRES cycle of scipy's default 30 vectors plus the carried
augmentation vectors (at most 10), whatever --nk-inner said; that call is kept unchanged when (a) and (b) are off;
(c) --source-continuation two-stage: on a record whose direct solve (version 31's start, the first solve's output) still fails the stop
rule, the fixed point is solved first (stage 1) with the sources of the midpoint covariance C_prev + (C_new - C_prev)/2 from the previous
record's accepted applied fields (the source densities are linear in the covariance, so the midpoint's base densities of the sea and of
the dust are the mean of the previous record's and this record's; the reference R2 is continued from the previous record at the trial
fields as in every solve), then (stage 2) with this record's sources from stage 1's kept iterate's applied fields; every evaluation enters
the record's iteration_history; the record keeps the smallest-residual evaluation of this record's map (the direct solve's or stage 2's),
and that iterate's confirming plain step is the history's last entry (repeated when the direct solve's iterate is kept); the continuation
needs a previous record of this run (none at the run's first record, which is then the direct solve's);
(d) --end-above X (32.1 (3)'s reading addition, fixed in turn 40): a record whose kept residual exceeds X after the continuation ends the
member at that record: the record is written, nothing is propagated from it, the json carries end_v32, and the run's last line says so.
Per record (options on): solve_v32 (each solve's start, status, evaluations, seconds, kept residual and evaluation, confirming residual and
|dE/E|, the line search's trials per Newton step, its slice of iteration_history; the solve kept; the continuation's outcome); one
"newton (...)" line per solve and the solves' evaluations and residuals at the end of the record line. Version 31's record fields stand
(the kept residual in unconverged, the confirming step's residual last in iteration_history, the unconverged flags). Nothing else changes;
version 31's docstring follows.
engine_collapse_v31.py (builder analyst; a diagnostic for engine spec 30.3's sigma/a 2 member, not a rule change: version 30's
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
import engine_ops_v34_cert as CERT

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
V36 = {"fixes": False, "gap_none_substeps": 0, "kcache": False}   # version 36
PAIR = {"n": 0, "workers": None, "cfg": None, "products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []}   # version 36b


def pair_start():
    """version 36b: spawn the PAIR["n"] persistent workers once (engine_pairsolve_v1.worker_main)."""
    import multiprocessing as mp
    import engine_pairsolve_v1 as PW
    ctx = mp.get_context("spawn")
    ws = []
    for _ in range(PAIR["n"]):
        pc, cc = ctx.Pipe()
        p = ctx.Process(target=PW.worker_main, args=(cc, PAIR["cfg"]), daemon=True)
        p.start()
        ws.append((p, pc))
    for p, pc in ws:
        tag, pid = pc.recv()
    PAIR["workers"] = ws
    print("version 36b: %d pair workers started (pids %s)" % (len(ws), ",".join(str(p.pid) for p, _ in ws)), flush=True)


def pair_solve(do_solve, src_list, g, R, ell):
    """version 36b: the solves of src_list (2 source tuples) at once on the workers; returns do_solve-shaped tuples (N, v, None, info) with
    info holding "failed" and "ivp" nfev only (all the exact J.v reads). A worker error makes the whole pair serial in the parent."""
    t0 = time.time()
    if PAIR["workers"] is None:
        pair_start()
    outs = None
    try:
        for (p, pc), s_ in zip(PAIR["workers"], src_list):
            pc.send((np.asarray(s_[0], float), np.asarray(s_[1], float), g, R, ell))
        res = [pc.recv() for (p, pc), _ in zip(PAIR["workers"], src_list)]
        if all(r_[0] == "ok" for r_ in res):
            outs = []
            for r_ in res:
                _, Nn, vn, failed, nfev, dh, dm = r_
                info = {"failed": failed} if failed else {}
                if nfev is not None:
                    info["ivp"] = {"nfev": nfev}
                outs.append((Nn, vn, None, info))
                PAIR["hits"] += dh; PAIR["misses"] += dm
                TM["solve_n"] += 1
                if nfev is not None:
                    TM["solve_nfev"] += nfev
            TM["solve_s"] += time.time() - t0
        else:
            PAIR["errors"].append([r_[1] for r_ in res if r_[0] != "ok"][0])
    except Exception as ex:
        PAIR["errors"].append("%s: %s" % (type(ex).__name__, str(ex)[:200]))
        PAIR["workers"] = None
    if outs is None:
        PAIR["fallbacks"] += 1
        outs = [do_solve(s_) for s_ in src_list]
    PAIR["products"] += 1; PAIR["seconds"] += time.time() - t0
    return outs
JV = {"products": 0, "seconds": 0.0, "dropped_pairs": 0, "extra_evaluations": 0, "ref_mismatch": None, "_chk": 0}   # version 35 (--nk-jvp exact)
NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other


def set_sig_v36c(masks):
    """version 36c: the occupied set of an evaluation (dict kappa -> mask): its signature, the occupied count and the concatenated bits."""
    import hashlib
    h = hashlib.sha1()
    parts = []
    for k_ in sorted(masks):
        m_ = np.asarray(masks[k_], bool)
        h.update(str(k_).encode()); h.update(str(m_.size).encode()); h.update(np.packbits(m_).tobytes())
        parts.append(m_)
    mc = np.concatenate(parts) if parts else np.zeros(0, bool)
    return h.hexdigest()[:16], int(mc.sum()), mc


def min_gap_v36c(cache_, chs):
    """version 36c: the smallest |e_occupied - e_unoccupied| over the channels (adjacent in the sorted spectrum where the mask changes)."""
    g_ = float("inf")
    for ch in chs:
        e_ = cache_[ch.kappa][1]
        ev = e_.detach().cpu().numpy() if hasattr(e_, "detach") else np.asarray(e_)
        m_ = np.asarray(cache_[ch.kappa][4]["_mask"], bool)
        o_ = np.argsort(ev, kind="stable"); es = ev[o_]; ms = m_[o_]
        d_ = np.nonzero(ms[1:] != ms[:-1])[0]
        if d_.size:
            g_ = min(g_, float(np.min(np.abs(es[d_ + 1] - es[d_]))))
    return g_


def nk_loop_v36c(a, F, x0, RS):
    """version 36c: version 32's Newton loop (nk_loop_v32 with the exact J.v and the halving search) with the held set: a trial on another set
    is not accepted (logged; the first kept in RS["far"]), the halving continued to 2^-12 while the trials cross. Returns the status; the
    steps are appended to RS["steps"]; RS["by_halving"] says whether the round ended by the halving rule."""
    from numpy.linalg import norm
    from scipy.optimize._nonlin import KrylovJacobian
    kw = {}
    if a.nk_krylov is not None:
        kw["inner_inner_m"] = int(a.nk_krylov)

    class _ExactKrylov(KrylovJacobian):
        def matvec(self, v_):
            return F.jvp(self.x0, np.asarray(v_, dtype=float))
    jac = _ExactKrylov(rdiff=None, method="lgmres", inner_maxiter=a.nk_inner, outer_k=10, **kw)
    if a.nk_restarts is not None:
        jac.method_kw["maxiter"] = 1 + int(a.nk_restarts)
    H = nk_halvings_eff(a); HMAX = 12
    RS["by_halving"] = False
    x = np.asarray(x0, dtype=float).flatten().copy()
    Fx = np.asarray(F(x), dtype=float).flatten()
    gap_x = RS["last"]["gap"]
    Fx_norm = norm(Fx)
    jac.setup(x.copy(), Fx, F)
    gamma, eta_max, eta_treshold, eta = 0.9, 0.9999, 0.1, 1e-3
    steps = RS["steps"]
    for n_ in range(a.nk_maxiter):
        r0 = float(np.max(np.abs(Fx)))
        if r0 <= a.stop_fields:
            return "converged"
        tol = min(eta, eta * Fx_norm)
        dp0 = JV["dropped_pairs"]
        dx = -jac.solve(Fx, tol=tol)
        dropped = JV["dropped_pairs"] - dp0
        if norm(dx) == 0:
            return "stopped: the Krylov solve returned a zero direction at Newton step %d" % (n_ + 1)
        s, trials, acc, crossed_last = 1.0, [], None, False
        for h_ in range(HMAX + 1):
            if h_ > H and not crossed_last:
                break
            xt = x + s * dx
            Ft = np.asarray(F(xt), dtype=float).flatten()
            rt = float(np.max(np.abs(Ft)))
            L = RS["last"]
            cross = not L["same"]
            trials.append({"step": s, "residual": rt, "set": L["set_id"], "crosses": cross, "flips_vs_held": L["flips"], "gap": L["gap"]})
            if cross:
                if RS["far"] is None:
                    RS["far"] = {"x": xt.copy(), "set_id": L["set_id"], "residual": rt, "newton_step": n_ + 1, "step": s}
                crossed_last = True
            else:
                crossed_last = False
                if rt < r0:
                    acc = (s, xt, Ft, L["gap"])
                    break
            s *= 0.5
        e_ = {"round": RS["round"], "newton_step": n_ + 1, "held_set": RS["sets"][RS["held"]]["id"], "residual_before": r0, "gap_at_iterate": gap_x,
              "dropped_pairs": dropped, "trials": trials, "accepted_step": (acc[0] if acc is not None else None)}
        steps.append(e_)
        print("  v36c round %d Newton step %d: held set %d, residual %.2e, gap at iterate %.3e, dropped pairs %d; trials %s; accepted %s" % (
            RS["round"], n_ + 1, e_["held_set"], r0, gap_x, dropped,
            " ".join("[%g %.2e set %d%s gap %.2e]" % (q["step"], q["residual"], q["set"], (" crosses, %d flips" % q["flips_vs_held"]) if q["crosses"] else "", q["gap"]) for q in trials),
            ("%g" % acc[0]) if acc is not None else "none"), flush=True)
        if acc is None:
            RS["by_halving"] = True
            return "the residual did not fall on the held set in the halvings (to 2^-%d while the trials crossed) at Newton step %d (the solve ends there)" % (
                (len(trials) - 1), n_ + 1)
        s, x, Fx, gap_x = acc
        Fx_norm_new = norm(Fx)
        jac.update(x.copy(), Fx)
        eta_A = gamma * Fx_norm_new ** 2 / Fx_norm ** 2
        if gamma * eta ** 2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma * eta ** 2))
        Fx_norm = Fx_norm_new
    r_ = float(np.max(np.abs(Fx)))
    return "converged" if r_ <= a.stop_fields else "no convergence within %d Newton steps" % a.nk_maxiter


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
            if V36["fixes"] and dh > 0 and (g0 is None or not np.isfinite(g0)):
                V36["gap_none_substeps"] += 1                 # version 36 (a): no finite gap fails the test
                why = "version 36: no finite gap at sub-step %d of %d" % (k, nsub)
                if not finest:
                    ok = False
                    break
                weyl_failed = True
            elif g0 is not None and dh > 0:
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
            if V36["fixes"] and dh > 0 and (g0 is None or not np.isfinite(g0)):
                V36["gap_none_substeps"] += 1                 # version 36 (a): no finite gap fails the test
                why = "version 36: no finite gap at sub-step %d of %d" % (k, nsub)
                if not finest:
                    ok = False
                    break
                weyl_failed = True
            elif g0 is not None and dh > 0:
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


def continue_ref_v34(a, ch, window, f_cur, e, U, n_sea):
    """version 34: the reference's continuation: version 3's (engine_ops_v3.continue_sea_v3, the defaults) unless --weyl-certified names a
    route, then engine_ops_v34_cert.continue_sea_v34 (the docstring's (c))."""
    if getattr(a, "weyl_certified", None):
        return CERT.continue_sea_v34(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse, a.weyl_certified, a.weyl_refine_max)
    return OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, a.tau, a.c_LZ, eig_at, clocked_host_sparse)


def v34_on(a):
    return bool(getattr(a, "weyl_certified", None) or getattr(a, "resume_slice", None) or getattr(a, "weyl_oracle", False)
                or getattr(a, "segment_end_T_over_rs", None) is not None)


def v34_desc(a):
    parts = []
    if a.resume_slice:
        parts.append("restarted from the stored slice %s of the run %s" % (a.resume_slice, a.resume_run))
    if a.weyl_certified:
        parts.append("the reference continued with the Weyl certificate on every sub-step (ceil(2 ||dh||/gap) sub-steps per channel, a too-small start gap "
                     "halving its sub-step at most %d times, a channel that still fails marked; route %s)" % (a.weyl_refine_max, a.weyl_certified))
    else:
        parts.append("the reference on version 3's continuation (the LZ convention where the Weyl test fails)")
    if a.weyl_oracle:
        parts.append("the full-route oracle and version 3's one-step continuation at every record's accepted fields")
    if a.segment_end_T_over_rs is not None:
        parts.append("the segment ends after T/rs %g" % a.segment_end_T_over_rs)
    return "; ".join(parts)


def resume_slice_v34(a, chs, R):
    """version 34 (a): the state at a stored complex128 slice of a v33 or v34 run, the run's stored rows to it, the state propagated over the
    slice's record step by the record's eigendecomposition recomputed at the slice's applied fields. Returns the loop's state."""
    torch = BK["torch"]
    rdir, rtag = os.path.split(a.resume_run)
    J = json.load(open(os.path.join(rdir, "collapse_%s.json" % rtag)))
    for k_ in ("m", "g", "rs", "nsites", "ell", "kappa_sea", "tau", "reference", "site_fields", "c_LZ", "mix", "massless_ball", "fields_v17"):
        if J["args"].get(k_) != getattr(a, k_):
            raise SystemExit("version 34 (a): the run's argument %s is %r, this call's %r" % (k_, J["args"].get(k_), getattr(a, k_)))
    Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)
    Z = np.load(a.resume_slice)
    if str(Z["dtype"]) != "complex128":
        raise SystemExit("version 34 (a): the restart needs a complex128 slice; %s holds %s" % (a.resume_slice, Z["dtype"]))
    Ts = float(Z["T"]); rs = a.rs
    recs = [r for r in J["records"] if r.get("T") is not None and r["T"] <= Ts + 1e-9]
    if not recs or abs(recs[-1]["T"] - Ts) > 1e-9 or recs[-1].get("E") is None:
        raise SystemExit("version 34 (a): the run's json holds no accepted record at the slice's clock T/rs %.4f" % (Ts / rs))
    Tz = np.asarray(Zf["T"], float); ks = int(np.argmin(np.abs(Tz - Ts)))
    if abs(Tz[ks] - Ts) > 1e-9:
        raise SystemExit("version 34 (a): the fields npz holds no row at the slice's clock")
    N_s = np.asarray(Z["N_applied"], float); v_s = np.asarray(Z["v_applied"], float)
    fchk = max(float(np.max(np.abs(N_s - Zf["N_applied"][ks]))), float(np.max(np.abs(v_s - Zf["v_applied"][ks]))))
    rec_s = recs[-1]
    st = {}; base = {}; chk = {}
    for ch in chs:
        k = ch.kappa
        e, U = eigT(clocked_gpu(ch, N_s, v_s))
        de = float(np.max(np.abs(e.real.cpu().numpy() - np.asarray(Z["e_%d" % k], float))))
        sea = torch.as_tensor(np.asarray(Z["sea_%d" % k]), dtype=torch.complex128, device=BK["dev"])
        dust = torch.as_tensor(np.asarray(Z["dust_%d" % k]), dtype=torch.complex128, device=BK["dev"])
        f = np.asarray(Z["f_%d" % k], float)
        ref = torch.as_tensor(np.asarray(Z["R2_%d" % k]), dtype=torch.complex128, device=BK["dev"])
        mask = np.unpackbits(np.asarray(Z["mask_%d" % k], np.uint8))[: 2 * a.nsites].astype(bool)
        c_ = (rec_s.get("continuation") or {}).get(str(k)) or {}
        gap = c_.get("gap_between_blocks"); wg = (0.0 if c_.get("partly_occupied_blocks") else gap)
        base[k] = (dens(ch.h, sea), dens(ch.Q, sea), dens(ch.h, dust, f), dens(ch.Q, dust, f))
        ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
        prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
        sea_n = prop(sea)
        dust_n = prop(dust) if dust.shape[1] else dust
        st[k] = {"sea": sea_n, "dust": dust_n, "f": f, "ref": ref, "n_sea": int(ref.shape[1]),
                 "window": [{"ref": ref, "f": (N_s.copy(), v_s.copy()), "gap": gap, "weyl_gap": wg, "mask": mask}]}
        chk[str(k)] = {"eig_vs_stored_max": de, "mask_bits": int(mask.sum()), "gap": gap, "weyl_gap": wg, "n_sea": int(ref.shape[1]), "n_ball": int(dust.shape[1])}
        del e, U
    sync_dev()
    fields = {k_: [] for k_ in ("T", "N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied", "r_nodes", "N_nodes", "v_nodes")}
    for i in range(ks + 1):
        for k_ in fields:
            fields[k_].append(float(Zf["T"][i]) if k_ == "T" else np.asarray(Zf[k_][i]))
    pre = {"hand": {}, "spectra": {}, "s11": {}, "lz_state": {}, "deg_state": {}, "unconv_first": None}
    for nm, fn in (("hand", "handoff"), ("spectra", "spectra"), ("s11", "s11")):
        p_ = os.path.join(rdir, "collapse_%s_%s.npz" % (rtag, fn))
        if not os.path.exists(p_):
            continue
        Zx = np.load(p_, allow_pickle=True)
        Tx = np.asarray(Zx["T"], float); keep = int(np.sum(Tx <= Ts + 1e-9))
        for k_ in Zx.files:
            arr = Zx[k_]
            if arr.ndim >= 1 and arr.shape[0] == len(Tx) and k_ not in ("R", "r_nodes", "kappas", "weights"):
                pre[nm][k_] = [arr[i] for i in range(keep)]
    for r in recs:
        for kk, c in (r.get("continuation") or {}).items():
            if not c.get("verified") and int(kk) not in pre["lz_state"]:
                pre["lz_state"][int(kk)] = r["T"] / rs
            if c.get("min_occupied_weight") is not None and c["min_occupied_weight"] < 0.5 and int(kk) not in pre["deg_state"]:
                pre["deg_state"][int(kk)] = r["T"] / rs
        if r.get("unconverged") and pre["unconv_first"] is None:
            pre["unconv_first"] = r["T"] / rs
    head = dict(J["head"])
    info34 = {"slice": a.resume_slice, "run": a.resume_run, "T_over_rs": Ts / rs, "record_index": ks, "slice_fields_vs_stored_max": fchk, "channels": chk}
    head["resume_slice_v34"] = info34
    print("version 34 (a): restarted from %s (T/rs %.4f, record %d) of %s: the slice's applied fields against the stored row %.1e; per channel the recomputed "
          "eigenvalues against the stored ones %s; the state propagated over the record's step; the run continues at T/rs %.4f" % (
              a.resume_slice, Ts / rs, ks, a.resume_run, fchk, ", ".join("%s: %.1e" % (k_, q["eig_vs_stored_max"]) for k_, q in chk.items()), (Ts + a.tau) / rs), flush=True)
    return st, (N_s.copy(), v_s.copy()), recs, fields, head, float(head["E0"]), np.asarray(Zf["N_cinit"]), np.asarray(Zf["v_cinit"]), ks + 1, Ts + a.tau, base, \
        {"from": a.resume_run, "T_over_rs": Ts / rs, "slice": a.resume_slice}, pre


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


# ------------------------------------------------------------------------------------------------------------------ version 33: S11
def fmt(x, f="%.3e"):
    """version 33: a number in format f, or "-" for None and NaN."""
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def chiral_seeds_v33(R, x0, sigma, b, k0):
    """P28 sec. 5.1's continuum chiral Slater seeds, re-derived: psi_j(x) = (sigma/pi)^(1/2) (sigma - i x)^j / (sigma + i x)^(j + 1),
    x = R - x0, j = 0 .. b - 1 (orthonormal on the line: x = sigma tan(theta) turns the Gram matrix into integrals of e^{2i(j-k)theta};
    Fourier support on k < 0, the ingoing half-line), times the inward boost e^{-i k0 x} (k0 >= 0, the inward momentum parameter: the
    support moves further into k < 0), on the radial chain's ingoing chiral spinor (1, -1)/sqrt(2): on it the sine bond gives h(k) =
    -sin k (positive for k < 0) and group velocity -cos k (inward), the two-step current sin(2k)/2 < 0 (J < 0). Returns (2n x b)."""
    x = np.asarray(R, float) - x0
    cols = np.zeros((2 * len(R), b), complex)
    for j in range(b):
        psi = math.sqrt(sigma / math.pi) * (sigma - 1j * x) ** j / (sigma + 1j * x) ** (j + 1) * np.exp(-1j * k0 * x)
        cols[0::2, j] = psi / math.sqrt(2.0); cols[1::2, j] = -psi / math.sqrt(2.0)
    return cols


def ball_orbitals_v33(spec):
    """'1:6,2:2' -> {1: 6, 2: 2}: the seed count per channel."""
    out = {}
    for item in spec.split(","):
        if item.strip():
            k_, b_ = item.split(":")
            out[int(k_)] = int(b_)
    return out


def ball_fixed_point_v33(chs, g, R, ell, a, k0, passes=200, tol=1e-10, log=None):
    """the ball's preparation at fixed k0: Psi per seeded channel, projected against the physical sea P_0 = 1(h_m(N, v) < 0) at the current
    fields, Phi_0 = (1 - P_0) Psi [Psi^dag (1 - P_0) Psi]^(-1/2) (P28 eq. 29; the Gram matrix's eigenvalues recorded, the rank test), the
    sources of the ball alone (the sea is P_0 = R2 at T = 0, so the pairs vanish there), one constraint solve, Anderson mixing of the fields
    until max |change| < tol. Returns (N, v, dust, info, diag)."""
    torch = BK["torch"]
    nsites = len(R)
    N = np.ones(nsites); v = np.zeros(nsites)
    mixer = MIX.Anderson()
    orbs = ball_orbitals_v33(a.ball_orbitals)
    x0 = a.ball_x0_over_R0 * a.ball_R0_over_rs * a.rs; sig = a.ball_sigma_over_R0 * a.ball_R0_over_rs * a.rs
    seeds = {k_: T_(chiral_seeds_v33(R, x0, sig, b_, k0)) for k_, b_ in orbs.items()}
    chm = {ch.kappa: ch for ch in chs}
    d = None; info = None; dust = {}; diag = {}
    for ip in range(passes):
        eps = np.zeros(nsites); J = np.zeros(nsites)
        diag = {"gram_min": [], "gram_max": [], "discarded_norm_max": 0.0, "zero_mode_min_abs_e": {}}
        dust = {}
        for kap, Psi in seeds.items():
            ch = chm[kap]
            e, U = eigT(clocked_gpu(ch, N, v))
            Up = U[:, e > 0]
            diag["zero_mode_min_abs_e"][str(kap)] = float(e.abs().min())
            proj = Up @ (Up.conj().T @ Psi)
            S = proj.conj().T @ proj
            sv, sU = torch.linalg.eigh(S)
            diag["gram_min"].append(float(sv.min())); diag["gram_max"].append(float(sv.max()))
            diag["discarded_norm_max"] = max(diag["discarded_norm_max"], float((1.0 - (proj.abs() ** 2).sum(dim=0) / (Psi.abs() ** 2).sum(dim=0)).max()))
            Phi = proj @ (sU * sv.clamp_min(1e-300).rsqrt()[None, :].to(sU.dtype)) @ sU.conj().T
            fk = np.ones(Phi.shape[1])
            eps += ch.w * dens(ch.h, Phi, fk); J += ch.w * dens(ch.Q, Phi, fk)
            dust[kap] = (Phi, fk)
            del e, U, Up
        Nn, vn, Wn, info = solve(eps, J, g, R, ell, N)
        if info.get("failed"):
            raise RuntimeError("massless ball: the solver failed at k0 %.6g pass %d: %s" % (k0, ip, info["failed"]))
        d = max(float(np.max(np.abs(Nn - N))), float(np.max(np.abs(vn - v))))
        x = mixer.step(np.concatenate([N, v]), np.concatenate([Nn, vn]))
        N, v = x[:nsites], x[nsites:]
        if d < tol:
            break
    # the final configuration: (1) the fields solved from the last sources; (2) the seeds projected at those fields and their sources
    # solved once more (the change against (1) printed as final_change); (3) the seeds projected at the fields of (2), which are the C-init
    # fields the run starts from, so the ball's columns are orthogonal to the run's initial sea exactly (their source change against (2)
    # printed as final_source_change). The returned info is the solve of (2).
    N, v, Wn, info = solve(eps, J, g, R, ell, N)

    def project_at(Nf, vf):
        eps_ = np.zeros(nsites); J_ = np.zeros(nsites); dust_ = {}; gr = {}
        for kap, Psi in seeds.items():
            ch = chm[kap]
            e, U = eigT(clocked_gpu(ch, Nf, vf))
            Up = U[:, e > 0]
            proj = Up @ (Up.conj().T @ Psi)
            S = proj.conj().T @ proj
            sv, sU = torch.linalg.eigh(S)
            Phi = proj @ (sU * sv.clamp_min(1e-300).rsqrt()[None, :].to(sU.dtype)) @ sU.conj().T
            fk = np.ones(Phi.shape[1])
            eps_ += ch.w * dens(ch.h, Phi, fk); J_ += ch.w * dens(ch.Q, Phi, fk)
            dust_[kap] = (Phi, fk)
            gr[kap] = [float(x_) for x_ in sv.cpu().numpy()]
            del e, U, Up
        return eps_, J_, dust_, gr
    eps, J, dust, gr = project_at(N, v)
    N2, v2, W2, info = solve(eps, J, g, R, ell, N)
    if info.get("failed"):
        raise RuntimeError("massless ball: the final solve failed at k0 %.6g: %s" % (k0, info["failed"]))
    diag["final_change"] = max(float(np.max(np.abs(N2 - N))), float(np.max(np.abs(v2 - v))))
    eps3, J3, dust, gr = project_at(N2, v2)
    diag["final_source_change"] = max(float(np.max(np.abs(eps3 - eps))), float(np.max(np.abs(J3 - J))))
    for kap, x_ in gr.items():
        diag["gram_final_%d" % kap] = x_
    diag["passes"] = ip + 1; diag["last_pass_change"] = d
    return N2, v2, dust, info, diag, eps3, J3


def prepare_massless_ball_v33(chs, g, R, ell, rs, a):
    """S11's preparation (docstring (b)): the k0 that brings the solved charge E (FLD.charge of the converged fixed point) to the target
    a.ball_E_target, by bisection on [0, a.ball_k0_max] after the bracket is checked (k0 = 0 below the target, k0_max above it); the charge
    is the constraint solve's, never the bare packet energy (P28 eq. 7: mu/E = 2 F(s)/s^2 for a thin layer). Returns (N, v, dust, info, head)."""
    if a.m != 0.0:
        raise RuntimeError("the massless ball is run at m a = 0 exactly (no mass regulator); got --m %g" % a.m)
    t0 = time.time()
    Et = a.ball_E_target
    hist = []

    def at(k0):
        out = ball_fixed_point_v33(chs, g, R, ell, a, k0, passes=a.ball_passes, tol=a.ball_tol)
        E = FLD.charge(out[3], g)
        hist.append({"k0": k0, "E": E, "passes": out[4]["passes"], "final_change": out[4]["final_change"], "gram_min": min(out[4]["gram_min"]) if out[4]["gram_min"] else None})
        print("massless ball: k0 %.8f -> E %.10f (target %.6f; passes %d, final change %.2e, Gram min %.3e)" % (k0, E, Et, out[4]["passes"], out[4]["final_change"],
              hist[-1]["gram_min"] if hist[-1]["gram_min"] is not None else float("nan")), flush=True)
        return E, out
    lo, hi = 0.0, a.ball_k0_max
    orbs = ball_orbitals_v33(a.ball_orbitals)
    reductions = []
    while True:
        E_lo, out_lo = at(lo)
        if E_lo < Et:
            break
        # the boost cannot be negative: the seed count is lowered by one in the channel holding the most seeds until the k0 = 0 charge lies
        # below the target (the builder's rule; every step printed and kept in the head)
        k_ = max(orbs, key=lambda q: orbs[q])
        orbs[k_] -= 1
        if orbs[k_] == 0:
            del orbs[k_]
        if not orbs:
            raise RuntimeError("massless ball: no seed count gives a k0 = 0 charge below the target %.6f" % Et)
        reductions.append({"from": a.ball_orbitals, "E_k0_0": E_lo})
        a.ball_orbitals = ",".join("%d:%d" % kv for kv in sorted(orbs.items()))
        print("massless ball: at k0 = 0 the solved charge %.10f is not below the target %.6f; the seeds lowered to %s" % (E_lo, Et, a.ball_orbitals), flush=True)
    E_hi, out_hi = at(hi)
    if E_hi < Et:
        raise RuntimeError("massless ball: at k0 = %g the solved charge %.6f is below the target %.6f: raise --ball-k0-max or the seed count" % (hi, E_hi, Et))
    out = out_hi; k0 = hi; E = E_hi
    for it in range(80):
        k0 = 0.5 * (lo + hi) if it % 2 == 0 or E_hi == E_lo else lo + (Et - E_lo) * (hi - lo) / (E_hi - E_lo)     # bisection and false position alternated
        k0 = min(max(k0, lo + 1e-12 * (hi - lo)), hi - 1e-12 * (hi - lo))
        E, out = at(k0)
        if abs(E - Et) < a.ball_E_tol * Et:
            break
        if E < Et:
            lo, E_lo = k0, E
        else:
            hi, E_hi = k0, E
    N, v, dust, info, diag, eps, J = out
    # diagnostics: per orbital the Killing energy, the bare source energy, the current ratio, the outgoing weight, the minimum site density
    torch = BK["torch"]
    chm = {ch.kappa: ch for ch in chs}
    orb = {}
    E_bare = 0.0
    pgrid = 2.0 * math.pi * np.fft.fftfreq(len(R))
    for kap, (Phi, fk) in dust.items():
        ch = chm[kap]
        hm = clocked_gpu(ch, N, v)
        ek = (Phi.conj() * (hm @ Phi)).sum(dim=0).real.cpu().numpy()
        rows = []
        for j in range(Phi.shape[1]):
            ej = dens(ch.h, Phi[:, j:j + 1]); Jj = dens(ch.Q, Phi[:, j:j + 1])
            P_np = Phi[:, j].cpu().numpy()
            c0 = np.fft.fft(P_np[0::2]); c1 = np.fft.fft(P_np[1::2])
            inw = 0.5 * np.abs(c0 - c1) ** 2                                  # the weight on the (1, -1)/sqrt(2) spinor per momentum
            tot = max(float((np.abs(c0) ** 2 + np.abs(c1) ** 2).sum()), 1e-300)
            rows.append({"j": j, "killing_energy": float(ek[j]), "bare_energy": float(ej.sum()), "current": float(Jj.sum()), "J_over_rho": float(Jj.sum() / ej.sum()) if ej.sum() != 0 else None,
                         "site_density_min": float(ej.min()), "site_current_max": float(Jj.max()), "weight_ingoing_spinor_k_negative": float(inw[np.sin(pgrid) < 0].sum()) / tot,
                         "weight_above_quarter_zone": float((np.abs(c0) ** 2 + np.abs(c1) ** 2)[np.abs(np.sin(pgrid)) > math.sin(math.pi / 4)].sum()) / tot})
            E_bare += ch.w * float(ej.sum())
        orb[str(kap)] = rows
    # the profile: the averaged density inside and outside R0, its minimum outside the central averaging cell, the mass function's maximum compactness
    rc = float(info["r_collar"]); rf = np.asarray(info["rfine"], float); rf = rf[(rf > 0) & (rf <= rc)]
    rho = np.asarray(info["rho_ell"](rf), float); Jl = np.asarray(info["J_ell"](rf), float); Wf = np.asarray(info["W_field"](rf), float)
    Nf = np.asarray(info["N_dense"](rf), float)
    R0 = a.ball_R0_over_rs * rs
    inside = (rf > ell) & (rf <= R0)
    comp = Wf / (np.maximum(rf, 1e-300) * np.maximum(Nf, 1e-300) ** 2)        # 2 g M / r = w^2 (M the Misner-Sharp mass; W = r v^2)
    mu4 = 4.0 * math.pi * rf ** 2
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    head = {"N_B": float(sum(Phi.shape[1] * chm[k_].w for k_, (Phi, _f) in dust.items())), "passes": diag["passes"], "last_change": diag["final_change"], "E0": FLD.charge(info, g),
            "member": "massless ball (m/T_H = 0), S11 pathfinder", "E_target": Et, "k0": k0, "k0_history": hist, "seed_reductions": reductions,
            "ball": {"R0": R0, "x0": a.ball_x0_over_R0 * R0, "sigma": a.ball_sigma_over_R0 * R0, "orbitals": a.ball_orbitals, "seeds": "P28 sec. 5.1 chiral Slater seeds with the inward boost e^{-i k0 x}",
                     "per_orbital": orb, "E_bare_weighted": E_bare, "E_bare_over_E": E_bare / FLD.charge(info, g),
                     "gram": {k_: v_ for k_, v_ in diag.items() if k_.startswith("gram")}, "discarded_norm_max": diag.get("discarded_norm_max"),
                     "zero_mode_min_abs_e": diag.get("zero_mode_min_abs_e"),
                     "rho_ell_min_inside_outside_cell": float(rho[inside].min()) if inside.any() else None, "J_ell_max_inside_outside_cell": float(Jl[inside].max()) if inside.any() else None,
                     "energy_fraction_beyond_R0": float(tz(mu4[rf > R0] * rho[rf > R0], rf[rf > R0])) / max(float(tz(mu4 * rho, rf)), 1e-300) if (rf > R0).sum() > 1 else None,
                     "max_compactness_2gM_over_r": float(comp.max()), "r_at_max_compactness": float(rf[int(np.argmax(comp))]),
                     "M_min_outside_cell": float((Wf / (2.0 * g * Nf ** 2))[rf > ell].min()) if (rf > ell).any() else None, "N_centre": float(Nf[0]),
                     "N_at_R0": float(np.interp(R0, rf, Nf)), "N_at_R0_thin_shell_control": 1.0 + math.sqrt(2.0 * g * FLD.charge(info, g) / R0)},
            "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
            "site_fields": "%s (%s, solve_constraints_ivp)" % (OPT["site_fields"], info.get("module")), "f_kin": None, "f_kin_bulk_frame": None,
            "dust_momentum_total": float(J.sum()), "projection": {"discarded_norm_max": diag.get("discarded_norm_max"), "gram_cond_max": None, "stationarity_max": None},
            "reading": "massless ball", "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"), "source_spline_error": info.get("source_spline_error"),
            "centre_series_ratio": info.get("centre_series_ratio"), "collar_margin": info.get("collar_margin"), "solver_iterations": info.get("it"), "dictionary": FLD.DICTIONARY,
            "seconds_preparation": time.time() - t0}
    gm = [x_ for k_, v_ in diag.items() if k_.startswith("gram_final") for x_ in v_]
    head["projection"]["gram_cond_max"] = (max(gm) / max(min(gm), 1e-300)) if gm else None
    head["projection"]["gram_min"] = min(gm) if gm else None
    return N, v, dust, info, head


def s11_smoothed(FLD_, eps_s, J_s, R, ell, info, grids):
    """the run's own smoothing (SmoothedSources5, the source kernels) applied to one sector's site sources, evaluated on each grid of grids;
    on a failure NaN arrays of the grids' lengths and the error's text."""
    try:
        src = FLD_.SmoothedSources5(np.asarray(eps_s, float), np.asarray(J_s, float), R, ell, float(info["r_nodes"][-1]), OPT["support_rel"])
        return [(np.asarray(src.rho(g_), float), np.asarray(src.J(g_), float)) for g_ in grids], None
    except Exception as ex:  # noqa: BLE001
        return [(np.full(len(g_), np.nan), np.full(len(g_), np.nan)) for g_ in grids], repr(ex)


def s11_record(a, info, R, rs, eps, J, eps_pairs, J_pairs, eps_dust, N_app, v_app, st, chs, E, cont_label, T):
    """S11 per record (docstring (c)): the averaged sources (total, the ball, the pairs) and the positive-branch reconstruction at the coarse
    nodes (arrays), the mass function's minimum outside the central averaging cell r > ell and the continued intervals, both constraint
    residuals, the ball's weight and energy beyond each collar; the stop flag (M(r) < 0 for some r > ell) and, when set, the three cases of
    P28 sec. 4 told apart. Returns (scalars, arrays)."""
    g = a.g; ell = a.ell
    rc = float(info["r_collar"]); rf = np.asarray(info["rfine"], float); rf = rf[(rf > 0) & (rf <= rc)]
    Wf = np.asarray(info["W_field"](rf), float); Nf = np.asarray(info["N_dense"](rf), float); vf = np.asarray(info["v_dense"](rf), float)
    rho = np.asarray(info["rho_ell"](rf), float); Jl = np.asarray(info["J_ell"](rf), float)
    Mf = Wf / (2.0 * g * np.maximum(Nf, 1e-300) ** 2)                   # the Misner-Sharp mass r w^2/(2g) (the field module's W = r v^2 = 2 g N^2 M)
    out_cell = rf > ell
    tolW = 1e-12 * max(float(np.max(np.abs(Wf))), 1e-300)
    neg = out_cell & (Wf < -tolW)
    w_ = np.where(np.abs(Nf) > 0, vf / Nf, 0.0)
    s_tot = rho + w_ * Jl
    J_dust = J - J_pairs
    nodes = np.asarray(info["r_nodes"], float)
    sm_d, err_d = s11_smoothed(FLD, eps_dust, J_dust, R, ell, info, [rf, nodes])
    sm_p, err_p = s11_smoothed(FLD, eps_pairs, J_pairs, R, ell, info, [rf, nodes])
    (rho_d, J_d), (rho_dn, J_dn) = sm_d
    (rho_p, J_p), (rho_pn, J_pn) = sm_p
    if err_d or err_p:
        rho_d = rho_p = None
    v_pos = -np.sqrt(np.maximum(Wf, 0.0) / np.maximum(rf, 1e-300))         # the positive branch: v = -(W/r)^(1/2), W = r v^2
    pos_ok = out_cell & (Wf > tolW)
    sc = {"T": T, "T_over_rs": T / rs, "E": E, "M_min_outside_cell": float(Mf[out_cell].min()) if out_cell.any() else None,
          "r_at_M_min": float(rf[out_cell][int(np.argmin(Mf[out_cell]))]) if out_cell.any() else None, "negative_points_outside_cell": int(neg.sum()),
          "U_phys_min_outside_cell": float(Wf[out_cell].min()) if out_cell.any() else None, "W_over_2g_min_outside_cell": float(Wf[out_cell].min()) / (2.0 * g) if out_cell.any() else None,
          "compactness_max_2gM_over_r": float(np.max(Wf / (np.maximum(rf, 1e-300) * np.maximum(Nf, 1e-300) ** 2))), "N_centre": float(Nf[0]),
          "positive_branch_v_residual_max": float(np.max(np.abs(vf[pos_ok] - v_pos[pos_ok]))) if pos_ok.any() else None,
          "source_min_outside_cell": float(s_tot[out_cell].min()) if out_cell.any() else None, "r_at_source_min": float(rf[out_cell][int(np.argmin(s_tot[out_cell]))]) if out_cell.any() else None,
          "rho_ell_min_outside_cell": float(rho[out_cell].min()) if out_cell.any() else None, "J_ell_max_outside_cell": float(Jl[out_cell].max()) if out_cell.any() else None,
          "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"),
          "continued_intervals": (info.get("gauge") or {}).get("intervals"), "empty_interior": {k_: v_ for k_, v_ in (info.get("empty_interior") or {}).items() if not isinstance(v_, np.ndarray)},
          "continuation_label": cont_label, "smoothing_errors": [err_d, err_p]}
    if rho_d is not None and rho_p is not None:
        sc["ball_source_min_outside_cell"] = float((rho_d + w_ * J_d)[out_cell].min()); sc["pairs_source_min_outside_cell"] = float((rho_p + w_ * J_p)[out_cell].min())
        sc["pairs_source_integral"] = float(np.sum(eps_pairs)); sc["ball_source_integral"] = float(np.sum(eps_dust))
    # the ball's own weight and energy beyond each collar (its leakage on the lattice branch)
    for Rc_rs in [float(x_) for x_ in a.s11_collars.split(",") if x_.strip()]:
        Rc = Rc_rs * rs
        i0 = int(np.searchsorted(np.asarray(R, float), Rc, side="right"))
        wsum = 0.0; ntot = 0.0
        for ch in chs:
            s_ = st[ch.kappa]
            if s_["dust"].shape[1]:
                col = s_["dust"]
                wb = (col[2 * i0:, :].abs() ** 2).sum(dim=0).cpu().numpy()
                wsum += ch.w * float(np.dot(np.asarray(s_["f"], float), wb)); ntot += ch.w * float(np.sum(s_["f"]))
        sel = np.asarray(R, float) > Rc
        sc["ball_beyond_%g" % Rc_rs] = {"weight": wsum, "weight_fraction": wsum / ntot if ntot > 0 else None,
                                        "energy_density_route": float(np.sum(N_app[sel] * eps_dust[sel] + v_app[sel] * J_dust[sel])),
                                        "pairs_energy_density_route": float(np.sum(N_app[sel] * eps_pairs[sel] + v_app[sel] * J_pairs[sel]))}
    # the stop and the three cases (P28 sec. 4)
    if neg.any():
        r_neg = float(rf[neg][0])
        inner = (rf > ell) & (rf <= r_neg)
        cases = {"r_first_negative_M": r_neg, "M_min": float(Mf[neg].min())}
        cases["averaged_source_min_inside_r_neg"] = float(s_tot[inner].min()) if inner.any() else None
        cases["rho_min_inside_r_neg"] = float(rho[inner].min()) if inner.any() else None
        cases["J_max_inside_r_neg"] = float(Jl[inner].max()) if inner.any() else None
        if rho_d is not None and rho_p is not None and inner.any():
            cases["ball_source_min_inside_r_neg"] = float((rho_d + w_ * J_d)[inner].min()); cases["pairs_source_min_inside_r_neg"] = float((rho_p + w_ * J_p)[inner].min())
        certified = (cont_label == "verified (Weyl)")
        if cases["averaged_source_min_inside_r_neg"] is not None and cases["averaged_source_min_inside_r_neg"] >= 0.0 and (cases["J_max_inside_r_neg"] or 0.0) <= 0.0:
            kind = "(i) numerical: the averaged sources inside the negative interval are nonnegative with an inward current, yet M < 0"
        elif certified:
            kind = "(ii) a source incompatible with the positive branch: the averaged physical-minus-reference source rho + w J is negative inside r_neg on a Weyl-certified reference"
        else:
            kind = "(iii) or (ii): the averaged source is negative inside r_neg and the reference transport is not Weyl-certified on this record (%s): a reference error cannot be excluded" % cont_label
        cases["case"] = kind; cases["reference_certified"] = certified
        sc["stop"] = cases
    arr = {"r_nodes": nodes, "rho_ell": np.asarray(info["rho_ell"](nodes), float), "J_ell": np.asarray(info["J_ell"](nodes), float),
           "W": np.asarray(info["W_field"](nodes), float), "N": np.asarray(info["N_dense"](nodes), float), "v": np.asarray(info["v_dense"](nodes), float)}
    arr["M"] = arr["W"] / (2.0 * g * np.maximum(arr["N"], 1e-300) ** 2)                     # the Misner-Sharp mass
    arr["v_positive_branch"] = -np.sqrt(np.maximum(arr["W"], 0.0) / np.maximum(nodes, 1e-300))
    arr.update({"rho_ell_ball": rho_dn, "J_ell_ball": J_dn, "rho_ell_pairs": rho_pn, "J_ell_pairs": J_pn})
    return sc, arr


def s11_audit_save(a, st, chs, cache, N_app, v_app, R, T, rs, eps, J, eps_pairs, J_pairs, info, sc):
    """at the stop record: the raw state (the sea's and the ball's columns, the continued reference R2), the raw and averaged sources, both
    residuals and the positive-branch reconstruction on the fine grid, as one complex128 npz (P28 sec. 4: what separates the three cases)."""
    path = os.path.join(a.out, "s11_audit_%s_T%07.3f.npz" % (a.tag, T / rs))
    arrs = {"T": T, "rs": rs, "R": R, "N_applied": N_app, "v_applied": v_app, "eps_tot": eps, "J_tot": J, "eps_pairs": eps_pairs, "J_pairs": J_pairs,
            "kappas": np.array([ch.kappa for ch in chs]), "weights": np.array([ch.w for ch in chs]), "stop": json.dumps(sc.get("stop"), default=float)}
    rc = float(info["r_collar"]); rf = np.asarray(info["rfine"], float); rf = rf[(rf > 0) & (rf <= rc)]
    arrs.update({"rfine": rf, "rho_ell": np.asarray(info["rho_ell"](rf), float), "J_ell": np.asarray(info["J_ell"](rf), float), "W": np.asarray(info["W_field"](rf), float),
                 "N_dense": np.asarray(info["N_dense"](rf), float), "v_dense": np.asarray(info["v_dense"](rf), float),
                 "residuals": np.array([info.get("residual_W") or np.nan, info.get("residual_invN") or np.nan], float)})
    for ch in chs:
        s_ = st[ch.kappa]; ref = cache[ch.kappa][3]
        arrs["sea_%d" % ch.kappa] = s_["sea"].cpu().numpy(); arrs["dust_%d" % ch.kappa] = s_["dust"].cpu().numpy(); arrs["f_%d" % ch.kappa] = np.asarray(s_["f"], float)
        arrs["R2_%d" % ch.kappa] = ref.cpu().numpy()
    np.savez(path, **arrs)
    return path


def v33_on(a):
    return bool(a.massless_ball or a.fields_v17)


def v33_desc(a):
    parts = []
    if a.massless_ball:
        parts.append("massless ball (R0 %g r_s, x0 %g R0, sigma %g R0, orbitals %s, E target %g; the S11 readouts and stop %s; collars %s r_s)" % (
            a.ball_R0_over_rs, a.ball_x0_over_R0, a.ball_sigma_over_R0, a.ball_orbitals, a.ball_E_target, "on" if a.s11_stop else "off", a.s11_collars))
    if a.fields_v17:
        parts.append("engine_fields_v17 (N' = 0 on an empty interior)")
    return "; ".join(parts)


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)
    label = ("quasi-static (tau = %g); ell = %d; reference: %s (%s); site fields: %s; engine_fields_v16 (one outward initial-value problem, the momentum "
             "equation kept everywhere, the flat-slice gauge continued and labelled); economy: lagged first iterate, stop |dE/E| < %g and fields < %g, "
             "%s iteration, at most %d iterations (the smallest-residual iterate returned, flagged); tau = %g; collar policy %s; %s") % (
        a.tau, a.ell, a.reference, ("the continued sea, the Landau-Zener-inspired maximum-overlap convention (c_LZ = %g; ops v3: one step, no window)" % a.c_LZ) if a.reference == "R2" else "the propagated sea",
        a.site_fields, a.stop_E, a.stop_fields, a.mix, a.maxit, a.tau, OPT["collar_policy"], DICT24)
    v32 = v32_on(a)
    if v32:
        label += "; v32: " + v32_desc(a)
    if v33_on(a):
        label += "; v33: " + v33_desc(a)
    if v34_on(a):
        label += "; v34: " + v34_desc(a)
    rs = a.rs
    NB = rs / (2.0 * a.g * a.m) if a.m > 0 else None                       # version 33 (a): no rest number for the massless ball
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
    elif a.resume_slice:
        st, f_ref, recs, fields, head, E0, N_cinit, v_cinit, step0, T, base_resume34, resume, pre34 = resume_slice_v34(a, chs, R)
        head["label_v34"] = label
    else:
        prep_ell = a.prep_ell if a.prep_ell is not None else a.ell
        if a.massless_ball:
            N, v, dust, info, head = prepare_massless_ball_v33(chs, a.g, R, prep_ell, rs, a)
            for k_ in ("f_kin", "f_kin_bulk_frame"):
                head[k_] = float("nan") if head.get(k_) is None else head[k_]
            for k_ in ("discarded_norm_max", "gram_cond_max", "stationarity_max"):
                head["projection"][k_] = float("nan") if head["projection"].get(k_) is None else head["projection"][k_]
        else:
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
        if a.massless_ball:
            b_ = head["ball"]
            print("massless ball (S11): R0 %.2f, x0 %.2f, sigma %.2f, orbitals %s, k0 %.10f; E0 %.10f (target %.6f); the bare energy (weighted) %.6f = %.6f E; Gram eigenvalues %s; discarded norm max %.3e; smallest |e| %s; rho_ell min over (ell, R0] %.4e, J_ell max there %.4e; energy fraction beyond R0 %s; max 2gM/r %.4f at r %.2f; M min outside the cell %s; per orbital (Killing energy, bare energy, J/rho, site density min, weight on the ingoing spinor at k < 0, weight above the quarter zone): %s; preparation %.0f s" % (
                b_["R0"], b_["x0"], b_["sigma"], b_["orbitals"], head["k0"], head["E0"], head["E_target"], b_["E_bare_weighted"], b_["E_bare_over_E"],
                json.dumps(b_["gram"], default=float), b_["discarded_norm_max"] if b_["discarded_norm_max"] is not None else float("nan"), json.dumps(b_["zero_mode_min_abs_e"]),
                b_["rho_ell_min_inside_outside_cell"] if b_["rho_ell_min_inside_outside_cell"] is not None else float("nan"),
                b_["J_ell_max_inside_outside_cell"] if b_["J_ell_max_inside_outside_cell"] is not None else float("nan"), b_["energy_fraction_beyond_R0"],
                b_["max_compactness_2gM_over_r"], b_["r_at_max_compactness"], b_["M_min_outside_cell"],
                "; ".join("kappa %s j %d (%.5f, %.5f, %.4f, %.2e, %.6f, %.2e)" % (k_, q["j"], q["killing_energy"], q["bare_energy"], q["J_over_rho"] if q["J_over_rho"] is not None else float("nan"),
                                                                              q["site_density_min"], q["weight_ingoing_spinor_k_negative"], q["weight_above_quarter_zone"])
                          for k_, rows_ in b_["per_orbital"].items() for q in rows_), head["seconds_preparation"]), flush=True)
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
    solver_meta = {"rtol": 1e-12, "atol": 1e-16, "collar_policy": OPT["collar_policy"], "support_rel": OPT["support_rel"], "module": FLD.__name__,
                   "solver": "solve_constraints_ivp", "site_fields": a.site_fields, "engine": "engine_collapse_v25", "stop_E": a.stop_E, "stop_fields": a.stop_fields,
                   "mix": a.mix, "mix_alpha": (a.mix_alpha if a.mix == "damped" else None), "maxit": a.maxit, "c_LZ": a.c_LZ, "window_max": a.window_max, "delta_rtol_every": a.delta_rtol_every, "streams": int(a.streams),
                   "resumed": resume is not None}
    if v32:
        solver_meta["v32"] = v32_options(a)
    end32 = None
    base_last = None
    hand = {"T": [], "N2avg": [], "eps_sea": [], "eps_R2": [], "eps_dust": [], "S_r_ell": [], "S_t_ell": []}
    for part in ("dust", "pairs"):
        for q in ("F_face", "fc", "b", "sy_k"):
            hand["%s_%s" % (q, part)] = []
    spectra = {"T": []}
    for ch in chs:
        spectra["mask_%d" % ch.kappa] = []; spectra["e_%d" % ch.kappa] = []

    s11_store = {}
    end33 = None
    end34 = None
    if a.resume_slice:
        base_last = base_resume34
        for k_, v_ in pre34["hand"].items():
            hand[k_] = list(v_)
        for k_, v_ in pre34["spectra"].items():
            spectra[k_] = list(v_)
        for k_, v_ in pre34["s11"].items():
            s11_store[k_] = list(v_)
        lz_state.update(pre34["lz_state"]); deg_state.update(pre34["deg_state"]); unconv_state["first"] = pre34["unconv_first"]

    def save_fields():
        if s11_store:
            np.savez_compressed(os.path.join(a.out, "collapse_%s_s11.npz" % a.tag), **{k_: np.array(v_) for k_, v_ in s11_store.items()})
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

        def sources_with(base_, cache_):
            """version 32: sources(cache_) with the base densities base_ (stage 1 of the continuation: the midpoint covariance's)."""
            t_d = time.time()
            eps = np.zeros(a.nsites); J = np.zeros(a.nsites); eps_pairs = np.zeros(a.nsites); eps_dust = np.zeros(a.nsites); J_pairs = np.zeros(a.nsites)
            for ch in chs:
                ref = cache_[ch.kappa][3]
                hs_, Qs_, dd, Jd = base_[ch.kappa]
                ds = hs_ - dens(ch.h, ref); Js = Qs_ - dens(ch.Q, ref)
                eps += ch.w * (ds + dd); J += ch.w * (Js + Jd); eps_pairs += ch.w * ds; eps_dust += ch.w * dd; J_pairs += ch.w * Js
            TM["densities_s"] += time.time() - t_d
            return eps, J, eps_pairs, eps_dust, J_pairs

        PAIR["R"] = R   # version 36b: the grid the workers solve on (the same R do_solve passes)

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
        solve32 = None
        if a.mix == "newton" and not info.get("failed") and v32:
            k_it, converged, best, cache, srcs, Nn, vn, Wn, info, N_app, v_app, solve32 = newton_record_v32(
                a, chs, st, cache, sources_with, base, (base_last if (step > step0 or a.resume_slice) else None), ((f_ref[0].copy(), f_ref[1].copy()) if (step > step0 or a.resume_slice) else None),
                do_solve, Nn, vn, E_prev, it_hist, T)
        elif a.mix == "newton" and not info.get("failed"):
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
                    ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_app, v_app), e, U, s_["n_sea"])
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
                if cinfo.get("certified") is not None:
                    cont[str(ch.kappa)]["certified"] = cinfo["certified"]
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
        if NSR["on"]:                                                   # version 36c: the record's set against the previous record's
            sig_rec = set_sig_v36c(masks)[0]
            NSR["rec"] = {"nk_set": "resolve", "record_set": sig_rec, "previous_record_set": NSR["prev_sig"],
                          "equals_previous_record": (sig_rec == NSR["prev_sig"]) if NSR["prev_sig"] is not None else None}
            NSR["prev_sig"] = sig_rec
            NSR["prev_m"] = set_sig_v36c(masks)[2]                      # version 36d: the previous record's mask bits
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
        oracle34 = None
        if getattr(a, "weyl_oracle", False) and a.reference == "R2":
            oracle34 = {}
            t_o = time.time()
            for ch in chs:
                _hm, e_o, U_o, ref_o, _ci = cache[ch.kappa]
                oracle34[str(ch.kappa)] = CERT.oracle_v34(ch, st[ch.kappa]["window"], (N_app, v_app), e_o, U_o, st[ch.kappa]["n_sea"], a.tau, a.c_LZ, eig_at,
                                                          clocked_host_sparse, ref_o, masks[ch.kappa], a.weyl_refine_max)
            TM["oracle34_s"] = time.time() - t_o
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
        s11 = None
        if a.massless_ball:
            t_11 = time.time()
            s11, s11_arr = s11_record(a, info, R, rs, eps, J, eps_pairs, J_pairs, eps_dust, N_app, v_app, st, chs, E, cont_label, T)
            s11["seconds"] = time.time() - t_11
            s11_store.setdefault("T", []).append(T)
            for k_, v_ in s11_arr.items():
                s11_store.setdefault(k_, []).append(v_)
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
               "solver": {"module": FLD.__name__ + ".solve_constraints_ivp", "nfev": (info.get("ivp") or {}).get("nfev"), "N_centre": (info.get("ivp") or {}).get("N_centre"),
                          "residual_W": info.get("residual_W"), "residual_invN": info.get("residual_invN"), "residual_matching_region": info.get("residual_matching_region"),
                          "residual_ladder": info.get("residual_ladder"), "source_spline_error": info.get("source_spline_error"), "centre_series_ratio": info.get("centre_series_ratio"),
                          "centre_series_check": info.get("centre_series_check"), "collar_margin": info.get("collar_margin"), "W_tail": info.get("W_tail"),
                          "W_roundoff_negative_points": info.get("W_roundoff_negative_points"), "W_negative": info.get("W_negative"), "site_fields": a.site_fields, "kernel_norms": info.get("kernel_norms")},
               "spectrum": info["spectrum"], "r_collar": info["r_collar"], "interp_floor_N": info.get("interp_floor_N"), "interp_floor_v": info.get("interp_floor_v"),
               "label": label, "seconds": time.time() - tr, "indicator": ind,
               "record_label": (FLD.__name__ + " (rtol %g, atol %g); v31 (v30 with the Newton-Krylov solve as an option; mixer %s%s; ops v3);" % (SOLVE_TOL["rtol"], SOLVE_TOL["atol"], a.mix, (" a = %g" % a.mix_alpha) if a.mix == "damped" else "") if not v32 else
                                FLD.__name__ + " (rtol %g, atol %g); v32 (v31 with %s; mixer %s; ops v3);" % (SOLVE_TOL["rtol"], SOLVE_TOL["atol"], v32_desc(a), a.mix)) + " kernel: the Gaussian of standard deviation ell, sigma^2 = ell^2 (plus a^2/12 where the cell average enters), evolved at ell %d, prepared at ell %d; tau = %g; continuation: %s; stresses: the lane's declared stress (the Noether flux of Q at unit clock and zero flow); rows: flat-slice ADM, K^ADM = -K, raw, floor in the stress's trace (25.3); J_ell = <P_r> = -J_r of the goal's display; gauge: %s; %s; %s; collar: %s" % (
                   a.ell, (a.prep_ell if a.prep_ell is not None else a.ell), a.tau, cont_label.replace("LZ convention (c_LZ = %g)" % a.c_LZ, "the LZ-inspired maximum-overlap convention (c_LZ = %g, the lane's declaration)" % a.c_LZ), gauge_label, unconv_label, ind_label, ind["collar_label"]),
               "timing": dict(TM),
               "M1_parts": {"X1": X1, "X2": X2, "Y1": Y1, "Y2": Y2, "I_E_end_prev_interval": IEend, "I_E_start": IE, "cut": cut, "I_E_mid": IE_mid}}
        if resume is not None and not any(q.get("engine") in ("engine_collapse_v19", "engine_collapse_v24", "engine_collapse_v25") for q in recs):
            rec["resumed_from"] = {"run": resume["from"], "T_over_rs": resume["T_over_rs"]}
        rec["engine"] = "engine_collapse_v25"
        if solve32 is not None:
            rec["solve_v32"] = solve32
        if s11 is not None:
            rec["s11"] = s11
        if v33_on(a):
            rec["engine_v33"] = v33_desc(a)
        if v34_on(a):
            rec["engine_v34"] = v34_desc(a)
        if getattr(a, "nk_jvp", None) == "exact":
            rec["jvp_v35"] = {k_: v_ for k_, v_ in JV.items() if not k_.startswith("_")}
            rec["engine_v35"] = "the Newton direction by the exact J.v (--nk-jvp exact; engine spec 40.4 (2) (b))"
            for k_ in JV:
                JV[k_] = None if k_ == "ref_mismatch" else (0.0 if k_ == "seconds" else 0)
        if PAIR["n"]:
            rec["v36b"] = {"pair_workers": PAIR["n"], "products_paired": PAIR["products"], "seconds": PAIR["seconds"], "serial_fallbacks": PAIR["fallbacks"],
                           "worker_kcache_hits": PAIR["hits"], "worker_kcache_misses": PAIR["misses"], "errors": PAIR["errors"][-3:]}
            PAIR.update({"products": 0, "seconds": 0.0, "fallbacks": 0, "hits": 0, "misses": 0, "errors": []})   # per record, as JV
        if NSR["on"] and NSR["rec"] is not None:
            rec["v36c"] = NSR["rec"]
            labs = [q.get("v36c", {}).get("label") for q in (solve32 or {}).get("solves", [])]
            if any(labs):
                rec["v36c"]["label"] = "two self-consistent sets"
            NSR["rec"] = None
        if NSR["on"] and NSR["other"] is not None:                       # version 36d: the other self-consistent set's solve beside the record
            o_ = NSR["other"]; NSR["other"] = None
            p_ = os.path.join(a.out, "collapse_%s_twosets_T%07.3f.npz" % (a.tag, T / rs))
            np.savez_compressed(p_, T=T, N=o_["N"], v=o_["v"], N_applied=o_["N_applied"], v_applied=o_["v_applied"], eps=o_["eps"], J=o_["J"],
                                E=o_["meta"]["E"], kept_residual=o_["meta"]["kept_residual"])
            rec["v36d"] = dict(o_["meta"], fields_file=os.path.basename(p_))
        if V36["fixes"] or V36["kcache"]:
            rec["v36"] = {"fixes": V36["fixes"], "gap_none_substeps": V36["gap_none_substeps"], "kernel_cache": dict(V36["kcache"]) if V36["kcache"] else None}
        if oracle34 is not None:
            rec["certificate_oracle"] = oracle34
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
        _line = ("T/rs %.3f E/E0 %.7f dust %.4f pairs %.3e R_t %s (2gE %.3f; d1..d4 of N+v over exact %s; r_in %s) min(N-|v|) %.4f R_sh %s R_c %s pockets %s inversions %d intermediate %d mask=neg %s nsub max %d it %d (eigh %d)%s M1 %.3e dE/dT %s res %.1e %.1e; B_estimate %s; r0 check dE %s; gauge %s; regime max %.2e; continuation [%s]; weights min occ %s max empty %s; tau_LZ min %s; [h,R2] max %s; stress identity %.1e %.1e virial %.1e %.1e trace %.1e; M1 balance: evolution %s work %s reset %s closure %s [%s] last iteration dE %s dN %.1e dv %.1e (%.0f s; eigh %d in %.1f s, continuation %.1f s, densities %.1f s, solves %d in %.1f s, indicator %.1f s, handoff %.1f s, M1 %.1f s, propagation %.1f s)" % (
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
            TM["densities_s"], TM["solve_n"], TM["solve_s"], TM["indicator_s"], TM.get("handoff_s", 0.0), TM["M1_s"], TM["propagation_prev_s"]))
        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        if s11 is not None:
            print("S11 T/rs %.4f: M min outside the cell %s at r %s (negative points %d); U_phys min %s; positive-branch v residual %s; rho + wJ min outside the cell %s at r %s (ball %s, pairs %s); residuals W %s 1/N %s; continued %s; %s; %s (%.1f s)" % (
                T / rs, fmt(s11["M_min_outside_cell"], "%+.6e"), fmt(s11["r_at_M_min"], "%.3f"), s11["negative_points_outside_cell"], fmt(s11["U_phys_min_outside_cell"], "%+.4e"),
                fmt(s11["positive_branch_v_residual_max"], "%.2e"), fmt(s11["source_min_outside_cell"], "%+.4e"), fmt(s11["r_at_source_min"], "%.3f"),
                fmt(s11.get("ball_source_min_outside_cell"), "%+.4e"), fmt(s11.get("pairs_source_min_outside_cell"), "%+.4e"), fmt(s11["residual_W"], "%.1e"), fmt(s11["residual_invN"], "%.1e"),
                s11["continued_intervals"], "; ".join("ball beyond %s r_s: weight %s (fraction %s), energy %s, pairs energy %s" % (
                    k_.replace("ball_beyond_", ""), fmt(v_["weight"], "%.4e"), fmt(v_["weight_fraction"], "%.4e"), fmt(v_["energy_density_route"], "%+.4e"), fmt(v_["pairs_energy_density_route"], "%+.4e"))
                    for k_, v_ in s11.items() if k_.startswith("ball_beyond_")), ("STOP: " + s11["stop"]["case"]) if s11.get("stop") else "no stop", s11["seconds"]), flush=True)
        if step > step0 and (step - step0) % 10 == 0:
            last = [q["seconds"] for q in recs[-10:] if "seconds" in q]
            mean = float(np.mean(last)) if last else 0.0
            remain = (nsteps - step) * mean
            end = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + remain))
            print("projection at T/rs %.2f: %.0f s per re-solve over the last 10; %d re-solves left; the end about %s UTC" % (T / rs, mean, nsteps - step, end), flush=True)
        if step % 5 == 0 or step == nsteps or step == step0:
            with open(path, "w") as fh:
                json.dump({"script": "engine_collapse_v34.py", "args": vars(a), "head": head, "records": recs}, fh, default=float)
            save_fields()
        if a.segment_end_T_over_rs is not None and T / rs >= a.segment_end_T_over_rs - 1e-9:
            end34 = {"T": T, "T_over_rs": T / rs, "segment_end_T_over_rs": a.segment_end_T_over_rs, "rule": "version 34 (b): the segment ends after this record"}
            print("T/rs %.4f: version 34 (b): the segment ends after this record" % (T / rs), flush=True)
            break
        if solve32 is not None and a.end_above is not None and solve32["kept_residual"] > a.end_above:
            # engine spec 32.1 (3): the member ends at this record (written above and again below); nothing is propagated from it
            solve32["ends_member"] = True
            end32 = {"T": T, "T_over_rs": T / rs, "kept_residual": solve32["kept_residual"], "end_above": a.end_above, "kept_from": solve32["kept_from"],
                     "continuation": solve32["continuation"],
                     "rule": "engine spec 32.1 (3): a record whose kept residual exceeds the bound after the continuation ends the member at that record; no record is propagated from an uncertified state"}
            print("T/rs %.3f: the member ends at this record (engine spec 32.1 (3)): the kept residual %.2e (%s) exceeds %g after the source continuation (%s); the record is written and no record is propagated from an uncertified state" % (
                T / rs, solve32["kept_residual"], solve32["kept_from"], a.end_above, solve32["continuation"]), flush=True)
            break
        if s11 is not None and s11.get("stop") and a.s11_stop:
            audit = s11_audit_save(a, st, chs, cache, N_app, v_app, R, T, rs, eps, J, eps_pairs, J_pairs, info, s11)
            end33 = {"T": T, "T_over_rs": T / rs, "stop": s11["stop"], "audit": audit,
                     "rule": "S11: the record stopped where M(r) < 0 outside the central averaging cell (r > ell); the three cases of P28 sec. 4 told apart; nothing propagated from it"}
            print("T/rs %.4f: S11 STOP: M(r) < 0 outside the central averaging cell from r %.4f (M min %.6e): %s; the audit state %s; the record is written and nothing is propagated from it" % (
                T / rs, s11["stop"]["r_first_negative_M"], s11["stop"]["M_min"], s11["stop"]["case"], audit), flush=True)
            with open(path, "w") as fh:
                json.dump({"script": "engine_collapse_v34.py", "args": vars(a), "head": head, "records": recs, "end_s11": end33}, fh, default=float)
            save_fields()
            break
        if a.benchmark or step >= nsteps:
            break
        if v32:
            base_last = base
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
        json.dump(dict({"script": "engine_collapse_v34.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)
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
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
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


V32_KEYS = ("nk_line_search", "nk_halvings", "nk_krylov", "nk_restarts", "source_continuation", "end_above")


def v32_on(a):
    """version 32: any of its options set (each None by default, so that the defaults are version 31's)."""
    return any(getattr(a, k, None) is not None for k in V32_KEYS)


def nk_halvings_eff(a):
    return int(a.nk_halvings) if a.nk_halvings is not None else 6


def v32_own_loop(a):
    """(a) or (b) on: the Newton solves run nk_loop_v32; otherwise scipy's newton_krylov as version 31 calls it."""
    return a.nk_line_search is not None or a.nk_krylov is not None or a.nk_restarts is not None


def v32_options(a):
    """the effective version 32 options (the solver entry of the fields file and every record's solve_v32)."""
    return {"line_search": (("halving on the fixed-point residual, at most %d halvings" % nk_halvings_eff(a)) if a.nk_line_search == "halving"
                            else "scipy's Armijo search on |F|_2^2 (version 31)"),
            "newton_loop": "nk_loop_v32" if v32_own_loop(a) else "scipy newton_krylov (version 31's call)",
            "newton_steps": int(a.nk_maxiter),
            "krylov_vectors_per_cycle": int(a.nk_krylov) if a.nk_krylov is not None else 30,
            "krylov_restarts": int(a.nk_restarts) if a.nk_restarts is not None else 0,
            "augmentation_vectors": 10,
            "source_continuation": a.source_continuation or "off",
            "end_above": a.end_above}


def v32_desc(a):
    """version 32's options in words (the run label and the record label)."""
    o = v32_options(a)
    return ("the Newton solve %s, line search: %s, at most %d Newton steps, LGMRES %d Krylov vectors per cycle with %d restarts (10 augmentation vectors); "
            "source continuation: %s; the member ends at a record whose kept residual exceeds %s") % (
        o["newton_loop"], o["line_search"], o["newton_steps"], o["krylov_vectors_per_cycle"], o["krylov_restarts"],
        ("two-stage (the midpoint covariance from the previous record's accepted fields, then this record's)" if o["source_continuation"] == "two-stage" else "off"),
        ("%g" % o["end_above"]) if o["end_above"] is not None else "nothing (off)")


def nk_loop_v32(a, F, x0):
    """(a) and (b): scipy 1.14.1's nonlin_solve loop for newton_krylov (the Krylov Jacobian with rdiff None, LGMRES, 10 augmentation vectors
    carried across the Newton steps; the forcing terms gamma 0.9, eta_max 0.9999, eta_treshold 0.1, eta 1e-3 at the start; the inner
    tolerance min(eta, eta |F|_2)), with LGMRES's Krylov vectors per cycle (--nk-krylov) and restarts (--nk-restarts) when set, the stop
    max|F| <= --stop-fields tested before every Newton step (the first included: scipy's test cannot pass before its first step, its initial
    step norm being infinite) and after the last, and the line search either scipy's Armijo search on |F|_2^2 or, --nk-line-search halving,
    the backtracking search on the fixed-point residual max|F| along the Newton direction: the steps 1, 1/2, ..., 2^-H in turn, the first
    whose residual falls below the current point's taken; when none falls the solve ends there. Returns (status, the line search's steps)."""
    from numpy.linalg import norm
    from scipy.optimize._nonlin import KrylovJacobian, _nonlin_line_search
    kw = {}
    if a.nk_krylov is not None:
        kw["inner_inner_m"] = int(a.nk_krylov)
    if getattr(a, "nk_jvp", None) == "exact" and getattr(F, "jvp", None) is not None:
        class _ExactKrylov(KrylovJacobian):                   # version 35: J.v exact at the iterate (the docstring)
            def matvec(self, v_):
                return F.jvp(self.x0, np.asarray(v_, dtype=float))
        jac = _ExactKrylov(rdiff=None, method="lgmres", inner_maxiter=a.nk_inner, outer_k=10, **kw)
    else:
        jac = KrylovJacobian(rdiff=None, method="lgmres", inner_maxiter=a.nk_inner, outer_k=10, **kw)
    if a.nk_restarts is not None:
        jac.method_kw["maxiter"] = 1 + int(a.nk_restarts)
    H = nk_halvings_eff(a)
    x = np.asarray(x0, dtype=float).flatten().copy()
    Fx = np.asarray(F(x), dtype=float).flatten()
    Fx_norm = norm(Fx)
    jac.setup(x.copy(), Fx, F)
    gamma, eta_max, eta_treshold, eta = 0.9, 0.9999, 0.1, 1e-3
    steps = []
    for n_ in range(a.nk_maxiter):
        r0 = float(np.max(np.abs(Fx)))
        if r0 <= a.stop_fields:
            return "converged", steps
        tol = min(eta, eta * Fx_norm)
        dx = -jac.solve(Fx, tol=tol)
        if norm(dx) == 0:
            return "stopped: the Krylov solve returned a zero direction at Newton step %d" % (n_ + 1), steps
        if a.nk_line_search == "halving":
            s, trials, acc = 1.0, [], None
            for h_ in range(H + 1):
                xt = x + s * dx
                Ft = np.asarray(F(xt), dtype=float).flatten()
                rt = float(np.max(np.abs(Ft)))
                trials.append([s, rt])
                if rt < r0:
                    acc = (s, xt, Ft)
                    break
                s *= 0.5
            steps.append({"newton_step": n_ + 1, "residual_before": r0, "trials": trials, "accepted_step": (acc[0] if acc is not None else None)})
            if acc is None:
                return "the residual did not fall in %d halvings at Newton step %d (the solve ends there)" % (H, n_ + 1), steps
            s, x, Fx = acc
            Fx_norm_new = norm(Fx)
        else:
            s, x, Fx, Fx_norm_new = _nonlin_line_search(F, x, Fx, dx, "armijo")
            Fx = np.asarray(Fx, dtype=float).flatten()
            steps.append({"newton_step": n_ + 1, "residual_before": r0, "accepted_step": float(s), "residual_after": float(np.max(np.abs(Fx)))})
        jac.update(x.copy(), Fx)
        eta_A = gamma * Fx_norm_new ** 2 / Fx_norm ** 2
        if gamma * eta ** 2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma * eta ** 2))
        Fx_norm = Fx_norm_new
    r_ = float(np.max(np.abs(Fx)))
    return ("converged" if r_ <= a.stop_fields else "no convergence within %d Newton steps" % a.nk_maxiter), steps


def confirm_v32(a, b, evalG, it_hist, ctr):
    """version 31's confirming plain step at the kept iterate's output (|dE/E| and a second residual), appended to the history."""
    Nn_b, vn_b, Wn_b, info_b = b["sol"]
    cache_c, srcs_c, sol_c = evalG(np.array(Nn_b, float), np.array(vn_b, float))
    ctr["n"] += 1
    dE_c, r_c = None, None
    if not sol_c[3].get("failed"):
        E_c = FLD.charge(sol_c[3], a.g)
        dE_c = abs(E_c - b["E"]) / abs(E_c)
        r_c = max(float(np.max(np.abs(sol_c[0] - Nn_b))), float(np.max(np.abs(sol_c[1] - vn_b))))
        it_hist.append([float(dE_c), float(np.max(np.abs(sol_c[0] - Nn_b))), float(np.max(np.abs(sol_c[1] - vn_b)))])
    del cache_c, srcs_c, sol_c
    conv = bool(b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E)
    return {"r_c": r_c, "dE_c": dE_c, "conv": conv}


def cross_lower_v36d(steps):
    """version 36d: True when a Newton step had a crossing trial (not taken) with a residual below every non-crossing trial of that step
    (flags_v36c_records_v1's CROSS-LOWER)."""
    for ls in steps:
        tr = ls.get("trials", [])
        acc = [t for t in tr if not t.get("crosses")]
        crs = [t for t in tr if t.get("crosses")]
        if acc and crs and min(t["residual"] for t in crs) < min(t["residual"] for t in acc):
            return True
    return False


def rounds_v36c(a, name, start, nk, RS, F, evalG, x0v, it_hist, ctr, t0, h0):
    """version 36c: the set-resolving solve (the docstring): rounds of nk_loop_v36c, at most 3, each from the previous round's far-side
    start with that set held; the round kept by the stop rule (the previous record's set preferred when two meet it). Returns
    nk_stage_v32's entry."""
    n_start = ctr["n"] - 0
    rounds = []
    xs = x0v
    v36d_extra = False
    for k_round in range(1, 4):
        nk["best"] = None; nk["failed"] = None
        RS.update({"held": None, "held_m": None, "far": None, "round": k_round, "by_halving": False})
        h_r, n_r, t_r, s_r = len(it_hist), ctr["n"], time.time(), len(RS["steps"])
        try:
            st_r = nk_loop_v36c(a, F, xs, RS)
        except Exception as ex:  # a failed solve inside the exact J.v, a non-finite residual
            st_r = "stopped: %s" % str(ex)[:120]
        ent = {"round": k_round, "held_set": (RS["sets"][RS["held"]]["id"] if RS["held"] is not None else None), "held_sig": RS["held"],
               "status": st_r, "best": nk["best"], "failed": nk["failed"], "r_c": None, "dE_c": None, "conv": False,
               "far_start": ({k_: v_ for k_, v_ in RS["far"].items() if k_ != "x"} if RS["far"] is not None else None),
               "steps": [s_r, len(RS["steps"])]}
        if nk["best"] is not None:
            ent.update(confirm_v32(a, nk["best"], evalG, it_hist, ctr))
        ent["evaluations"] = ctr["n"] - n_r; ent["seconds"] = time.time() - t_r; ent["history"] = [h_r, len(it_hist)]
        rounds.append(ent)
        print("  v36c %s round %d: held set %s; %s; %d evaluations; kept %s, confirming %s, |dE/E| %s; %s; far-side start %s" % (
            name, k_round, ent["held_set"], st_r, ent["evaluations"], ("%.2e" % nk["best"]["resid"]) if nk["best"] is not None else "none",
            ("%.2e" % ent["r_c"]) if ent["r_c"] is not None else "none", ("%.1e" % ent["dE_c"]) if ent["dE_c"] is not None else "none",
            "meets the stop rule" if ent["conv"] else "does not meet the stop rule",
            ("set %d at step %g of Newton step %d (residual %.2e)" % (RS["far"]["set_id"], RS["far"]["step"], RS["far"]["newton_step"], RS["far"]["residual"])) if RS["far"] is not None else "none"), flush=True)
        ent["cross_lower"] = cross_lower_v36d(RS["steps"][s_r:])            # version 36d
        if ent["conv"] and k_round == 1 and ent["cross_lower"] and RS["far"] is not None:
            ent["E_prev_after"] = ctr["E_prev"]
            print("  v36d %s: round 1 met the stop rule after refusing a CROSS-LOWER trial: a second round from its far-side start (set %d)" % (
                name, RS["far"]["set_id"]), flush=True)
            xs = RS["far"]["x"]; v36d_extra = True
            continue
        if v36d_extra or ent["conv"] or not RS["by_halving"] or RS["far"] is None:
            break
        xs = RS["far"]["x"]
    meet = [r for r in rounds if r["conv"]]
    label = None
    v36d = None
    for r in rounds:                                                    # version 36d: each round's kept set against the previous record's
        r["kept_gap"] = None; r["flips_vs_previous"] = None
        if r["best"] is not None:
            g_ = [q[4] for q in RS["evals"] if q[0] == r["best"]["k_it"]]
            r["kept_gap"] = g_[0] if g_ else None
            m_ = RS["sets"].get(r["held_sig"], {}).get("_m")
            if NSR["prev_m"] is not None and m_ is not None and m_.size == NSR["prev_m"].size:
                r["flips_vs_previous"] = int(np.sum(m_ != NSR["prev_m"]))
    if meet:
        if len(meet) > 1:
            known = all(r["flips_vs_previous"] is not None for r in meet)
            key = (lambda r: (r["flips_vs_previous"], -(r["kept_gap"] if r["kept_gap"] is not None else -1.0))) if known else (
                lambda r: (-(r["kept_gap"] if r["kept_gap"] is not None else -1.0),))
            chosen = min(meet, key=key)
            label = "two self-consistent sets"
            other = [r for r in meet if r is not chosen][0]
            v36d = {"rule": ("fewer flips from the previous record's set, ties by the larger smallest gap" if known else
                             "the previous record's set not known (first record after a restart): the larger smallest gap"),
                    "kept": {"round": chosen["round"], "set": chosen["held_set"], "flips_vs_previous": chosen["flips_vs_previous"], "kept_gap": chosen["kept_gap"],
                             "kept_residual": chosen["best"]["resid"], "confirming_residual": chosen["r_c"], "confirming_dE": chosen["dE_c"], "E": chosen["best"]["E"]},
                    "other": {"round": other["round"], "set": other["held_set"], "signature": other["held_sig"], "flips_vs_previous": other["flips_vs_previous"],
                              "kept_gap": other["kept_gap"], "kept_residual": other["best"]["resid"], "confirming_residual": other["r_c"],
                              "confirming_dE": other["dE_c"], "E": other["best"]["E"], "label": "two self-consistent sets"}}
            ob = other["best"]
            NSR["other"] = {"meta": v36d["other"], "N": np.asarray(ob["sol"][0], float), "v": np.asarray(ob["sol"][1], float),
                            "N_applied": np.asarray(ob["app"][0], float), "v_applied": np.asarray(ob["app"][1], float),
                            "eps": np.asarray(ob["srcs"][0], float), "J": np.asarray(ob["srcs"][1], float)}
            print("  v36d %s: two rounds meet the stop rule: kept round %d (set %s, %s flips from the previous record's, gap %s); other round %d (set %s, %s flips, gap %s) stored beside" % (
                name, chosen["round"], chosen["held_set"], chosen["flips_vs_previous"], chosen["kept_gap"], other["round"], other["held_set"],
                other["flips_vs_previous"], other["kept_gap"]), flush=True)
        else:
            chosen = meet[0]
    else:
        cands = [r for r in rounds if r["best"] is not None]
        chosen = min(cands, key=lambda r: r["best"]["resid"]) if cands else rounds[-1]
    if v36d_extra and chosen is rounds[0] and rounds[0].get("E_prev_after") is not None:
        ctr["E_prev"] = rounds[0]["E_prev_after"]                       # version 36d: round 1 kept after the second round
    for r in rounds:
        if r is not chosen and r["best"] is not None:
            r["best"] = {k: r["best"][k] for k in ("resid", "k_it", "app", "dE", "E")}
    nk["best"] = chosen["best"]; nk["failed"] = chosen["failed"]
    status = ("round %d of %d: " % (chosen["round"], len(rounds))) + chosen["status"]
    v36c = {"rounds": [{k: (v if k != "best" else ({"kept_residual": v["resid"], "kept_evaluation": v["k_it"]} if v is not None else None))
                        for k, v in r.items()} for r in rounds],
            "kept_round": chosen["round"], "kept_set": chosen["held_set"], "kept_sig": chosen["held_sig"], "label": label,
            "kept_set_equals_previous_record": (chosen["held_sig"] == NSR["prev_sig"]) if NSR["prev_sig"] is not None else None,
            "sets": [{k: v for k, v in s_.items() if k != "_m"} for s_ in RS["sets"].values()],
            "evaluation_sets": RS["evals"], "steps": RS["steps"]}
    out = {"name": name, "start": start, "status": status, "line_search": RS["steps"], "best": nk["best"], "failed": nk["failed"], "evalG": evalG,
           "r_c": chosen["r_c"], "dE_c": chosen["dE_c"], "conv": chosen["conv"], "v36c": v36c}
    if v36d is not None:
        out["v36d"] = v36d
    out["evaluations"] = ctr["n"] - n_start
    out["seconds"] = time.time() - t0
    out["history"] = [h0, len(it_hist)]
    if label:
        print("  v36c %s: TWO SELF-CONSISTENT SETS; kept round %d (set %s)" % (name, chosen["round"], chosen["held_set"]), flush=True)
    return out


def nk_stage_v32(a, chs, st, cache0, sources_fn, do_solve, x0, it_hist, T, name, start, ctr):
    """one Newton solve of a fixed point x -> G(x) (version 31's evaluation: the eigendecompositions at x, the continuation of R2 from the
    previous record, the sources by sources_fn, one field solve) from x0, then the confirming plain step; every evaluation enters it_hist
    and ctr counts the record's evaluations across the solves. Returns the solve's entry (the kept iterate under "best")."""
    n = a.nsites
    nk = {"best": None, "failed": None}
    n_start = ctr["n"]

    def evalG(N_a, v_a):
        eig = eigh_all(chs, N_a, v_a, bool(a.streams))
        cache_ = dict(cache0)
        for ch in chs:
            e, U = eig.pop(ch.kappa)
            if a.reference == "R2":
                t_c = time.time()
                s_ = st[ch.kappa]
                ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (N_a, v_a), e, U, s_["n_sea"])
                cinfo["_mask"] = mask
                TM["continuation_s"] += time.time() - t_c
                TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1
            else:
                ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}
            cache_[ch.kappa] = (None, e, U, ref, cinfo)
        del eig
        srcs_ = sources_fn(cache_)
        return cache_, srcs_, do_solve(srcs_)

    def F(x):
        N_a = np.array(x[:n], dtype=float); v_a = np.array(x[n:], dtype=float)
        cache_, srcs_, sol_ = evalG(N_a, v_a)
        Nn_, vn_, Wn_, info_ = sol_
        ctr["n"] += 1
        if jv_on:
            memo["x"] = np.asarray(x, dtype=float).copy(); memo["cache"] = cache_; memo["srcs"] = srcs_
        held_ok = True
        if NSR["on"]:                                                   # version 36c: the evaluation's set
            sig_, nocc_, mc_ = set_sig_v36c({k_: c_[4]["_mask"] for k_, c_ in cache_.items() if c_[1] is not None})
            if RS["held"] is None:
                RS["held"] = sig_; RS["held_m"] = mc_
            if sig_ not in RS["sets"]:
                RS["sets"][sig_] = {"id": len(RS["sets"]), "n_occupied": nocc_, "signature": sig_,
                                    "flips_vs_stage_start": (int(np.sum(mc_ != RS["first_m"])) if RS["first_m"] is not None and RS["first_m"].size == mc_.size else 0),
                                    "_m": mc_}
                if RS["first_m"] is None:
                    RS["first_m"] = mc_
            held_ok = sig_ == RS["held"]
            gap_ = min_gap_v36c(cache_, chs)
            RS["last"] = {"set_id": RS["sets"][sig_]["id"], "same": held_ok, "gap": gap_, "sig": sig_,
                          "flips": (int(np.sum(mc_ != RS["held_m"])) if RS["held_m"].size == mc_.size else -1)}
        if info_.get("failed"):
            nk["failed"] = info_["failed"]
            if NSR["on"]:
                RS["evals"].append([ctr["n"], RS["round"], RS["last"]["set_id"], None, RS["last"]["gap"]])
            return np.full(2 * n, np.inf if V36["fixes"] else 1.0)     # version 36 (c): a failed solve is rejected
        E_ = FLD.charge(info_, a.g)
        dN_ = float(np.max(np.abs(Nn_ - N_a))); dv_ = float(np.max(np.abs(vn_ - v_a)))
        dE_ = abs(E_ - ctr["E_prev"]) / abs(E_)
        if not V36["fixes"]:
            ctr["E_prev"] = E_
        elif held_ok and (nk["best"] is None or max(dN_, dv_) < nk["best"]["resid"]):
            ctr["E_prev"] = E_                                          # version 36 (b): moved at the accepted (best) iterate only
        it_hist.append([float(dE_), dN_, dv_])
        r_ = max(dN_, dv_)
        if NSR["on"]:
            RS["evals"].append([ctr["n"], RS["round"], RS["last"]["set_id"], r_, RS["last"]["gap"]])
        if held_ok and (nk["best"] is None or r_ < nk["best"]["resid"]):
            nk["best"] = {"resid": r_, "k_it": ctr["n"], "cache": cache_, "srcs": srcs_, "sol": sol_, "app": (N_a.copy(), v_a.copy()), "dE": dE_, "E": E_}
        return np.concatenate([Nn_ - N_a, vn_ - v_a])

    def jvp(x, dx):
        """version 35: the exact J.v at the iterate x (the docstring's --nk-jvp exact)."""
        torch = BK["torch"]
        t_j = time.time()
        if memo["x"] is None or not np.array_equal(memo["x"], np.asarray(x, dtype=float)):
            F(np.asarray(x, dtype=float))             # the iterate's own evaluation (counted as one), when the last one was elsewhere
            JV["extra_evaluations"] += 1
        cache_, srcs_ = memo["cache"], memo["srcs"]
        dN = np.asarray(dx[:n], dtype=float); dv = np.asarray(dx[n:], dtype=float)
        deps = np.zeros(n); dJ = np.zeros(n)
        for ch in chs:
            _, e, U, ref, cinfo = cache_[ch.kappa]
            mask = torch.as_tensor(np.asarray(cinfo["_mask"], bool), device=U.device)
            Uo = U[:, mask]; Uu = U[:, ~mask]
            eo = e[mask].to(torch.float64); eu = e[~mask].to(torch.float64)
            dh = dev_csr(clocked_host_sparse(ch, dN, dv))
            Hm = Uu.conj().T @ (dh @ Uo)
            den = eo[None, :] - eu[:, None]
            small = den.abs() < 1e-12
            JV["dropped_pairs"] += int(small.sum())
            M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))
            X = Uu @ M
            if JV["ref_mismatch"] is None or JV["_chk"] < len(chs):
                dg = (ref.conj() * ref).real.sum(dim=1) - (Uo.conj() * Uo).real.sum(dim=1)
                JV["ref_mismatch"] = max(JV["ref_mismatch"] or 0.0, float(dg.abs().max())); JV["_chk"] += 1
            for A_, acc in ((ch.h, deps), (ch.Q, dJ)):
                d_ = (Uo.conj() * (A_ @ X)).real.sum(dim=1) + (X.conj() * (A_ @ Uo)).real.sum(dim=1)
                acc -= ch.w * d_.reshape(-1, 2).sum(dim=1).cpu().numpy()
        eps0 = np.asarray(srcs_[0], float); J0 = np.asarray(srcs_[1], float)
        sc = max(float(np.max(np.abs(deps))), float(np.max(np.abs(dJ))))
        if sc == 0.0:
            out = -np.asarray(dx, dtype=float)
        else:
            t = 1e-4 * max(1.0, float(np.max(np.abs(eps0))), float(np.max(np.abs(J0)))) / sc
            if PAIR["n"]:
                sp_, sm_ = pair_solve(do_solve, [(eps0 + t * deps, J0 + t * dJ) + tuple(srcs_[2:]), (eps0 - t * deps, J0 - t * dJ) + tuple(srcs_[2:])], a.g, PAIR["R"], a.ell)
            else:
                sp_ = do_solve((eps0 + t * deps, J0 + t * dJ) + tuple(srcs_[2:]))
                sm_ = do_solve((eps0 - t * deps, J0 - t * dJ) + tuple(srcs_[2:]))
            if sp_[3].get("failed") or sm_[3].get("failed"):
                raise RuntimeError("version 35: a field solve failed inside the exact J.v")
            dNo = (np.asarray(sp_[0]) - np.asarray(sm_[0])) / (2.0 * t); dvo = (np.asarray(sp_[1]) - np.asarray(sm_[1])) / (2.0 * t)
            out = np.concatenate([dNo - dN, dvo - dv])
        JV["products"] += 1; JV["seconds"] += time.time() - t_j
        return out

    jv_on = getattr(a, "nk_jvp", None) == "exact"
    memo = {"x": None, "cache": None, "srcs": None}
    if jv_on:
        F.jvp = jvp
    t0 = time.time()
    h0 = len(it_hist)
    x0v = np.concatenate([np.asarray(x0[0], float), np.asarray(x0[1], float)])
    steps = []
    RS = {"held": None, "held_m": None, "first_m": None, "sets": {}, "evals": [], "steps": [], "far": None, "round": 0, "last": None, "by_halving": False}
    if NSR["on"]:
        return rounds_v36c(a, name, start, nk, RS, F, evalG, x0v, it_hist, ctr, t0, h0)
    try:
        if v32_own_loop(a):
            status, steps = nk_loop_v32(a, F, x0v)
        else:
            from scipy.optimize import newton_krylov, NoConvergence
            try:
                newton_krylov(F, x0v, f_tol=a.stop_fields, maxiter=a.nk_maxiter, method="lgmres", inner_maxiter=a.nk_inner, verbose=False)
                status = "converged"
            except NoConvergence:
                status = "no convergence within %d Newton steps" % a.nk_maxiter
    except Exception as ex:  # a failed solve inside a line search, a non-finite residual
        status = "stopped: %s" % str(ex)[:120]
    out = {"name": name, "start": start, "status": status, "line_search": steps, "best": nk["best"], "failed": nk["failed"], "evalG": evalG,
           "r_c": None, "dE_c": None, "conv": False}
    if nk["best"] is not None:
        out.update(confirm_v32(a, nk["best"], evalG, it_hist, ctr))
    out["evaluations"] = ctr["n"] - n_start
    out["seconds"] = time.time() - t0
    out["history"] = [h0, len(it_hist)]
    return out


def stage_entry_v32(s):
    """a solve's json entry (no arrays)."""
    b = s["best"]
    return {"name": s["name"], "start": s["start"], "status": s["status"], "evaluations": s["evaluations"], "seconds": s["seconds"],
            "kept_residual": (b["resid"] if b is not None else None), "kept_evaluation": (b["k_it"] if b is not None else None),
            "confirming_residual": s["r_c"], "confirming_dE": s["dE_c"], "meets_stop_rule": s["conv"], "failed_solve": s["failed"],
            "iteration_history_slice": s["history"], "line_search": s["line_search"], **({"v36c": s["v36c"]} if "v36c" in s else {})}


def print_stage_v32(a, T, s):
    b = s["best"]
    acc = [q.get("accepted_step") for q in s["line_search"]]
    print("T/rs %.3f newton (%s): %s; %d evaluations in %.0f s; kept residual %s (evaluation %s), the confirming plain step's residual %s and abs dE/E %s; %s; line search steps %s" % (
        T / a.rs, s["name"], s["status"], s["evaluations"], s["seconds"], ("%.2e" % b["resid"]) if b is not None else "none", (b["k_it"] if b is not None else "-"),
        ("%.2e" % s["r_c"]) if s["r_c"] is not None else "none", ("%.1e" % s["dE_c"]) if s["dE_c"] is not None else "none",
        "converged" if s["conv"] else "not converged", ("[" + ", ".join("%g" % q if q is not None else "none" for q in acc) + "]") if acc else "-"), flush=True)


def drop_arrays_v32(s):
    """free a solve's kept eigendecompositions once its iterate is not the record's (the fields kept)."""
    if s.get("best") is not None:
        s["best"] = {k: s["best"][k] for k in ("resid", "k_it", "app", "dE", "E")}


def newton_record_v32(a, chs, st, cache0, sources_with, base_new, base_prev, f_prev, do_solve, Nn0, vn0, E0, it_hist, T):
    """version 32's record solve (the docstring's (a) to (c)): the direct solve on this record's map from version 31's start; on a failure of
    the stop rule, the two-stage continuation in the propagated covariance; returns version 31's tuple and the record's solve_v32 entry."""
    ctr = {"n": 0, "E_prev": E0}
    src_new = lambda c_: sources_with(base_new, c_)
    solves = []
    d = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, (Nn0, vn0), it_hist, T, "direct", "the first solve's output (version 31's start)", ctr)
    solves.append(d)
    print_stage_v32(a, T, d)
    if d["best"] is None:
        e32 = {"options": v32_options(a), "solves": [stage_entry_v32(q) for q in solves], "continuation": "not run (the direct solve has no evaluation)",
               "kept_from": None, "kept_residual": None, "confirming_residual": None, "ends_member": False}
        return ctr["n"], False, None, cache0, None, Nn0, vn0, None, {"failed": d["failed"] or "no evaluation"}, Nn0, vn0, e32
    final = d
    if d["conv"]:
        cont = "not needed (the direct solve meets the stop rule)"
    elif a.source_continuation != "two-stage":
        cont = "off"
    elif base_prev is None or f_prev is None:
        cont = "not run: no previous record in this run"
    else:
        base_mid = {k: tuple(0.5 * (p_ + q_) for p_, q_ in zip(base_prev[k], base_new[k])) for k in base_new}
        s1 = nk_stage_v32(a, chs, st, cache0, (lambda c_: sources_with(base_mid, c_)), do_solve, f_prev, it_hist, T,
                          "stage 1, the midpoint covariance", "the previous record's accepted applied fields", ctr)
        solves.append(s1)
        print_stage_v32(a, T, s1)
        if s1["best"] is None:
            cont = "ran; stage 1 has no evaluation (%s); kept from the direct solve" % s1["failed"]
            s2 = None
        else:
            x1 = s1["best"]["app"]
            drop_arrays_v32(s1)
            s2 = nk_stage_v32(a, chs, st, cache0, src_new, do_solve, x1, it_hist, T, "stage 2, this record's covariance", "stage 1's kept iterate's applied fields", ctr)
            solves.append(s2)
            print_stage_v32(a, T, s2)
        if s2 is not None and s2["best"] is not None and s2["best"]["resid"] < d["best"]["resid"]:
            final = s2
            drop_arrays_v32(d)
            cont = "ran; kept from stage 2"
        else:
            if s2 is not None:
                drop_arrays_v32(s2)
                cont = "ran; kept from the direct solve (stage 2's kept residual is not smaller)"
            rc = confirm_v32(a, d["best"], d["evalG"], it_hist, ctr)   # the kept iterate's confirming step again, the history's last entry
            d["confirm_repeated"] = rc
    b = final["best"]
    b["dE"] = final["dE_c"] if final["dE_c"] is not None else b["dE"]
    for q in solves:
        q.pop("evalG", None)
    e32 = {"options": v32_options(a), "solves": [stage_entry_v32(q) for q in solves], "continuation": cont, "kept_from": final["name"],
           "kept_residual": b["resid"], "kept_evaluation": b["k_it"], "confirming_residual": final["r_c"], "confirming_dE": final["dE_c"],
           "meets_stop_rule": final["conv"], "evaluations": ctr["n"], "ends_member": False}
    if "confirm_repeated" in d:
        e32["confirming_step_repeated"] = {"residual": d["confirm_repeated"]["r_c"], "dE": d["confirm_repeated"]["dE_c"]}
    Nn_b, vn_b, Wn_b, info_b = b["sol"]
    N_app_b, v_app_b = b["app"]
    return ctr["n"], final["conv"], b, b["cache"], b["srcs"], Nn_b, vn_b, Wn_b, info_b, N_app_b, v_app_b, e32


def v32_record_segment(e32):
    """the record line's version 32 part: each solve's evaluations and residuals, the solve kept."""
    parts = []
    for q in e32["solves"]:
        parts.append("%s %d evaluations, kept %s, confirming %s (%s)" % (
            q["name"], q["evaluations"], ("%.2e" % q["kept_residual"]) if q["kept_residual"] is not None else "none",
            ("%.2e" % q["confirming_residual"]) if q["confirming_residual"] is not None else "none", q["status"]))
    return "; v32 solve [%s; continuation: %s; kept from %s: residual %s, confirming %s]" % (
        "; ".join(parts), e32["continuation"], e32["kept_from"], ("%.2e" % e32["kept_residual"]) if e32["kept_residual"] is not None else "none",
        ("%.2e" % e32["confirming_residual"]) if e32["confirming_residual"] is not None else "none")


def main():
    global FLD
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
    # version 32 (engine spec 32.1 (3)); every option None by default: the defaults are version 31's
    ap.add_argument("--nk-line-search", default=None, choices=["halving"], help="(a) the backtracking line search on the fixed-point residual (version 32)")
    ap.add_argument("--nk-halvings", type=int, default=None, help="(a) at most this many halvings (6 when absent)")
    ap.add_argument("--nk-krylov", type=int, default=None, help="(b) LGMRES Krylov vectors per cycle (scipy's default 30 when absent)")
    ap.add_argument("--nk-restarts", type=int, default=None, help="(b) LGMRES restarts (none when absent)")
    ap.add_argument("--source-continuation", default=None, choices=["two-stage"], help="(c) the two-stage continuation in the propagated covariance")
    ap.add_argument("--end-above", type=float, default=None, help="(d) the member ends at a record whose kept residual exceeds this after the continuation")
    # version 33 (S11): every option off by default (the defaults are version 32 bitwise)
    ap.add_argument("--massless-ball", dest="massless_ball", action="store_true", help="(b) the massless Fermi ball's preparation (requires --m 0)")
    ap.add_argument("--ball-R0-over-rs", dest="ball_R0_over_rs", type=float, default=4.0)
    ap.add_argument("--ball-x0-over-R0", dest="ball_x0_over_R0", type=float, default=0.5)
    ap.add_argument("--ball-sigma-over-R0", dest="ball_sigma_over_R0", type=float, default=0.25)
    ap.add_argument("--ball-orbitals", dest="ball_orbitals", default="1:6", help="kappa:b,... the chiral seeds per channel")
    ap.add_argument("--ball-E-target", dest="ball_E_target", type=float, default=None, help="the solved charge's target (default r_s/(2g))")
    ap.add_argument("--ball-k0-max", dest="ball_k0_max", type=float, default=0.5)
    ap.add_argument("--ball-E-tol", dest="ball_E_tol", type=float, default=1e-10)
    ap.add_argument("--ball-passes", dest="ball_passes", type=int, default=200)
    ap.add_argument("--ball-tol", dest="ball_tol", type=float, default=1e-10)
    ap.add_argument("--fields-v17", dest="fields_v17", action="store_true", help="(e) engine_fields_v17 for every solve")
    ap.add_argument("--s11-collars", dest="s11_collars", default="8,16")
    ap.add_argument("--s11-stop", dest="s11_stop", type=int, default=1)
    ap.add_argument("--resume-slice", dest="resume_slice", default=None, help="(v34 a) a stored complex128 slice of a v33 or v34 run to restart from")
    ap.add_argument("--resume-run", dest="resume_run", default=None, help="(v34 a) that run's dir/tag")
    ap.add_argument("--segment-end-T-over-rs", dest="segment_end_T_over_rs", type=float, default=None, help="(v34 b) end after the record at this clock")
    ap.add_argument("--weyl-certified", dest="weyl_certified", default=None, choices=["full", "eigvals"], help="(v34 c) the certified continuation's route")
    ap.add_argument("--weyl-refine-max", dest="weyl_refine_max", type=int, default=10, help="(v34 c) at most this many halvings of a sub-step")
    ap.add_argument("--weyl-oracle", dest="weyl_oracle", action="store_true", help="(v34 d) the full-route oracle at every record's accepted fields")
    ap.add_argument("--field-kernel-cache", dest="field_kernel_cache", action="store_true", help="(v36) the field solve's kernels cached, bitwise")
    ap.add_argument("--v36-fixes", dest="v36_fixes", action="store_true", help="(v36) #65 item 7's three corrections")
    ap.add_argument("--nk-jvp", dest="nk_jvp", default=None, choices=["exact"], help="(v35) the Krylov Jacobian's product by the exact linearization")
    ap.add_argument("--pair-workers", dest="pair_workers", type=int, default=0, choices=[0, 2], help="(v36b) the exact J.v's 2 field solves per product in 2 persistent worker processes")
    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")
    a = ap.parse_args()
    if a.m == 0.0 and not a.massless_ball:
        ap.error("version 33 (a): m = 0 runs only with --massless-ball")
    if a.massless_ball and a.m != 0.0:
        ap.error("version 33 (b): the massless ball needs --m 0 exactly")
    if a.massless_ball and a.ball_E_target is None:
        a.ball_E_target = a.rs / (2.0 * a.g)
    if a.fields_v17:
        import engine_fields_v17 as F17
        FLD = F17
    if a.field_kernel_cache:
        import engine_fields_kcache_v1 as KC
        import engine_fields_v4 as _F4, engine_fields_v12 as _F12, engine_fields_v14 as _F14, engine_fields_v16 as _F16
        KC.install([_F4, _F12, _F14, _F16, FLD]); V36["kcache"] = KC.STATS
    V36["fixes"] = bool(a.v36_fixes)
    if a.resume_slice and (a.resume_from or not a.resume_run):
        ap.error("version 34 (a): --resume-slice needs --resume-run and excludes --resume-from")
    if a.nk_jvp and (a.mix != "newton" or not v32_own_loop(a)):
        ap.error("version 35: --nk-jvp acts inside version 32's own Newton loop (--mix newton with --nk-line-search, --nk-krylov or --nk-restarts)")
    if a.weyl_oracle and a.reference != "R2":
        ap.error("version 34 (d): the oracle compares R2's continuation")
    if v32_on(a) and a.mix != "newton":
        ap.error("version 32's options act on the Newton solve: --mix newton is required")
    if a.nk_halvings is not None and a.nk_line_search != "halving":
        ap.error("--nk-halvings needs --nk-line-search halving")
    for k_, v_ in (("--nk-halvings", a.nk_halvings), ("--nk-krylov", a.nk_krylov), ("--nk-restarts", a.nk_restarts)):
        if v_ is not None and v_ < (0 if k_ != "--nk-krylov" else 1):
            ap.error("%s must be %s" % (k_, "at least 1" if k_ == "--nk-krylov" else "nonnegative"))
    OPT["collar_policy"] = a.collar_policy; OPT["support_rel"] = a.support_rel; OPT["site_fields"] = a.site_fields
    SOLVE_TOL["rtol"] = a.field_rtol; SOLVE_TOL["atol"] = a.field_atol
    if a.pair_workers:
        if a.nk_jvp != "exact":
            ap.error("version 36b: --pair-workers acts inside the exact J.v (--nk-jvp exact)")
        PAIR["n"] = a.pair_workers
        PAIR["cfg"] = {"engine": "engine_collapse_v36c", "fld": FLD.__name__, "kcache": bool(a.field_kernel_cache),
                       "opt": dict(OPT), "tol": dict(SOLVE_TOL)}
    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":
            ap.error("version 36c: --nk-set resolve needs --nk-jvp exact and --nk-line-search halving")
        NSR["on"] = True
    C5.backend(a.gpu)
    run(a)


if __name__ == "__main__":
    main()
