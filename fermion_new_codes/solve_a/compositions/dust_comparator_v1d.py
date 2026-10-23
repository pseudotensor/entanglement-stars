#!/usr/bin/env python
"""dust_comparator_v1d.py (the solve lane's edited copy of the delivered dust_comparator_v1c.py, md5 740a94d922178c5be8bbdcf1658be396;
ReviewA_reading_web40j_C5_v1.md part 8, "the exact change", items 1 to 6, written without reading the checking lane's subclass).

v1d, the angular force with both valleys' barriers. (1) The profile: per stored dust column c of channel kappa the count per site
n_c(j) = (16 kappa) f_c (|d_c(2j)|^2 + |d_c(2j+1)|^2) and the column's exact lattice barrier L2_c = sum_j r_j^2 t_c(j) / sum_j
(|d_c(2j)|^2 + |d_c(2j+1)|^2), t_c(j) = Re[conj(u_j)(T u)_j + conj(l_j)(T l)_j], (T u)_j = (kappa/r_j)^2 u_j + (1/2)[(A_(j+1) - A_j) u_(j+1)
- (A_(j-1) - A_j) u_(j-1)], (T l)_j = (kappa/r_j)^2 l_j - (1/2)[the same bracket on l], A = kappa/r, open ends; one shell per (column, site)
whose count is above the delivered floor (n_floor x total), no pruning of any tail. Shells with the same release radius and the same L2
move identically (the same Hamiltonian per particle, the same release momentum), so they are merged into one shell carrying their summed
count; this is exact, and it returns the delivered per-site shells in the control. Modes: control (L2 = 0 for every shell), columns (L2_c
per column), census (per channel and site the count-weighted mean of the columns' L2_c at the site), constant (kappa(kappa - 1), the
0-valley value). (2) ShellSystem: an L2 per shell; in _fields eps = sqrt(m_eff^2 + k^2), m_eff^2 = m^2 + L2/R^2, so the deposit n eps and
shell_masses carry the barrier energy; in rhs dR/dt = N k/eps + sgn v q'(k) and dk/dt = -(sgn dN eps + dv q) + sgn N L2/(eps |R|^3); only
L2 = 0 shells can be pinned at the centre. (3) release: k = m_eff(R_0) w_e/sqrt(1 - w_e^2), w_e = -v_app/N_app (the delivered
k_of_velocity with m replaced by m_eff). (4) k_max 1000 for the barrier runs (20, the delivered value, for the control). (5) The sine
variant: eps = sqrt(m^2 + sin^2 k + L2/R^2), q = sin(2k)/2, the same force, the rest momentum the root of sin(2k)/(2 eps) = -w cos 2k.
(6) Printed per run: E_0, sum n eps and sum n q at rest, the per-channel count inside the fold radius and the per-channel innermost
radius at the fold. The delivered W2 T = 0 slice (ladders/r3/slices/..._v36f_T0000.00.npz) is no longer on the box; the run reads the
W5_k10_tau8 member's T = 0 slice of the same preparation, whose per-site count equals the delivered run's saved profile bit for bit and
whose 16 column barriers equal the checking lane's list (C5_v1d_slice_check_v1.py).

The delivered docstring follows unchanged.

dust_comparator_v1c.py (reviewer A for the busi lane; REVIEWA_REVIEW_web40i.md section 4, busi W4 (a): the dust comparator on the
production's own stored T = 0 profile at sigma/a 2, 4 and 8, the root clock and radius over 2GE printed beside the records' 12.000/0.65,
12.125/0.66, 12.750/0.76; gates the formation sentence's mark, Y3). Box CPU; a classical shell code with a self-test; nothing is read or
ruled by the script, every number is printed with its definition.

v1c (v1b's self-test: 14 PASS, 2 FAIL at 3b and 3d in 5 s). Both FAILs are test expectations that do not hold for the averaged-kernel
comparator; the dynamics has no defect (3a, 3c, 3e, 4, 5, 6, 7 pass, E is conserved to 6e-11 through the fold). (3b) compared the
outermost MASSIVE shell with the point-exterior quadrature of the full charge to 2e-3. That shell is not outside the kernel's support:
it sits at the centre of its own kernel, so its applied fields read the charge inside it as M_K = E - mu_s/2 - (the inner neighbours'
tails beyond it), half its own smeared mass counted outside (1.2 percent of E at release for the test's 128 shells, whose surface shell
carries 2.2 percent of E; the neighbour at 2 sigma has 2.3 percent of its mass beyond), and a clock N_app = 1 - g (mass smeared
beyond R_s)/R_s (0.17 percent); at rest the force per eps is g <N M/x^2>_K exactly (the local density cancels between N' and v'), so its
acceleration is a/a_0 = <N M/x^2>_K N_app (1 - w_app^2)/[(E/R_s^2)(1 - 2gE/R_s)], about 1 - 0.7 delta_K - 2 nu_K, and its clock lags the
quadrature by 1 - sqrt(a/a_0), 0.6 percent at release and growing as the shells crowd below the kernel width (the spacing 2 sigma at release,
under 0.5 sigma near the fold): the observed 0.96 to 1.0 percent. The Birkhoff statement holds beyond the kernel reach, not at the surface
shell. v1c: (3b-i) a massless probe shell released at rest at R_surf + 16 sigma, beyond the kernel reach (7 sigma each way) of every massive
shell, must follow the point-exterior quadrature of the full E_0 to 5 sigma^2/(8 R_p,min^2) + 2e-6 (the density kernel's vacuum average of
x^(-1/2) reduces the flow by sigma^2/(8R^2) and the force by 5 times that; the integrator's share at rtol 1e-10 below 2e-7); (3b-ii) the
surface shell's early lag in t and in tau (records with 0.03 to 0.1 of its quadrature clock) must equal the first-order prediction from
the release fields, 1 - sqrt(a/a_0) and N_app m/eps_s - (m/eps_0) sqrt(a/a_0), to 5 percent of itself plus 5e-6 (the second-order terms are
O(delta (R_0 - R)/R_0), under 3 percent of the lag in that window); its late lag prints with the kernel-read deficits. (3d) compared the
dense mass function at every shell's position at the last pre-crossing record with its release value to 2 percent: that quantity is
sum_j mu_j Phi_j(R_i), the sharp enclosed mass plus the overlap of the neighbours' smeared masses, a few 1e-4 of M at release and
percent-level once the shells crowd below sigma (one-sided at the ball's edge, (3/sqrt(2 pi)) sigma/R_s of E for a smeared uniform edge, 0.5
percent at R_s 255 and 2 percent at 58.5; 3.4 percent measured at the worst shell with the chart's non-homologous crowding). What the dust
dynamics conserves is the sharp enclosed mass, the sum over the shells inside of mu_j = n_j (eps_j + <w>_j q_j) = n_j m e_j, the LTB energy
integral, plus half its own: exactly for sharp shells, and for the smeared shells up to a pairwise exchange g int K_i K_j/(x u) n_i n_j
[q_j R-dot_i e_i - q_i R-dot_j e_j] m, first order in the neighbours' differences across a kernel width (the self-term cancels exactly), a part
in 1e-4 to 1e-3 of the enclosed mass over the fall. v1c: (3d) the sharp enclosed mass kept to 2e-3 and the identity M(x) = sum_j mu_j
Phi_j(x) to 5e-3 of E_0 (its derived residual sigma^2 |u'| s_J printed), the kernel-read drift printed with its decomposition into the sharp
part, the overlap and the identity residual, the pins listed: the log's 2 pins are the ball's innermost shells (R_0 1 a and 3 a, 2.3e-4 and
2.1e-3 particles, 3.6e-6 of the count), inside the central kernel, advected to the centre by the density kernel's nonzero flow there once
the ball has contracted and held as the sliding solution; they move 1e-10 of E, lie inside every counted shell before and after, and explain
none of the drift. Also: the fold's snapshot now counts the shell-order inversions against the last record before it (v1b counted none
there), and the test shell's 'flat clock to the centre' prints at r_g 59.05 and at r_g 64 a (the review's 13.42 is the latter).

v1b (the hang of v1's self-test at its item 3b: 5 PASS lines in 0.6 s, then one thread at 100 percent for 49 minutes with nothing
printed; the first attempt's 15-minute timeout at the same place). THE CAUSE is the signed-radius continuation of a shell through the
centre. Under the scalar kernels (density, radial) a shell at the origin carries a nonzero radial momentum source, K(x; 0) != 0, so the
applied flow at the centre, v_app(0) = int K(x; 0) vbar(x) dx, is nonzero and points inward from both sides, and the velocity of a shell
crossing the centre jumps from N eps'(k) + v_app(0) q'(k) to N eps'(k) - v_app(0) q'(k). Where |N eps'(k)| <= |v_app(0) q'(k)|, which holds
for every shell once the centre is trapped (|v_app(0)| >= N_app(0)) and for slow shells before that, both velocities point toward the centre:
an attracting discontinuity of the right-hand side (a sliding mode), whose exact Filippov solution is R = 0, and on which an explicit
adaptive integrator collapses its step to the tolerance scale and never recovers, every trial step straddling the jump. v1's uniform-ball
item integrated 15 percent past the surface quadrature's clock, through the dust caustic at the centre, so its DOP853 chattered there
without end (no root find, no quadrature and no loop count is involved); the comparator runs to T/r_s 16 would have hung the same way,
after their formation. v1b: (i) a shell that reaches |R| = R_pin (1e-3 a) while it cannot pass is pinned at R = 0 by a terminal event and
the integration restarts (the sliding solution exactly: the force at the centre vanishes for the even kernels, so its momentum is frozen;
it deposits from the centre with the sign of its arrival side so the sources do not jump; it is released when |N eps'| exceeds
|v_app(0) q'| again, on the side its momentum points to); a shell that can pass crosses as before (a transversal jump the integrator takes
in a few rejected steps); for the vector kernel v_app(0) = 0 and no shell is ever pinned; (ii) the energy ledger: every pin, release and
centre crossing is logged with the exact change of E it causes (E on the two sides of the flip, from the fields functional), because
E = M(infinity) is discontinuous at a scalar-kernel centre crossing by -2 n q v_app(0) at first order (the central momentum source flips
sign), so 'E conserved' is read net of the ledger and the raw drift prints beside it; (iii) the guards: a per-run wall-clock budget raised
inside the right-hand side (the records so far stand, the end is labelled), a cap on restarts, a momentum guard k_max for the continuum
dispersion (a vector-kernel trapped centre blueshifts a shell's momentum without bound), the first pin at a trapped centre recorded as the
caustic clock and --after-caustic stop ending the run there, a progress line every minute; (iv) the self-test: the ball runs to its fold
(terminal) and tests what is exact there (v1's 'first root at the surface at the Oppenheimer-Snyder clock within 1 percent' and 'no crossing
before the root' were not properties of the ball released at coordinate rest on the flat slice: that slice is not the comoving slice of
time symmetry, and the lapse gradient makes the chart's contraction non-homologous); shell and centre crossings conserve E on the vector
kernel, and on the density kernel E changes by the exact flip of the central source and is conserved otherwise; a new item 7, a deeply
trapped cluster, pins every shell and completes in bounded time. The fields are cached on (R, k) so the events reuse the right-hand side's
solve.

THE COMPARATOR, DEFINED (input 01's words; input 04's single-shell control generalized to every shell).
The cold comparator is the pressureless (dust) shell system of the model's own spherical chart: flat slices, the clock N and the flow v
fixed on every slice by the 2 constraint rows from the shells' sources, the clock 1 at infinity (the clock at infinity; the lane's T/r_s
with the slice's r_s = 64 a). Shell i carries n_i particles (the per-site dust count of the T = 0 covariance, weighted 16 kappa), an areal
radius R_i(t) and a canonical radial momentum per particle k_i(t); per particle the bare energy eps(k) = sqrt(m^2 + k^2) and the momentum
q(k) = k (the continuum dust, the primary; --dispersion sine replaces them by the chain's sqrt(m^2 + sin^2 k) and sin(2k)/2, the lattice
branch's control, a = 1). The single-particle Hamiltonian in the applied fields is h_i = N(R_i) eps(k_i) + v(R_i) q(k_i) (the model's
coupling (1/2){N, h_0} + (1/2){v, Q} in the one-particle limit), so dR_i/dt = N eps'(k) + v q'(k) and dk_i/dt = -(N' eps + v' q) at the
shell; released at coordinate rest: dR_i/dt = 0 at T = 0, i.e. q/eps = -v/N = |w| (requirements section 2, "the dust at coordinate rest
has q/eps = |w|"). The sources are the shell totals per unit radius, s_rho(x) = sum_i n_i eps_i K(x; R_i) and s_J(x) = sum_i n_i q_i
K(x; R_i), with K the averaging kernel of standard deviation sigma: --kernel density (the default; the 3-dimensional Gaussian of per-axis
standard deviation sigma applied to a thin shell, K = (x/R)[phi(x - R) - phi(x + R)], the form the acceptance criterion's "per-axis
variance ell^2 for the engine's Gaussian" names), radial (the Gaussian on the radial line with the even reflection at the origin) or
vector (the density kernel for rho and the 3-dimensional vector kernel K L(xR/sigma^2), L the Langevin function, for the radial momentum
J, which vanishes at the centre as a vector must); which of these the engine applies is not in the inputs, so the comparator's T = 0
applied fields are printed against the slice's stored N_applied, v_applied under all 3 (the lead reads the convention off that table).
On the averaged sources the energy row gives the mass function, M' = s_rho + w s_J with w = v/N = -sqrt(2gM/x) (M' = 4 pi x^2 (rho + w J),
M = x w^2/(2g) the Misner-Sharp mass, the inward branch), and the momentum row the clock, (ln N)' = -g s_J/(x w), N(infinity) = 1 (the
exact clock law of requirements section 2 with J kept, equivalently (ln N)' = 2 pi x^2 rho/M - M'/(2M)). The fields that move the shells
are the dense fields averaged back with the transpose of the source averaging in the radial measure (the paired averaging; the weighted
adjoint of code defect 2), so that the boundary charge E = M(infinity) is exactly the generator of the shells' motion: dE = sum_i n_i
(N deps_i + v dq_i + (N' eps_i + v' q_i) dR_i) holds to roundoff for the discrete solver (the implicit trapezoid for M and its discrete
adjoint for N; self-test 5) and E is conserved by the flow, the integrator's tolerance its only drift. E = sum_j mu_j exactly with
mu_j = n_j (eps_j + <w>_j q_j), <w>_j the J-kernel average of the dense flow over shell j: the shell's LTB energy integral n_j m e_j (v1c).
The shells cross freely (the multistream continuation): the sources are deposited from the shells' current positions at every
right-hand side, so the charge inside every shell is recomputed at every crossing; a shell that reaches the centre passes through it
where it can (the signed radius: the fields even in R, the flow odd) and re-expands, and is pinned at the centre where it cannot (v1b,
above), the exact sliding solution of the scalar kernels' discontinuous flow there; under the vector kernel the flow vanishes at the centre
and every shell passes. The root: the zeros of N - |v| on the dense fields of the averaged mass function, i.e. of 1 - 2gM(x)/x (w^2 = 1;
the same zero set as the records' trapping_zeros, the zeros of their dense N + v); the first root is the clock at which min_x (N - |v|)
first reaches 0, located by the integrator's event on the dense output (the fold clock), its radius the minimum's position; beside it
the first record clock holding a root on the 8 a record grid (the records' convention), the applied root (the zero of N_app + v_app) and
the depth min(N - |v|) at the record before. At a dense root the charge inside it over E equals its radius over 2gE exactly (2gM(R) = R),
so the comparator's "fraction of E inside the root" is its "radius over 2GE"; the count inside prints beside. The caustic: the first
clock at which a shell is pinned at a trapped centre (|v_app(0)| >= N_app(0)); the continuation past it keeps the pinned shells at the
centre (their mass regular under the kernel) and is the comparator's post-formation history, labelled.
Variants beside the primary (density, continuum, rest, lapse on, adaptive): --lapse off, the point-exterior release in which every
shell moves with N = 1 in the Painleve-Gullstrand field of the averaged charge inside it (input 04's control literally generalized: for
the outermost shell with all the charge inside it the 2 coincide and reproduce input 04's quadrature; with N forced to 1 the charge is
not exactly conserved at crossings and its drift prints); --step frozen --tau 8, the shells advanced with the fields frozen over record
steps of tau as the engine's frozen exponential step freezes its fields (the records' step structure; tau 4 beside); --dispersion sine;
--kernel radial, vector; --release read (the shells released with the mean coordinate velocity read from the T = 0 covariance instead
of rest); --sigma 0.25 (the narrow kernel, the point-shell limit approached).

THE T = 0 PROFILE'S SOURCE. The 3 rungs of COLLAPSE_ladder v2 (sigma/a 2, 4, 8 at r_s/a 64, tau 8; engine_collapse_v25 from T = 0; the
tags c32_rs64_A_R2_tau8_ell{2,4,8}_prep4_avg_plain_v36d_from0 of core_well_v1's inventory line) store states every 1 r_s from T = 0
(input 07: "integer clocks missing none"); their T0000 slices orbitals_<tag>_T0000.00.npz under <internal>/v36d_from0
hold the T = 0 covariance: the dust columns dust_k with their occupations f_k and multiplicities 16 kappa, the chain's radii R and the
record's accepted applied fields N_applied, v_applied. They are not in slices144_io_v1.inventory (which lists the production's slices from
T/r_s 12, B, B2 and the sigma/a 8 member c27), so they enter by path (--slice, or --slice-glob; RUN_W4a_v1.md's find command locates
them), read with slices144_io_v1.SliceReader by name where it imports and by a local reader of the same npz format otherwise. The
per-shell count is core_well_v1's dust number per site from the columns, n_j = sum_kappa 16 kappa sum_c f_c (|d_c(2j)|^2 + |d_c(2j+1)|^2)
(the total 640); the per-shell velocity is the dust's outward number current through every cut between sites by the range-2 straddling-bond
sum of core_well_v1 on h_m = (1/2){N, h_0} + (1/2){v, Q} of the slice's applied fields (page_ops_v6.chain_sparse and clocked_csr by name;
where page_ops_v6 does not import the column prints 'not read' and the release is at rest), divided by the count per unit length at the
cut: the coordinate velocity, 0 for a coordinate-rest preparation. The production's own T = 0 (its records 0 to 11 the replayed parent
c17's) has no stored state in the inventory; the rungs' preparation (prep4) is the one the comparator reads, each sigma from its own rung's
T0000 slice where 3 are given, and the 3 rungs' counts are compared (their largest difference printed). The records json of each rung
(--records) supplies E at T = 0 and the first record holding a trapping zero (the quoted 12.000/0.65, 12.125/0.66, 12.750/0.76 print in
any case, labelled quoted).

SELF-TEST (--self-test; v1c): (1) the local quadrature reproduces input 04's P20 table (13.2248621796 and 12.4186017985 at r_g 59.05,
R_0 256; 12.6937264016 and 11.8416820247 at r_g 64) and its canonical check; (2) a single test shell in the point exterior, advanced by the
comparator's own dynamics, reproduces that quadrature's t and tau to 1e-8; (3) a uniform ball released at rest (N_B 640, R_0 256, the
production's numbers) with a massless probe shell released at rest beyond the kernel reach of every massive shell (R_surf + 16 sigma), run
to its fold: (3a) the dust constraint's charge (against an independent integration of dM/dr = n m sqrt(1 - 2gM/r), the series
1 - (3/10) r_s/R_0 - (3/56)(r_s/R_0)^2 beside), (3b-i) the probe's t and tau against the point-exterior quadrature of the full E_0 to
5 sigma^2/(8 R_p,min^2) + 2e-6 (Birkhoff in the chart: beyond the kernel reach the fields are the point exterior's exactly, the density
kernel's vacuum average of x^(-1/2) the only correction), (3b-ii) the outermost massive shell's early lag behind that quadrature against its
first-order prediction from the release fields to 5 percent (the kernel reads half its own mass and the neighbours' tails outside it and a
clock below 1), its late lag printed with the kernel-read deficits, (3c) E conserved to 1e-7 through the fold, (3d) every shell's sharp
enclosed mass (the LTB energy integrals of the shells inside it plus half its own) kept to 2e-3 at the last record before the first
crossing, the identity M(x) = sum_j mu_j Phi_j(x) to 5e-3 of E_0, the kernel-read drift decomposed and the pins listed, (3e) the first root
forms at or after the surface shell's quadrature clock (the theorem's lower bound) and inside 2gE_0, the multistream delay T_f - t_q printed
as the ball's own; (4) shell crossings and centre crossings conserve the total charge on the vector kernel to 1e-8, and on the density
kernel the charge changes by the exact flip of the central momentum source at the centre passage and is conserved otherwise (the lapse-off
variant's drift prints); (5) the generator identity dE = sum n_i dh_i by central differences on a random configuration for all 3 kernels;
(6) the kernels integrate to 1 and the vacuum lapse is 1; (7) a deeply trapped cluster collapses to the centre: every shell is pinned, the
run completes in bounded time (no chattering), the caustic clock prints, E is conserved net of the ledger; the vector kernel on the same
cluster beside (print). PASS/FAIL per item, the exit code 1 on any FAIL; about 10 seconds in all.

Outputs (--out): DUST_comparator_v1c.md, dust_comparator_v1c_all.json, per run <label>.json and .npz, dust_comparator_v1c.log. CPU only, at
most 16 threads; no plotting; no dates (elapsed seconds only)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import quad, solve_ivp  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    import slices144_io_v1 as IO  # noqa: E402
    IO_NOTE = "slices144_io_v1.SliceReader (%s)" % getattr(IO, "__file__", "?")
except Exception as _ex:  # noqa: BLE001
    IO = None
    IO_NOTE = "slices144_io_v1 not importable (%s: %s): the local npz reader of the same slice format" % (type(_ex).__name__, _ex)
try:
    import numba as _nb  # noqa: E402
except Exception:  # noqa: BLE001
    _nb = None

CODE = "dust_comparator_v1d.py"
TITLE = "DUST_comparator_v1d"
SQRT2PI = math.sqrt(2.0 * math.pi)
T0 = time.time()
QUOTED_RECORDS = {2.0: (12.000, 0.65), 4.0: (12.125, 0.66), 8.0: (12.750, 0.76)}
QUOTED_SOURCE = ("quoted from the request (REVIEWA_REVIEW_web40i.md D2 / the formation sentence): the records' first root at T/r_s 12.000, 12.125 and 12.750 at "
                 "0.65, 0.66 and 0.76 of the record's 2GE at sigma/a 2, 4 and 8")
P20_TABLE = {(59.05, 256.0): (13.2248621796, 12.4186017985), (64.0, 256.0): (12.6937264016, 11.8416820247)}
P20_CANONICAL = (846.3911794961956, 794.7905151009893)
DEFAULT_SLICE_GLOB = "<internal>/orbitals_c32_rs64_A_R2_tau8_ell*_prep4_avg_plain_v36d_from0_T0000.00.npz"
DEFAULT_RECORDS_GLOB = "<internal>/collapse_c32_rs64_A_R2_tau8_ell*_prep4_avg_plain_v36d_from0.json"
DEFAULT_PAGE_CODE = "<internal>/v6"
PROVENANCE = ("every number is computed by this script from the stored T = 0 slice (the counts and, where page_ops_v6 imports, the number current), the records json "
              "where given, and the comparator's own equations; the records' first roots are labelled 'quoted' (the request) or 'read' (the json); nothing is read or ruled")
CENTRE_NOTE = ("v1b, the centre: a shell that reaches |R| = R_pin while |N eps'(k)| <= |v_app(0) q'(k)| (it cannot pass the centre; under the scalar kernels the applied flow at "
               "the centre is nonzero and inward from both sides) is pinned at R = 0, the exact sliding solution, with its momentum frozen (the force at the centre vanishes "
               "for the even kernels) and its sources deposited from the centre with the sign of its arrival side; it is released when |N eps'| exceeds |v_app(0) q'| again; "
               "a shell that can pass crosses the centre (a transversal jump of its velocity); under the vector kernel v_app(0) = 0 and no shell is pinned. The ledger: "
               "every pin, release and centre crossing with the exact change of E it causes (E = M(infinity) jumps at a scalar-kernel centre crossing by -2 n q v_app(0) at "
               "first order, the central momentum source flipping sign), so 'E conserved' is read net of the ledger with the raw drift beside. The caustic: the first pin at "
               "a trapped centre (|v_app(0)| >= N_app(0)); the continuation past it keeps the pinned shells at the centre. Guards: the wall-clock budget per run, the restart "
               "cap, the momentum guard k_max for the continuum dispersion")


class Logger:
    def __init__(self, path=None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __call__(self, msg):
        line = "[%9.1f s] %s" % (time.time() - T0, msg)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def fx(x, fmt):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and not np.isfinite(x):
            return "nan"
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


def jdefault(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ------------------------------------------------------------------------------------------------------- the dispersion
class Dispersion:
    """eps(k), q(k) and their k-derivatives per particle (a = 1): continuum sqrt(m^2 + k^2), k; sine sqrt(m^2 + sin^2 k), sin(2k)/2 (the
    chain's sine hopping and the two-step current's symbol, whose k-derivative is the advection factor cos 2k)."""

    def __init__(self, kind, m):
        if kind not in ("continuum", "sine"):
            raise ValueError("dispersion %s" % kind)
        self.kind = kind
        self.m = float(m)

    def eps(self, k, X=0.0):
        """v1d: X = L2/R^2 the shell's barrier term, eps = sqrt(m^2 + X + k^2) (continuum) or sqrt(m^2 + X + sin^2 k) (sine)."""
        k = np.asarray(k, float)
        if self.kind == "continuum":
            return np.sqrt(self.m * self.m + X + k * k)
        s = np.sin(k)
        return np.sqrt(self.m * self.m + X + s * s)

    def q(self, k):
        k = np.asarray(k, float)
        return np.array(k, float, copy=True) if self.kind == "continuum" else 0.5 * np.sin(2.0 * k)

    def deps(self, k, X=0.0):
        k = np.asarray(k, float)
        e = self.eps(k, X)
        return k / e if self.kind == "continuum" else 0.5 * np.sin(2.0 * k) / e

    def dq(self, k):
        k = np.asarray(k, float)
        return np.ones_like(k) if self.kind == "continuum" else np.cos(2.0 * k)

    def k_of_velocity(self, N, vt, u, X=0.0):
        """the canonical momentum at which the coordinate velocity N eps'(k) + vt q'(k) equals u (vt the signed applied flow at the shell);
        nan where none exists (|u - vt| >= N on the continuum branch). v1d: X = L2/R^2 per shell, m replaced by m_eff = sqrt(m^2 + X)."""
        N = np.asarray(N, float)
        vt = np.asarray(vt, float)
        u = np.asarray(u, float) * np.ones_like(N)
        X = np.asarray(X, float) * np.ones_like(N)
        if self.kind == "continuum":
            we = (u - vt) / N
            out = np.full(we.shape, np.nan)
            ok = np.abs(we) < 1.0
            out[ok] = np.sqrt(self.m * self.m + X[ok]) * we[ok] / np.sqrt(1.0 - we[ok] ** 2)
            return out
        out = np.full(N.shape, np.nan)
        lo, hi = -0.25 * math.pi, 0.25 * math.pi
        for i in range(N.size):
            Ni, vi, ui, Xi = float(N[i]), float(vt[i]), float(u[i]), float(X[i])

            def f(kk):
                return float(Ni * self.deps(kk, Xi) + vi * self.dq(kk) - ui)
            flo, fhi = f(lo), f(hi)
            if flo < 0.0 < fhi:
                out[i] = brentq(f, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=300)
            elif flo == 0.0:
                out[i] = lo
            elif fhi == 0.0:
                out[i] = hi
        return out


# ------------------------------------------------------------------------------------------------------------ the kernel
class Kernel:
    """the averaging kernel K(x; R): the shell total per unit radius at x of a unit shell at radius R (x, R >= 0). radial: the Gaussian on
    the radial line with the even reflection; density: the 3-dimensional Gaussian of per-axis standard deviation sigma applied to a thin
    shell; vector: the density kernel for rho and, for the radial momentum, the 3-dimensional vector kernel K L(xR/sigma^2), L(a) = coth a
    - 1/a. Every form integrates the rho kernel to 1 on x >= 0; the cutoff is the window half-width in sigmas."""

    def __init__(self, sigma, kind="density", cutoff=7.0):
        if kind not in ("density", "radial", "vector"):
            raise ValueError("kernel %s" % kind)
        self.sigma = float(sigma)
        self.kind = kind
        self.cutoff = float(cutoff)

    def phi(self, y):
        s = self.sigma
        return np.exp(-0.5 * (y / s) ** 2) / (SQRT2PI * s)

    @staticmethod
    def _langevin(al):
        al = np.asarray(al, float)
        small = np.abs(al) < 1e-3
        a2 = np.where(small, 1.0, al)
        a2c = np.minimum(np.abs(a2), 30.0) * np.sign(a2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            L = np.where(small, al / 3.0 - al ** 3 / 45.0, 1.0 / np.tanh(a2) - 1.0 / a2)
            Lp = np.where(small, 1.0 / 3.0 - al * al / 15.0, 1.0 / (a2 * a2) - 1.0 / np.sinh(a2c) ** 2)
        return L, Lp

    def K(self, x, R, which="rho"):
        x = np.asarray(x, float)
        R = np.asarray(R, float)
        if self.kind == "radial":
            return self.phi(x - R) + self.phi(x + R)
        s = self.sigma
        small = R < 1e-6 * s
        Rs = np.where(small, 1.0, R)
        D = self.phi(x - Rs) - self.phi(x + Rs)
        Kd = np.where(small, 2.0 * x * x * self.phi(x) / (s * s), (x / Rs) * D)
        if self.kind == "vector" and which == "J":
            L, _ = self._langevin(x * R / (s * s))
            return Kd * L
        return Kd

    def dK(self, x, R, which="rho"):
        """d K(x; R)/d R."""
        x = np.asarray(x, float)
        R = np.asarray(R, float)
        s = self.sigma
        if self.kind == "radial":
            return ((x - R) * self.phi(x - R) - (x + R) * self.phi(x + R)) / (s * s)
        small = R < 1e-3 * s
        Rs = np.where(small, 1.0, R)
        pm, pp = self.phi(x - Rs), self.phi(x + Rs)
        D = pm - pp
        Dp = ((x - Rs) * pm + (x + Rs) * pp) / (s * s)
        dKd = -(x / (Rs * Rs)) * D + (x / Rs) * Dp
        dlim = -(2.0 * x * R / 3.0) * (3.0 * x / s ** 4 - x ** 3 / s ** 6) * self.phi(x)
        dKd = np.where(small, dlim, dKd)
        if self.kind == "vector" and which == "J":
            al = x * R / (s * s)
            L, Lp = self._langevin(al)
            return dKd * L + self.K(x, R, "rho") * Lp * x / (s * s)
        return dKd


class Grid:
    """x_j = j h, j = 0..G, with the trapezoid weights omega (h inside, h/2 at the ends): the radial measure of the paired averaging."""

    def __init__(self, h, x_max):
        self.h = float(h)
        self.G = int(math.ceil(float(x_max) / self.h))
        self.x = self.h * np.arange(self.G + 1)
        self.omega = np.full(self.G + 1, self.h)
        self.omega[0] = 0.5 * self.h
        self.omega[-1] = 0.5 * self.h


def _recursion_core(x, h, s_rho, s_J, M_c, g):
    """the energy row on the grid: the implicit trapezoid M_j = M_{j-1} + (h/2)(f_{j-1} + f_j), f = s_rho - u s_J, u = sqrt(2gM/x), solved
    in closed form per step (a quadratic in u); u clamped at 0 where the mass function would turn negative (counted)."""
    G = x.shape[0] - 1
    M = np.empty(G + 1)
    u = np.empty(G + 1)
    M[0] = M_c
    u[0] = 0.0
    f_prev = s_rho[0]
    neg = 0
    two_g = 2.0 * g
    for j in range(1, G + 1):
        C = M[j - 1] + 0.5 * h * (f_prev + s_rho[j])
        bq = 0.5 * h * s_J[j]
        A = x[j] / two_g
        disc = bq * bq + 4.0 * A * C
        if disc >= 0.0:
            uj = (-bq + math.sqrt(disc)) / (2.0 * A)
        else:
            uj = -1.0
        if uj < 0.0:
            uj = 0.0
            neg += 1
            Mj = C
        else:
            Mj = A * uj * uj
        M[j] = Mj
        u[j] = uj
        f_prev = s_rho[j] - uj * s_J[j]
    return M, u, neg


RECURSION_ROUTE = "pure Python loop"
_recursion = _recursion_core
if _nb is not None:
    try:
        _rec_nb = _nb.njit(cache=False)(_recursion_core)
        _tx = 0.5 * np.arange(5, dtype=float)
        _rec_nb(_tx, 0.5, np.zeros(5), np.zeros(5), 0.0, 0.25)
        _recursion = _rec_nb
        RECURSION_ROUTE = "numba njit (checked against the Python loop at import)"
    except Exception as _ex:  # noqa: BLE001
        RECURSION_ROUTE = "pure Python loop (numba present but the jit failed: %s)" % type(_ex).__name__


class FieldSolver:
    """the 2 constraint rows on the averaged sources with the paired averaging: deposit (S), the mass function (the energy row), the clock
    by the discrete adjoint of the energy row's linearization (the momentum row's (ln N)' = -g s_J/(x w) to second order, exactly the
    weights that make E = M(infinity) the generator), the applied fields S^T (dense) with the kernel's R-derivatives for the forces."""

    def __init__(self, grid, kernel, g, lapse=True, M_c=0.0):
        self.grid = grid
        self.kernel = kernel
        self.g = float(g)
        self.lapse = bool(lapse)
        self.M_c = float(M_c)
        self.half = int(math.ceil(kernel.cutoff * kernel.sigma / grid.h))
        self.W = 2 * self.half + 3
        self._grid_kw = None

    def window(self, Rabs):
        j0 = np.floor((Rabs - self.kernel.cutoff * self.kernel.sigma) / self.grid.h).astype(np.int64) - 1
        idx = j0[:, None] + np.arange(self.W, dtype=np.int64)[None, :]
        valid = (idx >= 0) & (idx <= self.grid.G)
        return np.where(valid, idx, 0), valid

    def kernels(self, Rabs):
        idx, valid = self.window(Rabs)
        xw = self.grid.x[idx]
        Rc = Rabs[:, None]
        K_rho = self.kernel.K(xw, Rc, "rho") * valid
        dK_rho = self.kernel.dK(xw, Rc, "rho") * valid
        if self.kernel.kind == "vector":
            K_J = self.kernel.K(xw, Rc, "J") * valid
            dK_J = self.kernel.dK(xw, Rc, "J") * valid
        else:
            K_J, dK_J = K_rho, dK_rho
        return {"idx": idx, "valid": valid, "om": self.grid.omega[idx], "K_rho": K_rho, "K_J": K_J, "dK_rho": dK_rho, "dK_J": dK_J}

    def grid_kernels(self):
        if self._grid_kw is None:
            self._grid_kw = self.kernels(self.grid.x)
        return self._grid_kw

    def deposit(self, KW, a_rho, a_J):
        G1 = self.grid.G + 1
        flat = KW["idx"].ravel()
        s_rho = np.bincount(flat, weights=(a_rho[:, None] * KW["K_rho"]).ravel(), minlength=G1)
        s_J = np.bincount(flat, weights=(a_J[:, None] * KW["K_J"]).ravel(), minlength=G1)
        return s_rho, s_J

    def solve(self, s_rho, s_J):
        if self.M_c > 0.0 and abs(float(s_J[0])) > 0.0:
            raise ValueError("a central point mass with a momentum source at the origin (the radial kernel): unsupported")
        M, u, neg = _recursion(self.grid.x, self.grid.h, np.asarray(s_rho, float), np.asarray(s_J, float), self.M_c, self.g)
        if self.lapse:
            Nbar, ok = self.adjoint_lapse(s_J, M, u)
        else:
            Nbar, ok = np.ones_like(M), True
        return M, u, Nbar, int(neg), ok

    def adjoint_lapse(self, s_J, M, u):
        x, h, G = self.grid.x, self.grid.h, self.grid.G
        c = np.zeros(G + 1)
        pos = (M > 0.0) & (u > 0.0)
        pos[0] = False
        c[pos] = -self.g / (x[pos] * u[pos])
        a = 0.5 * h * np.asarray(s_J, float) * c
        ok = bool(np.all(a < 1.0))
        A_ = 1.0 + a
        B_ = 1.0 / (1.0 - np.minimum(a, 0.999999))
        fac = B_[1:] * A_[:-1]
        mu = np.empty(G + 1)
        mu[G] = 1.0
        mu[:-1] = np.cumprod(fac[::-1])[::-1]
        Wt = np.empty(G + 1)
        Wt[1:-1] = 0.5 * h * (mu[2:] * B_[2:] + mu[1:-1] * B_[1:-1])
        Wt[0] = 0.5 * h * mu[1] * B_[1]
        Wt[G] = 0.5 * h * mu[G] * B_[G]
        return Wt / self.grid.omega, ok

    def applied(self, KW, Nbar, vbar):
        idx, om = KW["idx"], KW["om"]
        Nb, vb = Nbar[idx], vbar[idx]
        return (np.sum(om * KW["K_rho"] * Nb, axis=1), np.sum(om * KW["K_J"] * vb, axis=1),
                np.sum(om * KW["dK_rho"] * Nb, axis=1), np.sum(om * KW["dK_J"] * vb, axis=1))


def _zeros_of(x, f):
    """the sign changes of f on the grid (x[0] excluded), linear between points: (position, +1 rising with x / -1 falling)."""
    f = np.asarray(f, float)
    out = []
    s0, s1 = f[1:-1], f[2:]
    idx = np.nonzero(((s0 < 0.0) & (s1 >= 0.0)) | ((s0 >= 0.0) & (s1 < 0.0)))[0]
    for i in idx:
        if s1[i] == s0[i]:
            continue
        j = i + 1
        out.append((float(x[j] - s0[i] * (x[j + 1] - x[j]) / (s1[i] - s0[i])), 1 if s1[i] > s0[i] else -1))
    return out


def _outer_inner(zeros):
    """the outer root: the largest zero at which N - |v| rises outward (the trapped region's outer edge); the inner: the largest falling
    zero below it."""
    rising = [z for z, d in zeros if d > 0]
    if not rising:
        return None, None
    outer = max(rising)
    falling = [z for z, d in zeros if d < 0 and z < outer]
    return outer, (max(falling) if falling else None)


# ------------------------------------------------------------------------------------------------------------ the shells
class ShellSystem:
    def __init__(self, n, R0, disp, solver, g, m, rs, vacuum=False, R_pin=1e-3, max_pins_per_shell=6, L2=None):
        self.n = np.asarray(n, float).reshape(-1)
        self.S = self.n.size
        # v1d: the barrier per shell (L2 = 0 is the delivered comparator)
        self.L2 = np.zeros(self.S) if L2 is None else np.asarray(L2, float).reshape(-1)
        self.has_L2 = self.L2 > 0.0
        self.any_L2 = bool(self.has_L2.any())
        self.R0 = np.asarray(R0, float).reshape(-1)
        self.disp = disp
        self.solver = solver
        self.g = float(g)
        self.m = float(m)
        self.rs = float(rs)
        self.vacuum = bool(vacuum)
        self.E0 = None
        self.n_rhs = 0
        # v1b: the centre's state (the pinned shells, their deposit sign, the pin counts) and the fields cache
        self.R_pin = float(R_pin)
        self.max_pins_per_shell = int(max_pins_per_shell)
        self.pinned = np.zeros(self.S, bool)
        self.pin_sgn = np.ones(self.S)
        self.pin_count = np.zeros(self.S, np.int64)
        self.pins = 0
        self.unpins = 0
        self.KW0 = None if self.vacuum else solver.kernels(np.zeros(1))
        self._key = None
        self._F = None

    def invalidate(self):
        """v1b: drops the cached fields (after a pin or a release, which change the deposit signs at unchanged positions)."""
        self._key = None
        self._F = None

    def pack(self, R, k, tau):
        return np.concatenate([np.asarray(R, float), np.asarray(k, float), np.asarray(tau, float)])

    def unpack(self, y):
        S = self.S
        return y[:S], y[S:2 * S], y[2 * S:3 * S]

    def signs(self, R):
        """v1b: the side of every shell (+1 at R >= 0), the pinned shells keeping their arrival side."""
        sgn = np.where(R >= 0.0, 1.0, -1.0)
        if self.pinned.any():
            sgn = np.where(self.pinned, self.pin_sgn, sgn)
        return sgn

    def fields(self, R, k, frozen=None):
        """the sources deposited from the shells, the 2 rows solved, the applied fields and their derivatives at the shells, and the
        applied fields at the centre (N0, v0); with frozen = (Nbar, vbar, u) the applied fields are read from those dense arrays (the
        frozen-field step). v1b: cached on (R, k, frozen), so the events at a step's end reuse the right-hand side's solve."""
        R = np.asarray(R, float)
        k = np.asarray(k, float)
        key = self._key
        if self._F is not None and key is not None and key[2] is frozen and np.array_equal(key[0], R) and np.array_equal(key[1], k):
            return self._F
        F = self._fields(R, k, frozen)
        self._key = (R.copy(), k.copy(), frozen)
        self._F = F
        return F

    def barrier_X(self, Rabs):
        """v1d: X = L2/R^2 per shell (0 where L2 = 0)."""
        if not self.any_L2:
            return np.zeros_like(Rabs)
        return np.where(self.has_L2, self.L2 / np.maximum(Rabs, 1e-12) ** 2, 0.0)

    def _fields(self, R, k, frozen):
        Rabs = np.abs(R)
        sgn = self.signs(R)
        X = self.barrier_X(Rabs)
        eps = self.disp.eps(k, X)
        q = self.disp.q(k)
        F = {"Rabs": Rabs, "sgn": sgn, "eps": eps, "q": q, "X": X}
        if self.vacuum:
            Mc = self.solver.M_c
            Rs = np.maximum(Rabs, 1e-300)
            F.update({"N_app": np.ones_like(Rabs), "v_app": -np.sqrt(2.0 * self.g * Mc / Rs), "dN": np.zeros_like(Rabs),
                      "dv": 0.5 * np.sqrt(2.0 * self.g * Mc) * Rs ** -1.5, "E": Mc, "M": None, "u": None, "N0": 1.0, "v0": 0.0, "neg": 0, "lapse_ok": True, "frozen": False})
            return F
        KW = self.solver.kernels(Rabs)
        if frozen is not None:
            Nbar, vbar, u = frozen
            N_app, v_app, dN, dv = self.solver.applied(KW, Nbar, vbar)
            N0, v0, _, _ = self.solver.applied(self.KW0, Nbar, vbar)
            F.update({"KW": KW, "M": None, "u": u, "Nbar": Nbar, "vbar": vbar, "N_app": N_app, "v_app": v_app, "dN": dN, "dv": dv, "E": None, "neg": 0, "lapse_ok": True,
                      "N0": float(N0[0]), "v0": float(v0[0]), "frozen": True})
            return F
        s_rho, s_J = self.solver.deposit(KW, self.n * eps, self.n * q * sgn)
        M, u, Nbar, neg, ok = self.solver.solve(s_rho, s_J)
        vbar = -Nbar * u
        N_app, v_app, dN, dv = self.solver.applied(KW, Nbar, vbar)
        N0, v0, _, _ = self.solver.applied(self.KW0, Nbar, vbar)
        F.update({"KW": KW, "s_rho": s_rho, "s_J": s_J, "M": M, "u": u, "Nbar": Nbar, "vbar": vbar, "N_app": N_app, "v_app": v_app, "dN": dN, "dv": dv,
                  "E": float(M[-1]), "neg": neg, "lapse_ok": ok, "N0": float(N0[0]), "v0": float(v0[0]), "frozen": False})
        return F

    def kernel_read(self, F, arr, which="rho"):
        """v1c: the kernel average of a dense grid array at every shell, sum_c om_c K(x_c; |R_i|) arr(x_c) (the applied-field weights;
        'J' uses the momentum kernel, which differs from the density one for the vector kernel only)."""
        KW = F["KW"]
        K = KW["K_J"] if which == "J" else KW["K_rho"]
        return np.sum(KW["om"] * K * np.asarray(arr, float)[KW["idx"]], axis=1)

    def shell_masses(self, F):
        """v1c: every shell's contribution to the mass function beyond its kernel, mu_j = n_j (eps_j - sgn_j q_j <u>_j), <u>_j the J-kernel
        average of the dense u = |w| (the discrete form of n_j (eps_j + w_j q_j) = n_j m e_j, the shell's LTB energy integral); the implicit
        trapezoid's E = M(infinity) is sum_j mu_j exactly (the kernel's normalization to 1e-16), so a pair of shells can only exchange mu."""
        u_av = self.kernel_read(F, F["u"], "J")
        return self.n * (F["eps"] - F["sgn"] * F["q"] * u_av), u_av

    def rhs(self, t, y, frozen=None):
        """the shells' Hamilton equations (the pinned shells at rest with their momenta frozen; the force at the centre vanishes for the
        even kernels, so this is the sliding solution exactly)."""
        self.n_rhs += 1
        R, k, tau = self.unpack(y)
        F = self.fields(R, k, frozen)
        dR = F["N_app"] * self.disp.deps(k, F["X"]) + F["sgn"] * F["v_app"] * self.disp.dq(k)
        dk = -(F["sgn"] * F["dN"] * F["eps"] + F["dv"] * F["q"])
        if self.any_L2:  # v1d: the centrifugal force -N d eps/dR = sgn N L2/(eps |R|^3)
            Ra = np.maximum(F["Rabs"], 1e-12)
            dk = dk + np.where(self.has_L2, F["sgn"] * F["N_app"] * self.L2 / (F["eps"] * Ra ** 3), 0.0)
        dtau = F["N_app"] * self.m / F["eps"]
        if self.pinned.any():
            dR = np.where(self.pinned, 0.0, dR)
            dk = np.where(self.pinned, 0.0, dk)
        return np.concatenate([dR, dk, dtau])

    def stuck(self, k, N0, v0):
        """v1b: the shells whose velocity would reverse on crossing the centre, |N eps'(k)| <= |v_app(0) q'(k)|: the attracting case."""
        return (np.abs(N0 * self.disp.deps(k)) <= np.abs(v0 * self.disp.dq(k))) & ~self.has_L2  # v1d: only L2 = 0 shells pin

    def free_margin(self, k, N0, v0):
        """v1b: positive where a pinned shell can leave the centre (with a hysteresis of 1e-6 against the stuck test)."""
        return np.abs(N0 * self.disp.deps(k)) - np.abs(v0 * self.disp.dq(k)) * (1.0 + 1e-6) - 1e-12

    def snapshot(self, t, y, prev_ranks=None):
        R, k, tau = self.unpack(y)
        F = self.fields(R, k)
        grid = self.solver.grid
        x, h, G = grid.x, grid.h, grid.G
        u, Nbar, M = F["u"], F["Nbar"], F["M"]
        depth_arr = Nbar * (1.0 - u)
        j = 1 + int(np.argmin(depth_arr[1:]))
        depth, R_min = float(depth_arr[j]), float(x[j])
        if 1 <= j <= G - 1:
            y0_, y1_, y2_ = depth_arr[j - 1], depth_arr[j], depth_arr[j + 1]
            den = y0_ - 2.0 * y1_ + y2_
            if den > 0.0 and y1_ <= y0_ and y1_ <= y2_:
                R_min = float(x[j] + 0.5 * h * (y0_ - y2_) / den)
                depth = float(y1_ - 0.125 * (y0_ - y2_) ** 2 / den)
        outer, inner = _outer_inner(_zeros_of(x, 1.0 - u))
        KWg = self.solver.grid_kernels()
        N_g, v_g, _, _ = self.solver.applied(KWg, Nbar, F["vbar"])
        a_outer, a_inner = _outer_inner(_zeros_of(x, N_g + v_g))
        Rabs = F["Rabs"]
        order = np.argsort(Rabs, kind="stable")
        ranks = np.empty(self.S, dtype=np.int64)
        ranks[order] = np.arange(self.S)
        inv = None
        if prev_ranks is not None:
            seq = prev_ranks[order]
            inv = int(np.sum(np.triu(seq[:, None] > seq[None, :], 1)))
        cum = np.cumsum(self.n[order])
        tot = float(cum[-1]) if cum.size else 0.0
        R_50 = float(Rabs[order][min(int(np.searchsorted(cum, 0.5 * tot)), self.S - 1)]) if tot > 0 else None
        E = float(M[-1])
        E0 = self.E0 if self.E0 else E
        massive = self.n > 0.0  # v1c: the outermost and innermost MASSIVE shells (a massless probe shell is not the surface)
        if massive.any():
            i_out = int(np.argmax(np.where(massive, self.R0, -np.inf)))
            i_in = int(np.argmin(np.where(massive, self.R0, np.inf)))
        else:
            i_out, i_in = int(np.argmax(self.R0)), int(np.argmin(self.R0))
        count_in = float(np.sum(self.n[Rabs <= R_min])) / tot if tot > 0 else None
        return {"t": float(t), "T_over_rs": float(t) / self.rs, "E": E, "E_over_E0": E / E0, "depth": depth, "depth_R": R_min, "depth_R_over_2gE0": R_min / (2.0 * self.g * E0),
                "depth_R_over_rs": R_min / self.rs, "count_fraction_inside_depth_R": count_in,
                "outer": outer, "inner": inner, "outer_over_2gE0": (outer / (2.0 * self.g * E0)) if outer is not None else None,
                "outer_over_2gE_t": (outer / (2.0 * self.g * E)) if (outer is not None and E > 0) else None, "outer_over_rs": (outer / self.rs) if outer is not None else None,
                "applied_outer": a_outer, "applied_inner": a_inner, "inversions": inv, "n_center": int(np.sum(R < 0.0)), "R_50": R_50, "max_u": float(np.max(u)),
                "N_centre": float(Nbar[1]) if G >= 1 else None, "N_min": float(np.min(Nbar)), "neg_mass_points": int(F["neg"]), "lapse_ok": bool(F["lapse_ok"]),
                "tau_outer_shell": float(tau[i_out]), "tau_inner_shell": float(tau[i_in]), "R_outer_shell": float(Rabs[i_out]), "R_inner_shell": float(Rabs[i_in]),
                "n_pinned": int(self.pinned.sum()), "count_pinned": float(self.n[self.pinned].sum()), "N0": F["N0"], "v0": F["v0"], "ranks": ranks}


def cumulative_kernels(system, Rabs):
    """v1c (the self-test's overlap decomposition): Phi[j, i] = int_0^{R_i} K(y; R_j) dy, the cumulative density kernel of shell j at shell
    i's radius, by the trapezoid on the solver's grid and linear interpolation at the shells' radii (the rho kernel; for the vector kernel
    the J part would need its own)."""
    grid, kern = system.solver.grid, system.solver.kernel
    x = grid.x
    Rabs = np.asarray(Rabs, float)
    Phi = np.empty((Rabs.size, Rabs.size))
    for j in range(Rabs.size):
        Kj = kern.K(x, Rabs[j], "rho")
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (Kj[1:] + Kj[:-1]) * np.diff(x))])
        Phi[j] = np.interp(Rabs, x, cum)
    return Phi


def sharp_theta(Rabs):
    """v1c: theta[j, i] = 1 where shell j lies inside shell i by the current radii (stable order), 1/2 for j = i, 0 where it lies outside:
    the sharp-shell enclosed mass is sum_j mu_j theta[j, i]."""
    Rabs = np.asarray(Rabs, float)
    order = np.argsort(Rabs, kind="stable")
    rank = np.empty(Rabs.size, np.int64)
    rank[order] = np.arange(Rabs.size)
    th = (rank[:, None] < rank[None, :]).astype(float)
    np.fill_diagonal(th, 0.5)
    return th


def release(system, mode="rest", u_target=None, maxit=300, tol=1e-13):
    """the shells' momenta of coordinate rest (dR/dt = 0 at T = 0) or of the read coordinate velocity, self-consistently with the fields
    (the fixed point converges fast: at rest kappa = eps + w q is stationary in k)."""
    S = system.S
    R0 = system.R0
    k = np.zeros(S)
    u_t = np.zeros(S) if (mode == "rest" or u_target is None) else np.nan_to_num(np.asarray(u_target, float), nan=0.0)
    d = None
    it = 0
    bad = 0
    for it in range(1, maxit + 1):
        F = system.fields(R0, k)
        vt = F["sgn"] * F["v_app"]
        k_new = system.disp.k_of_velocity(F["N_app"], vt, u_t, F["X"])  # v1d: m_eff(R_0)
        if np.any(~np.isfinite(k_new)):
            bad = int(np.sum(~np.isfinite(k_new)))
            k_new = np.where(np.isfinite(k_new), k_new, k)
        else:
            bad = 0
        d = float(np.max(np.abs(k_new - k))) if S else 0.0
        k = k_new
        if d < tol:
            break
    F = system.fields(R0, k)
    vt = F["sgn"] * F["v_app"]
    resid = float(np.max(np.abs(F["N_app"] * system.disp.deps(k, F["X"]) + vt * system.disp.dq(k) - u_t))) if S else 0.0
    return k, {"iterations": it, "last_change": d, "velocity_residual": resid, "shells_without_rest": bad}, F


# ------------------------------------------------------------------------------------------------- the integrator (v1b)
class Budget(Exception):
    """v1b: raised inside the right-hand side when a run's wall-clock budget is spent."""


def ledger_sums(ledger):
    pins = [e for e in ledger if e["kind"] in ("pin", "unpin")]
    cr = [e for e in ledger if e["kind"] == "crossing"]
    return {"entries": len(ledger), "pins": sum(1 for e in ledger if e["kind"] == "pin"), "unpins": sum(1 for e in ledger if e["kind"] == "unpin"), "crossings": len(cr),
            "dE_pins_exact": float(sum(e["dE_exact"] for e in pins)), "dE_crossings_exact": float(sum(e["dE_exact"] for e in cr)),
            "dE_crossings_first_order": float(sum((e["dE_first_order"] or 0.0) for e in cr)), "dE_total_exact": float(sum(e["dE_exact"] for e in ledger))}


def net_drift(recs, ledger, E0):
    """v1b: the charge's largest drift over the records net of the ledger's exact jumps (pins, releases, centre crossings) up to each clock."""
    worst = 0.0
    for s in recs:
        led = sum(e["dE_exact"] for e in ledger if e["t"] <= s["t"] + 1e-12)
        worst = max(worst, abs(s["E"] - E0 - led) / E0)
    return float(worst)


def integrate(system, y0, t0, t1, t_eval, rtol, atol, x_guard, budget_s, frozen=None, fold_terminal=False, k_max=None, max_restarts=None,
              stop_at_caustic=False, log=None, heartbeat_s=60.0):
    """v1b. DOP853 over [t0, t1] in segments. Events: the fold (min(N - |v|) crossing 0 downward on the dense fields; terminal where asked),
    the grid guard, the momentum guard, the centre's pin (a shell that cannot pass reaching |R| = R_pin) and release (a pinned shell whose
    |N eps'| exceeds |v_app(0) q'|), the last 2 terminal: the state is changed (the shell set at R = 0 with its arrival side kept as its
    deposit sign and its momentum frozen, or set at +-R_pin on the side its momentum points to) and the integration restarts. The wall-clock
    budget is raised inside the right-hand side. The ledger holds every pin, release and centre crossing with the exact change of E it
    causes (E on the two sides of the flip from the fields functional; for a crossing also the first-order -2 n q v_app(0)). Returns the
    records at t_eval (snapshots and states), the fold (v1c: its shell-order inversions counted against the last record at or before it),
    the ledger, the caustic (the first pin at a trapped centre) and the end label."""
    disp = system.disp
    live = frozen is None
    t_start = time.time()
    hb = [t_start]
    n0 = system.n_rhs
    t_eval = np.asarray(t_eval, float).reshape(-1)
    rec_set = set(np.round(t_eval, 9).tolist())
    recs, states, folds, ledger = [], [], [], []
    rank_hist = []  # v1c: (t, ranks) of every record, for the fold's inversion count against the last record at or before it
    fold = None
    label = None
    status = 0
    restarts = 0
    caustic = None
    prev_ranks = [None]
    cap = int(max_restarts) if max_restarts is not None else int(16 + 8 * system.S)

    def rhs(t, y):
        now = time.time()
        if now - t_start > budget_s:
            raise Budget("the wall-clock budget of %.0f s" % budget_s)
        if log is not None and now - hb[0] > heartbeat_s:
            hb[0] = now
            log("    ... t %.2f a (T/r_s %.4f): %d right-hand sides, %d pinned, %d restarts, %.0f s" % (t, t / system.rs, system.n_rhs - n0, int(system.pinned.sum()), restarts, now - t_start))
        return system.rhs(t, y, frozen)

    def ev_fold(t, y):
        R, k, _ = system.unpack(y)
        F = system.fields(R, k, frozen)
        return float(np.min(1.0 - F["u"][1:]))
    ev_fold.direction = -1.0
    ev_fold.terminal = bool(fold_terminal)

    def ev_escape(t, y):
        R = system.unpack(y)[0]
        return float(x_guard - np.max(np.abs(R)))
    ev_escape.direction = -1.0
    ev_escape.terminal = True

    def ev_pin(t, y):
        R, k, _ = system.unpack(y)
        F = system.fields(R, k, frozen)
        st = system.stuck(k, F["N0"], F["v0"]) & ~system.pinned
        if not st.any():
            return 1.0
        return float(np.min(np.abs(R[st])) - system.R_pin)
    ev_pin.direction = -1.0
    ev_pin.terminal = True

    def ev_unpin(t, y):
        cand = system.pinned & (system.pin_count < system.max_pins_per_shell)
        if not cand.any():
            return -1.0
        R, k, _ = system.unpack(y)
        F = system.fields(R, k, frozen)
        return float(np.max(system.free_margin(k[cand], F["N0"], F["v0"])))
    ev_unpin.direction = 1.0
    ev_unpin.terminal = True

    def ev_kmax(t, y):
        k = system.unpack(y)[1]
        return float(k_max - np.max(np.abs(k)))
    ev_kmax.direction = -1.0
    ev_kmax.terminal = True

    events = ([ev_fold] if live else []) + [ev_escape, ev_pin, ev_unpin] + ([ev_kmax] if k_max is not None else [])
    I_FOLD = 0 if live else None
    I_ESC = 1 if live else 0
    I_PIN, I_UNPIN = I_ESC + 1, I_ESC + 2
    I_K = (I_ESC + 3) if k_max is not None else None

    def live_E(R, k):
        return float(system.fields(R, k, None)["E"])

    def settle(t, y, trigger=None):
        """pins the shells inside the band that cannot pass and releases the pinned ones that can, with the exact energy change logged."""
        nonlocal caustic
        R, k, tau = [np.array(v, float, copy=True) for v in system.unpack(y)]
        changed_any = False
        for _round in range(8):
            F = system.fields(R, k, frozen)
            N0, v0 = F["N0"], F["v0"]
            a_ = np.abs(N0 * disp.deps(k))
            b_ = np.abs(v0 * disp.dq(k))
            band = (~system.pinned) & (np.abs(R) <= system.R_pin * (1.0 + 1e-6))
            band = band & ~system.has_L2  # v1d: only L2 = 0 shells pin
            arriving = band & (a_ <= b_ * (1.0 + 1e-6) + 1e-12)
            if trigger == "pin" and not arriving.any() and not changed_any and band.any():
                arriving = band & (a_ <= b_ * (1.0 + 1e-3) + 1e-9)
            cand = system.pinned & (system.pin_count < system.max_pins_per_shell)
            freeing = cand & (system.free_margin(k, N0, v0) > 0.0)
            if not arriving.any() and not freeing.any():
                break
            E_before = live_E(R, k)
            trapped = bool(abs(v0) >= N0)
            if arriving.any():
                idx = np.nonzero(arriving)[0]
                system.pin_sgn[idx] = np.where(R[idx] >= 0.0, 1.0, -1.0)
                R[idx] = 0.0
                system.pinned[idx] = True
                system.pin_count[idx] += 1
                system.pins += int(idx.size)
                kind = "pin"
                if trapped and caustic is None:
                    caustic = {"t": float(t), "T_over_rs": float(t) / system.rs, "N0": N0, "v0": v0, "shells": int(idx.size), "count": float(system.n[idx].sum()),
                               "definition": "the first pin at a trapped centre, |v_app(0)| >= N_app(0): the dust caustic of the comparator; the pinned shells stay at the centre afterwards"}
            else:
                idx = np.nonzero(freeing)[0]
                de = disp.deps(k[idx])
                s = np.where(de > 0.0, 1.0, np.where(de < 0.0, -1.0, system.pin_sgn[idx]))
                R[idx] = s * system.R_pin
                system.pinned[idx] = False
                system.unpins += int(idx.size)
                kind = "unpin"
            changed_any = True
            system.invalidate()
            E_after = live_E(R, k)
            ledger.append({"kind": kind, "t": float(t), "T_over_rs": float(t) / system.rs, "shells": [int(i) for i in idx], "count": float(system.n[idx].sum()),
                           "dE_exact": E_after - E_before, "dE_first_order": None, "N0": N0, "v0": v0, "trapped": trapped})
        return system.pack(R, k, tau)

    def crossings(sol, ta, ya, tb, yb):
        """the shells whose signed radius changed sign between the samples a and b: the crossing clock by brentq on the dense output,
        the exact jump E(R_i -> -0) - E(R_i -> +0) at that state and its first-order form -2 n q v_app(0) sgn_before."""
        Ra = system.unpack(ya)[0]
        Rb = system.unpack(yb)[0]
        flip = np.nonzero((np.sign(Ra) * np.sign(Rb) < 0.0) & ~system.pinned)[0]
        for i in flip:
            fn = lambda tt, i=i: float(sol.sol(tt)[i])  # noqa: E731
            try:
                tc = brentq(fn, ta, tb, xtol=1e-9, maxiter=200)
            except (ValueError, RuntimeError):
                continue
            yc = np.array(sol.sol(tc), float)
            Rc, kc, _ = system.unpack(yc)
            sb = 1.0 if Ra[i] > 0.0 else -1.0
            Rp = Rc.copy()
            Rp[i] = sb * 1e-9
            Rm = Rc.copy()
            Rm[i] = -sb * 1e-9
            Fp = system.fields(Rp, kc, frozen)
            v0, N0 = Fp["v0"], Fp["N0"]
            Ep = live_E(Rp, kc)
            Em = live_E(Rm, kc)
            ledger.append({"kind": "crossing", "t": float(tc), "T_over_rs": float(tc) / system.rs, "shells": [int(i)], "count": float(system.n[i]), "dE_exact": Em - Ep,
                           "dE_first_order": float(-2.0 * system.n[i] * float(disp.q(kc[i])) * v0 * sb), "N0": N0, "v0": v0, "trapped": bool(abs(v0) >= N0)})

    t = float(t0)
    y = settle(t, np.array(y0, float, copy=True))
    if round(t, 9) in rec_set:
        s = system.snapshot(t, y, None)
        prev_ranks[0] = s["ranks"]
        rank_hist.append((t, s["ranks"]))
        recs.append(s)
        states.append(y.copy())
    if stop_at_caustic and caustic is not None:
        label = "stopped at the central caustic at T/r_s %.4f (--after-caustic stop)" % (t / system.rs)
        status = 5
    while label is None:
        if t >= t1 - 1e-12:
            break
        te = t_eval[(t_eval > t + 1e-12) & (t_eval <= t1 + 1e-12)]
        te_full = te if (te.size and abs(te[-1] - t1) <= 1e-9) else np.concatenate([te, [t1]])
        try:
            sol = solve_ivp(rhs, (t, t1), y, method="DOP853", rtol=rtol, atol=atol, t_eval=te_full, dense_output=True, events=events)
        except Budget as ex:
            label = "ended by %s at t about %.2f a (T/r_s %.4f) after %d right-hand sides: the records so far stand" % (ex, t, t / system.rs, system.n_rhs - n0)
            status = -2
            break
        term = None
        if sol.status == 1:
            cands = [(float(sol.t_events[j][0]), j) for j in range(len(events)) if getattr(events[j], "terminal", False) and sol.t_events[j].size]
            if cands:
                t_ev, j_ev = min(cands)
                term = (t_ev, np.array(sol.y_events[j_ev][0], float), j_ev)
        pts = [(t, y)] + [(float(tt), np.array(sol.y[:, j], float)) for j, tt in enumerate(sol.t)]
        if term is not None:
            pts.append((term[0], term[1]))
        for p_a, p_b in zip(pts[:-1], pts[1:]):
            if p_b[0] > p_a[0] + 1e-14:
                crossings(sol, p_a[0], p_a[1], p_b[0], p_b[1])
        for j, tt in enumerate(sol.t):
            if round(float(tt), 9) in rec_set:
                s = system.snapshot(float(tt), np.array(sol.y[:, j], float), prev_ranks[0])
                prev_ranks[0] = s["ranks"]
                rank_hist.append((float(tt), s["ranks"]))
                recs.append(s)
                states.append(np.array(sol.y[:, j], float))
        if live and sol.t_events[I_FOLD].size:
            folds.extend(float(tf) for tf in sol.t_events[I_FOLD])
            if fold is None:
                tf = float(sol.t_events[I_FOLD][0])
                yf = np.array(sol.sol(tf), float)
                prev_f = next((rk for (tr, rk) in reversed(rank_hist) if tr <= tf + 1e-12), None)  # v1c
                fold = system.snapshot(tf, yf, prev_f)
                fold["y"] = yf
        if sol.status == 0:
            t = float(t1)
            y = np.array(sol.y[:, -1], float)
            break
        if sol.status == -1 or term is None:
            t_last = float(sol.t[-1]) if sol.t.size else t
            label = "the integrator stopped at t %.3f a (T/r_s %.4f): %s" % (t_last, t_last / system.rs, sol.message)
            status = -1
            if sol.t.size:
                t, y = t_last, np.array(sol.y[:, -1], float)
            break
        t_ev, y_ev, j_ev = term
        if live and j_ev == I_FOLD:
            label = "stopped at the fold (min(N - |v|) = 0 first) at T/r_s %.5f" % (t_ev / system.rs)
            t, y = t_ev, y_ev
            break
        if j_ev == I_ESC:
            label = "ESCAPE: a shell reached the grid guard %.1f a at T/r_s %.4f; the run stopped" % (x_guard, t_ev / system.rs)
            status = 2
            t, y = t_ev, y_ev
            break
        if I_K is not None and j_ev == I_K:
            label = ("a shell's momentum reached k_max %g at T/r_s %.4f (the central caustic blueshifts a shell's momentum without bound; the continuum "
                     "comparator's momenta past the lattice band top); the run stopped" % (k_max, t_ev / system.rs))
            status = 3
            t, y = t_ev, y_ev
            break
        restarts += 1
        t = t_ev
        y = settle(t_ev, y_ev, "pin" if j_ev == I_PIN else "unpin")
        if restarts > cap:
            label = "the restart cap %d (pins %d, releases %d) ended the run at T/r_s %.4f" % (cap, system.pins, system.unpins, t / system.rs)
            status = 4
            break
        if stop_at_caustic and caustic is not None:
            label = "stopped at the central caustic at T/r_s %.4f (--after-caustic stop)" % (t / system.rs)
            status = 5
            break
    if label is None:
        label = "completed to T/r_s %.4f" % (t1 / system.rs)
    return {"records": recs, "states": states, "fold": fold, "fold_times": folds, "status": status, "label": label, "escaped": status == 2,
            "nfev": int(system.n_rhs - n0), "seconds": time.time() - t_start, "pins": system.pins, "unpins": system.unpins, "restarts": restarts, "caustic": caustic,
            "ledger": ledger, "ledger_sums": ledger_sums(ledger), "t_end": float(t), "y_end": np.array(y, float), "n_pinned_end": int(system.pinned.sum())}


def run_adaptive(system, y0, T_end, dt_rec, rtol, atol, x_guard, budget_s, k_max=None, stop_at_caustic=False, log=None):
    """DOP853 on the shells with the fields solved at every right-hand side; the fold (min(N - |v|) = 0) located by the event on the dense
    output; the records at the dt_rec clocks; the centre by the pin and release events (v1b)."""
    t_eval = np.arange(0.0, float(T_end) + 1e-9, float(dt_rec))
    out = integrate(system, y0, 0.0, float(T_end), t_eval, rtol, atol, x_guard, budget_s, frozen=None, fold_terminal=False, k_max=k_max, stop_at_caustic=stop_at_caustic, log=log)
    out["mode"] = "adaptive (DOP853, the fields at every right-hand side; the fold by the event on the dense output; the centre by the pin and release events, v1b)"
    out["message"] = out["label"]
    return out


def run_frozen(system, y0, T_end, tau, rtol, atol, x_guard, budget_s, k_max=None, log=None):
    """the fields solved at each record's start and frozen over the step tau (the engine's frozen exponential step's structure); the fold by
    the linear interpolation of the depth between the records (engine_fold_birth's linear clock); the centre as in the adaptive runs (v1b)."""
    t_start = time.time()
    n0 = system.n_rhs
    y = np.array(y0, float, copy=True)
    t = 0.0
    recs = [system.snapshot(0.0, y, None)]
    states = [y.copy()]
    prev = recs[0]["ranks"]
    ledger, restarts, caustic, label, status = [], 0, None, None, 0
    n_steps = int(round(float(T_end) / float(tau)))
    for _ in range(n_steps):
        R, k, _tau = system.unpack(y)
        F = system.fields(R, k)
        frozen = (F["Nbar"].copy(), F["vbar"].copy(), F["u"].copy())
        remaining = budget_s - (time.time() - t_start)
        if remaining <= 1.0:
            label = "ended by the wall-clock budget of %.0f s at T/r_s %.4f: the records so far stand" % (budget_s, t / system.rs)
            status = -2
            break
        out = integrate(system, y, t, t + float(tau), np.zeros(0), rtol, atol, x_guard, remaining, frozen=frozen, k_max=k_max, log=log)
        ledger += out["ledger"]
        restarts += out["restarts"]
        if caustic is None:
            caustic = out["caustic"]
        if out["status"] != 0:
            label, status, t, y = out["label"], out["status"], out["t_end"], out["y_end"]
            break
        y, t = out["y_end"], out["t_end"]
        s = system.snapshot(t, y, prev)
        prev = s["ranks"]
        recs.append(s)
        states.append(y.copy())
    fold = None
    for i, s in enumerate(recs):
        if s["depth"] <= 0.0 and i > 0:
            p = recs[i - 1]
            T_lin = p["t"] + (s["t"] - p["t"]) * p["depth"] / (p["depth"] - s["depth"]) if p["depth"] != s["depth"] else s["t"]
            fold = {"t": T_lin, "T_over_rs": T_lin / system.rs, "linear": True, "depth_R": s["depth_R"], "depth_R_over_2gE0": s["depth_R_over_2gE0"],
                    "depth_R_over_rs": s["depth_R_over_rs"], "E_over_E0": s["E_over_E0"], "count_fraction_inside_depth_R": s["count_fraction_inside_depth_R"],
                    "tau_outer_shell": s["tau_outer_shell"], "tau_inner_shell": s["tau_inner_shell"], "n_pinned": s["n_pinned"], "inversions": None}
            break
    if label is None:
        label = "completed to T/r_s %.4f on frozen-field steps of tau %g a" % (t / system.rs, tau)
    return {"records": recs, "states": states, "fold": fold, "fold_times": [fold["t"]] if fold else [], "escaped": status == 2, "status": status, "label": label, "message": label,
            "nfev": int(system.n_rhs - n0), "seconds": time.time() - t_start,
            "mode": "frozen (the fields solved at each record's start and held over tau %g a; the fold by the depth's linear clock; the centre by the pin and release events, v1b)" % tau,
            "pins": system.pins, "unpins": system.unpins, "restarts": restarts, "caustic": caustic, "ledger": ledger, "ledger_sums": ledger_sums(ledger), "t_end": float(t), "y_end": np.array(y, float),
            "n_pinned_end": int(system.pinned.sum())}


# ------------------------------------------------------------------------------------- input 04's quadrature and canonical check
def quadrature(rg, R0, Rend):
    """input 04's quadrature (WEB40F P20 eqs. 6, 10, re-derived there): t(Rend) = int gamma/z dx, tau(Rend) = int dx/z, z = sqrt(r_g/x -
    r_g/R_0), gamma = (e^2 + b^2)/(e + b z), e = sqrt(1 - r_g/R_0), b = sqrt(r_g/x); x = R_0 - u^2 removes the endpoint singularity."""
    e = math.sqrt(1.0 - rg / R0)

    def dtau_du(u):
        x = R0 - u * u
        return 2.0 * math.sqrt(x * R0 / rg)

    def dt_du(u):
        x = R0 - u * u
        if x <= 0.0:
            return 0.0
        b = math.sqrt(rg / x)
        z = math.sqrt(max(rg / x - rg / R0, 0.0))
        gam = (e * e + b * b) / (e + b * z)
        return gam * 2.0 * math.sqrt(x * R0 / rg)
    U = math.sqrt(max(R0 - Rend, 0.0))
    tau = quad(dtau_du, 0.0, U, epsabs=0.0, epsrel=1e-13, limit=400)[0]
    t = quad(dt_du, 0.0, U, epsabs=0.0, epsrel=1e-13, limit=400)[0]
    return t, tau


def canonical(rg, R0, Rend):
    """input 04's independent check: dR/dt = p/gamma - b, dp/dt = -b p/(2R), dtau/dt = 1/gamma from coordinate rest (p = b/sqrt(1 - b^2))."""
    def rhs(t, y):
        R, p, tau = y
        gam = math.sqrt(1.0 + p * p)
        b = math.sqrt(rg / R)
        return [p / gam - b, -b * p / (2.0 * R), 1.0 / gam]
    ev = lambda t, y: y[0] - Rend  # noqa: E731
    ev.terminal = True
    ev.direction = -1
    b0 = math.sqrt(rg / R0)
    p0 = b0 / math.sqrt(1.0 - b0 * b0)
    s = solve_ivp(rhs, (0.0, 1e6), [R0, p0, 0.0], method="DOP853", rtol=1e-12, atol=1e-12, events=ev)
    if not s.t_events[0].size:
        return None, None
    return float(s.t_events[0][0]), float(s.y_events[0][0][2])


def run_test_shell(E, R0, R_stop, g, m, disp, rtol=1e-12):
    """a single test shell (no charge of its own) released at coordinate rest from R0 in the point exterior of the charge E, advanced by the
    comparator's dynamics (N = 1, v = -sqrt(2gE/R)) to R_stop: (t, tau, k0, the flat clock to the centre by the quadrature)."""
    solver = FieldSolver(Grid(1.0, 4.0), Kernel(1.0, "density"), g, lapse=True, M_c=E)
    sysm = ShellSystem(np.zeros(1), np.array([R0]), disp, solver, g, m, 64.0, vacuum=True)
    k, info, F = release(sysm, "rest")
    y0 = sysm.pack([R0], k, [0.0])
    ev = lambda t, y: y[0] - R_stop  # noqa: E731
    ev.terminal = True
    ev.direction = -1
    sol = solve_ivp(sysm.rhs, (0.0, 1e6), y0, method="DOP853", rtol=rtol, atol=1e-13, events=ev)
    if not sol.t_events[0].size:
        return None, None, float(k[0]), None
    t_s = float(sol.t_events[0][0])
    tau_s = float(sol.y_events[0][0][2])
    return t_s, tau_s, float(k[0]), quadrature(2.0 * g * E, R0, 0.0)[0]


# ------------------------------------------------------------------------------------------------------------ the profile
class LocalSliceReader:
    """the slice's small arrays and the dust columns from an npz of the engine's slice format (np.load; used where slices144_io_v1 does not import)."""

    def __init__(self, path):
        self.path = path
        Z = np.load(path, allow_pickle=False)
        self.Z = Z
        self.T = float(Z["T"])
        self.rs = float(Z["rs"])
        self.R = np.asarray(Z["R"], float).reshape(-1)
        self.nsites = len(self.R)
        kap = np.atleast_1d(Z["kappas"])
        self.kappas = [int(k) for k in kap]
        self.weights = {int(k): float(w) for k, w in zip(kap, np.atleast_1d(Z["weights"]))}
        self.N_applied = np.asarray(Z["N_applied"], float).reshape(-1) if "N_applied" in Z.files else None
        self.v_applied = np.asarray(Z["v_applied"], float).reshape(-1) if "v_applied" in Z.files else None
        self.n_dust = {k: (int(Z["dust_%d" % k].shape[1]) if ("dust_%d" % k) in Z.files else 0) for k in self.kappas}
        self.dtype_str = str(Z["dtype"]) if "dtype" in Z.files else "?"
        self.complex64 = "complex64" in self.dtype_str
        self.members = list(Z.files)

    def dust(self, kappa):
        f = np.asarray(self.Z["f_%d" % kappa], float).reshape(-1)
        d = np.asarray(self.Z["dust_%d" % kappa], dtype=np.complex128)
        return d, f

    def close(self):
        self.Z.close()


def open_slice(path, log):
    if IO is not None:
        try:
            return IO.SliceReader(path, {}, log), "slices144_io_v1.SliceReader"
        except Exception as ex:  # noqa: BLE001
            log("slices144_io_v1.SliceReader failed on %s (%s: %s): the local reader" % (path, type(ex).__name__, ex))
    return LocalSliceReader(path), "local npz reader"


def read_velocity(rd, n_site, m, page_code, n_floor, log):
    """the dust's outward number current through every cut between sites by the range-2 straddling-bond sum of core_well_v1 (2 sum_{a in, b out}
    Im(conj(h_ab) D_ab), D = X diag(f) X^dagger) on h_m = clocked_csr(chain_sparse(R, m, kappa), N_applied, v_applied), weighted 16 kappa, and
    the coordinate velocity u = I/(count per unit length at the cut)."""
    if rd.N_applied is None or rd.v_applied is None:
        return None, "not read (the slice holds no N_applied, v_applied)"
    try:
        if page_code and os.path.isdir(page_code) and page_code not in sys.path:
            sys.path.insert(0, page_code)
        import page_ops_v6 as OPS  # noqa: E402
    except Exception as ex:  # noqa: BLE001
        return None, "not read (page_ops_v6 not importable from %s: %s: %s)" % (page_code, type(ex).__name__, ex)
    R = np.asarray(rd.R, float)
    n = len(R)
    I = np.zeros(n)
    per_k = {}
    rng_max = 0
    for kap in rd.kappas:
        if rd.n_dust.get(kap, 0) == 0:
            continue
        h0, Q = OPS.chain_sparse(R, float(m), int(kap))
        H = OPS.clocked_csr(h0, Q, np.asarray(rd.N_applied, float), np.asarray(rd.v_applied, float)).tocsr().tocoo()
        d, f = rd.dust(kap)
        r, c = H.row, H.col
        sr, sc = r // 2, c // 2
        sel = sc > sr
        if not np.any(sel):
            continue
        hv = np.asarray(H.data, np.complex128)[sel]
        D = (d[r[sel]] * np.conj(d[c[sel]])) @ f
        val = 2.0 * np.imag(np.conj(hv) * D)
        srs, scs = sr[sel], sc[sel]
        rng = scs - srs
        rng_max = max(rng_max, int(rng.max()))
        Ik = np.bincount(srs, weights=val, minlength=n)
        for dd in range(2, int(rng.max()) + 1):
            for sh in range(1, dd):
                m2 = rng == dd
                Ik += np.bincount(srs[m2] + sh, weights=val[m2], minlength=n)
        I += rd.weights[kap] * Ik[:n]
        per_k[int(kap)] = float(np.sum(np.abs(Ik)))
    I_cut = I[:n - 1]
    dens = 0.5 * (n_site[:-1] + n_site[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        u_cut = np.where(dens > n_floor, I_cut / np.where(dens > n_floor, dens, 1.0), np.nan)
    u_site = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        for j in range(n):
            vals = []
            if j - 1 >= 0 and np.isfinite(u_cut[j - 1]):
                vals.append(u_cut[j - 1])
            if j < n - 1 and np.isfinite(u_cut[j]):
                vals.append(u_cut[j])
            if vals:
                u_site[j] = float(np.mean(vals))
    w = n_site > n_floor
    mean_abs = float(np.sum(n_site[w] * np.abs(np.nan_to_num(u_site[w])))) / float(np.sum(n_site[w])) if np.any(w) else None
    return {"I_cut": I_cut, "u_cut": u_cut, "u_site": u_site, "mean_abs_u_count_weighted": mean_abs, "max_abs_u": float(np.nanmax(np.abs(u_site[w]))) if np.any(w) else None,
            "range_sites_max": rng_max, "per_channel_abs_current_sum": per_k,
            "definition": "u at the cut between sites j and j + 1 = the weighted outward dust number current through the cut over (n_j + n_{j+1})/2 (a = 1); per site the mean of its 2 cuts"}, \
        "read (page_ops_v6, the straddling-bond sum; range %d sites)" % rng_max


def column_barrier(R, kappa, u, l):
    """v1d (item 1): L2_c = sum_j r_j^2 t(j) / sum_j (|u_j|^2 + |l_j|^2), t(j) = Re[conj(u_j)(T u)_j + conj(l_j)(T l)_j],
    (T u)_j = (kappa/r_j)^2 u_j + (1/2)[(A_(j+1) - A_j) u_(j+1) - (A_(j-1) - A_j) u_(j-1)], (T l)_j = (kappa/r_j)^2 l_j - (1/2)[the same
    bracket on l], A = kappa/r, open ends (the terms beyond the chain absent). Also <sigma_3> = sum (|u|^2 - |l|^2)/sum (|u|^2 + |l|^2)."""
    R = np.asarray(R, float)
    n = R.size
    A = kappa / R
    up = np.zeros(n)
    dn = np.zeros(n)
    up[:-1] = A[1:] - A[:-1]
    dn[1:] = A[:-1] - A[1:]

    def bracket(x):
        xp = np.zeros(n, complex)
        xm = np.zeros(n, complex)
        xp[:-1] = x[1:]
        xm[1:] = x[:-1]
        return 0.5 * (up * xp - dn * xm)
    a2 = (kappa / R) ** 2
    Tu = a2 * u + bracket(u)
    Tl = a2 * l - bracket(l)
    t = np.real(np.conj(u) * Tu + np.conj(l) * Tl)
    w = np.abs(u) ** 2 + np.abs(l) ** 2
    return float(np.sum(R * R * t) / np.sum(w)), float(np.sum(np.abs(u) ** 2 - np.abs(l) ** 2) / np.sum(w))


def staggered_weight(u, l):
    """v1d: the column's pi-valley weight, the fraction of sum_j |psi_(j+1) - psi_j|^2/... measured as (1 - Re<psi_j, psi_(j+1)>/<psi,psi>)/2
    on the upper and lower components together (0 for a smooth 0-valley column, 1 for a staggered pi-valley column)."""
    s = np.sum(np.real(np.conj(u[:-1]) * u[1:] + np.conj(l[:-1]) * l[1:]))
    w = np.sum(np.abs(u) ** 2 + np.abs(l) ** 2)
    return float(0.5 * (1.0 - s / w))


def read_slice_profile(path, m, page_code, n_floor_rel, log, velocity=True):
    rd, route = open_slice(path, log)
    R = np.asarray(rd.R, float)
    n = len(R)
    n_site = np.zeros(n)
    per_channel = {}
    columns = []  # v1d: per stored column, its count per site and its exact lattice barrier L2_c (item 1)
    for kap in rd.kappas:
        nd = rd.n_dust.get(kap, 0) if hasattr(rd, "n_dust") else 0
        if nd == 0:
            continue
        d, f = rd.dust(kap)
        p = np.abs(d) ** 2
        ns = (p[0::2] + p[1::2]) @ f
        per_channel[int(kap)] = {"columns": int(nd), "count": float(ns.sum()), "weight": float(rd.weights[int(kap)]), "f_min": float(np.min(f)) if f.size else None}
        n_site += float(rd.weights[int(kap)]) * ns
        for c in range(int(nd)):
            uc, lc = d[0::2, c], d[1::2, c]
            wc = np.abs(uc) ** 2 + np.abs(lc) ** 2
            L2c, s3 = column_barrier(R, int(kap), uc, lc)
            columns.append({"kappa": int(kap), "column": c, "f": float(f[c]), "n": float(rd.weights[int(kap)]) * float(f[c]) * wc, "L2": L2c, "sigma3": s3,
                            "w_pi": staggered_weight(uc, lc)})
    tot = float(n_site.sum())
    cum = np.cumsum(n_site) / tot if tot > 0 else np.zeros(n)
    rq = lambda q: float(R[min(int(np.searchsorted(cum, q)), n - 1)])  # noqa: E731
    floor = n_floor_rel * tot
    vel, vel_note = (read_velocity(rd, n_site, m, page_code, floor, log) if velocity else (None, "not read (--no-velocity)"))
    prof = {"path": path, "route": route, "T": float(rd.T), "rs": float(rd.rs), "nsites": n, "R": R, "n_site": n_site, "total": tot, "R_50": rq(0.5), "R_90": rq(0.9), "R_99": rq(0.99),
            "N_applied": (np.asarray(rd.N_applied, float) if rd.N_applied is not None else None), "v_applied": (np.asarray(rd.v_applied, float) if rd.v_applied is not None else None),
            "kappas": list(rd.kappas), "weights": {int(k): float(v) for k, v in rd.weights.items()}, "per_channel": per_channel, "dtype": getattr(rd, "dtype_str", "?"),
            "complex64": bool(getattr(rd, "complex64", False)), "velocity": vel, "velocity_note": vel_note, "n_floor": floor, "shells_above_floor": int(np.sum(n_site > floor)),
            "columns": columns,
            "count_dropped": float(np.sum(n_site[n_site <= floor])), "sigma_from_name": sigma_from_name(path)}
    mt = re.search(r"_ell(\d+)", os.path.basename(path))
    prof["ell_tag"] = int(mt.group(1)) if mt else None
    if hasattr(rd, "close"):
        rd.close()
    log("profile %s (%s; %s): T %.3f, %d sites, kappas %s, dust columns %s, total count %.6f (dropped below the floor %.2e: %.2e), R_50 %.2f R_90 %.2f R_99 %.2f; velocity %s" % (
        os.path.basename(path), route, prof["dtype"], prof["T"], n, prof["kappas"], {k: v["columns"] for k, v in per_channel.items()}, tot, floor, prof["count_dropped"],
        prof["R_50"], prof["R_90"], prof["R_99"], vel_note))
    log("v1d columns (kappa, column: L2_c, <sigma_3>, pi weight, count): %s" % "; ".join(
        "%d,%d: %.6f, %.4f, %.4f, %.4f" % (c["kappa"], c["column"], c["L2"], c["sigma3"], c["w_pi"], float(c["n"].sum())) for c in columns))
    return prof


def sigma_from_name(path):
    mt = re.search(r"_ell(\d+)", os.path.basename(path))
    return float(mt.group(1)) if mt else None


def read_records(path, rs_default):
    with open(path) as f:
        J = json.load(f)
    args = J.get("args", {}) or {}
    rs = float(args.get("rs", rs_default))
    g = float(args.get("g", 0.25))
    recs = sorted([r for r in J.get("records", []) if r.get("E") is not None], key=lambda r: float(r["T"]))
    head = J.get("head", {}) or {}
    r0 = recs[0] if recs else {}
    out = {"path": path, "rs": rs, "g": g, "m": args.get("m"), "ell": args.get("ell"), "tau": args.get("tau"), "head_E0": head.get("E0"), "head_R0": head.get("R0"),
           "E_T0": r0.get("E"), "T0": r0.get("T"), "R_c_T0": r0.get("R_c"), "min_T0": r0.get("min_N_minus_absv"), "n_records": len(recs), "sigma_from_name": sigma_from_name(path)}
    first = next((r for r in recs if r.get("trapping_zeros")), None)
    if first is not None:
        z = float(max(first["trapping_zeros"]))
        E = float(first["E"])
        prev = [r for r in recs if float(r["T"]) < float(first["T"])]
        out.update({"first_T_over_rs": float(first["T"]) / rs, "first_outer": z, "first_outer_over_2gE": z / (2.0 * g * E), "first_E": E, "first_E_over_E0": first.get("E_over_E0"),
                    "first_min": first.get("min_N_minus_absv"), "prev_min": (prev[-1].get("min_N_minus_absv") if prev else None), "prev_T_over_rs": (float(prev[-1]["T"]) / rs if prev else None)})
    return out


def read_fields_T0(path, log):
    try:
        Z = np.load(path, allow_pickle=True)
        Tz = np.asarray(Z["T"], float).reshape(-1)
        k = int(np.argmin(np.abs(Tz)))
        out = {"path": path, "T": float(Tz[k])}
        for key in ("eps_tot", "J_tot", "eps_pairs", "J_pairs"):
            if key in Z.files:
                arr = np.asarray(Z[key][k], float)
                out[key + "_sum"] = float(arr.sum())
        return out
    except Exception as ex:  # noqa: BLE001
        log("fields npz %s not read (%s: %s)" % (path, type(ex).__name__, ex))
        return None


# --------------------------------------------------------------------------------------------------------- the runs
BARRIER_MODES = ("control", "columns", "census", "constant")


def barrier_shells(prof, mode):
    """v1d (item 1): one piece per (column, site) with its count above the delivered floor, no pruning; its L2 by the mode (control 0,
    columns L2_c, census the count-weighted mean of the columns' L2_c at the site within the channel, constant kappa(kappa - 1)); pieces
    with the same site and the same L2 merged (exact: the same Hamiltonian per particle and the same release). Returns R0, n, L2, the
    per-shell site index and the per-shell count per channel."""
    R = prof["R"]
    floor = prof["n_floor"]
    cols = prof["columns"]
    kaps = sorted(set(c["kappa"] for c in cols))
    pieces = []  # (site, kappa, n, L2)
    if mode == "census":
        cen = {}
        for kap in kaps:
            num = np.zeros(R.size)
            den = np.zeros(R.size)
            for c in cols:
                if c["kappa"] == kap:
                    keep = c["n"] > floor
                    num += np.where(keep, c["n"] * c["L2"], 0.0)
                    den += np.where(keep, c["n"], 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                cen[kap] = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)
    for c in cols:
        kap = c["kappa"]
        js = np.nonzero(c["n"] > floor)[0]
        for j in js:
            if mode == "control":
                L2 = 0.0
            elif mode == "columns":
                L2 = c["L2"]
            elif mode == "census":
                L2 = float(cen[kap][j])
            elif mode == "constant":
                L2 = float(kap * (kap - 1))
            else:
                raise ValueError("mode %s" % mode)
            pieces.append((int(j), kap, float(c["n"][j]), float(L2)))
    groups = {}
    for j, kap, nn, L2 in pieces:
        key = (j, L2)
        if key not in groups:
            groups[key] = np.zeros(len(kaps))
        groups[key][kaps.index(kap)] += nn
    keys = sorted(groups.keys())
    site = np.array([k[0] for k in keys], np.int64)
    L2a = np.array([k[1] for k in keys], float)
    nk = np.array([groups[k] for k in keys], float)
    return R[site], nk.sum(axis=1), L2a, site, nk, kaps, len(pieces)


def build_system(prof, sigma, kernel_kind, disp_kind, lapse, g, m, h_arg=None, mode=None):
    R = prof["R"]
    n = prof["n_site"]
    if mode is None:
        sel = n > prof["n_floor"]
        R0, nn = R[sel], n[sel]
        L2 = None
        extra = {}
    else:  # v1d
        R0, nn, L2, site, nk, kaps, npieces = barrier_shells(prof, mode)
        extra = {"site": site, "n_by_kappa": nk, "kappas": kaps, "pieces": npieces}
    h = float(h_arg) if h_arg else min(0.5, 0.5 * float(sigma))
    x_max = 1.5 * float(R0.max()) + 7.0 * float(sigma) + 8.0
    grid = Grid(h, x_max)
    kernel = Kernel(float(sigma), kernel_kind)
    solver = FieldSolver(grid, kernel, g, lapse=lapse)
    disp = Dispersion(disp_kind, m)
    R_pin = min(1e-3, 0.25 * h)  # v1b
    sysm = ShellSystem(nn, R0, disp, solver, g, m, prof["rs"], R_pin=R_pin, L2=L2)
    meta = {"shells": int(nn.size), "h": h, "G": grid.G, "x_max": x_max, "window_points": solver.W, "count": float(nn.sum()), "x_guard": x_max - 7.0 * float(sigma) - 2.0 * h, "R_pin": R_pin}
    if mode is not None:
        sysm.site = extra["site"]
        sysm.n_by_kappa = extra["n_by_kappa"]
        sysm.kappas = extra["kappas"]
        meta.update({"mode": mode, "pieces_column_site": extra["pieces"], "L2_count_weighted_mean": float(np.sum(nn * sysm.L2) / np.sum(nn)),
                     "L2_min": float(sysm.L2.min()), "L2_max": float(sysm.L2.max()),
                     "L2_mean_per_channel": {int(kk): float(np.sum(extra["n_by_kappa"][:, i] * sysm.L2) / np.sum(extra["n_by_kappa"][:, i])) for i, kk in enumerate(extra["kappas"])}})
    return sysm, meta


def fermi_estimate(prof, m):
    """the requirements' pressure-correction order (p_F/m)^2/5 from the read density, 16 states per momentum cell: n_vol = (2/(3 pi^2)) p_F^3; count-weighted."""
    R, n = prof["R"], prof["n_site"]
    w = n > prof["n_floor"]
    if not np.any(w):
        return None
    n_vol = n[w] / (4.0 * math.pi * np.maximum(R[w], 0.5) ** 2)
    pF = (1.5 * math.pi * math.pi * n_vol) ** (1.0 / 3.0)
    corr = (pF / m) ** 2 / 5.0
    return {"pF_over_m_count_weighted": float(np.sum(n[w] * pF) / np.sum(n[w]) / m), "pF_over_m_max": float(pF.max() / m), "pressure_correction_order": float(np.sum(n[w] * corr) / np.sum(n[w])),
            "definition": "an estimate from the read count density with 16 states per momentum (the requirements' (p_F/m)^2/5), not a result"}


def compare_T0_fields(prof, sigma, kernel_kind, g, m, log):
    """the comparator's T = 0 applied fields (release at rest) at every site against the slice's stored N_applied, v_applied."""
    if prof["N_applied"] is None:
        return None
    sysm, meta = build_system(prof, sigma, kernel_kind, "continuum", True, g, m)
    k, info, F = release(sysm, "rest")
    KWs = sysm.solver.kernels(prof["R"])
    N_s, v_s, _, _ = sysm.solver.applied(KWs, F["Nbar"], F["vbar"])
    inside = prof["R"] <= prof["R_99"]
    dN = N_s - prof["N_applied"]
    dv = v_s - prof["v_applied"]
    return {"sigma": sigma, "kernel": kernel_kind, "E0": F["E"], "release_iterations": info["iterations"], "rms_N_inside": float(np.sqrt(np.mean(dN[inside] ** 2))), "max_N_inside": float(np.max(np.abs(dN[inside]))),
            "rms_v_inside": float(np.sqrt(np.mean(dv[inside] ** 2))), "max_v_inside": float(np.max(np.abs(dv[inside]))), "rms_N_all": float(np.sqrt(np.mean(dN ** 2))), "rms_v_all": float(np.sqrt(np.mean(dv ** 2))),
            "N_centre_comparator": float(F["Nbar"][1]), "N_applied_min_slice": float(np.min(prof["N_applied"])), "N_app_min_comparator": float(np.min(N_s)),
            "v_applied_min_slice": float(np.min(prof["v_applied"])), "v_app_min_comparator": float(np.min(v_s)), "N_s": N_s, "v_s": v_s}


def run_spec(spec, prof, a, log):
    t1 = time.time()
    sigma = float(spec["sigma"])
    mode = spec.get("mode")
    sysm, meta = build_system(prof, sigma, spec["kernel"], spec["dispersion"], spec["lapse"] == "on", a.g, a.m, a.h, mode=mode)
    u_read = None
    if spec["release"] == "read":
        if prof["velocity"] is None:
            log("run %s: the read velocity is absent, released at rest instead" % spec["label"])
        elif mode is not None:
            u_read = prof["velocity"]["u_site"][sysm.site]
        else:
            sel = prof["n_site"] > prof["n_floor"]
            u_read = prof["velocity"]["u_site"][sel]
    k, info, F = release(sysm, spec["release"] if u_read is not None else "rest", u_read)
    sysm.E0 = F["E"]
    y0 = sysm.pack(sysm.R0, k, np.zeros(sysm.S))
    T_end = a.T_end * prof["rs"]
    kmx = a.k_max_barrier if (mode is not None and mode != "control") else a.k_max  # v1d: k_max 1000 for the barrier runs
    k_max = float(kmx) if (spec["dispersion"] == "continuum" and kmx and kmx > 0) else None  # v1b
    rest_sums = {"sum_n_eps": float(np.sum(sysm.n * F["eps"])), "sum_n_q": float(np.sum(sysm.n * F["q"])),
                 "sum_n_meff_minus_m": float(np.sum(sysm.n * (np.sqrt(a.m * a.m + F["X"]) - a.m)))}
    if mode is not None:
        log("run %s: mode %s, %d shells from %d (column, site) pieces (count %.6f), L2 count-weighted mean %.4f in [%.4f, %.4f], per channel %s; E0 %.6f, at rest sum n eps %.5f, sum n q %.5f, sum n (m_eff - m) %.5f; k0 in [%.5f, %.5f]; k_max %s" % (
            spec["label"], mode, meta["shells"], meta["pieces_column_site"], meta["count"], meta["L2_count_weighted_mean"], meta["L2_min"], meta["L2_max"],
            {kk: round(v, 4) for kk, v in meta["L2_mean_per_channel"].items()}, F["E"], rest_sums["sum_n_eps"], rest_sums["sum_n_q"], rest_sums["sum_n_meff_minus_m"], float(k.min()), float(k.max()), k_max))
    if spec["step"] == "adaptive":
        out = run_adaptive(sysm, y0, T_end, a.dt_rec, a.rtol, a.atol, meta["x_guard"], a.budget_seconds, k_max=k_max, stop_at_caustic=(a.after_caustic == "stop"), log=log)
    else:
        out = run_frozen(sysm, y0, T_end, spec["tau"], min(a.rtol, 1e-10), min(a.atol, 1e-12), meta["x_guard"], a.budget_seconds, k_max=k_max, log=log)
    recs = out["records"]
    first = next((s for s in recs if s["outer"] is not None), None)
    before = None
    if first is not None:
        i = recs.index(first)
        before = recs[i - 1] if i > 0 else None
    inv_before = int(sum((s["inversions"] or 0) for s in recs if first is None or s["t"] <= first["t"]))
    centre_before = int(max([s["n_center"] for s in recs if first is None or s["t"] <= first["t"]] or [0]))
    fold = out["fold"]
    two_gE0 = 2.0 * a.g * sysm.E0
    E_net = net_drift(recs, out["ledger"], sysm.E0) if recs else None
    res = {"label": spec["label"], "spec": {k_: v_ for k_, v_ in spec.items()}, "meta": meta, "mode": out["mode"], "E0": sysm.E0, "two_gE0": two_gE0, "two_gE0_over_rs": two_gE0 / prof["rs"],
           "release": info, "k0_max": float(np.max(k)), "k0_min": float(np.min(k)), "N_centre_T0": float(F["Nbar"][1]), "max_u_T0": float(np.max(F["u"])), "depth_T0": recs[0]["depth"] if recs else None,
           "fold": None, "first_record": None, "record_before": None, "inversions_before_first_root": inv_before, "centre_crossings_before_first_root": centre_before,
           "E_drift_max": float(np.max(np.abs(np.array([s["E_over_E0"] for s in recs]) - 1.0))) if recs else None, "E_drift_at_end": (recs[-1]["E_over_E0"] - 1.0) if recs else None,
           "E_drift_net_max": E_net,
           "neg_mass_points_max": int(max(s["neg_mass_points"] for s in recs)) if recs else 0, "lapse_ok_all": bool(all(s["lapse_ok"] for s in recs)) if recs else True,
           "nfev": out["nfev"], "status": out["status"], "message": out["message"], "end_label": out["label"], "escaped": out["escaped"], "seconds": time.time() - t1, "T_end_over_rs": a.T_end, "dt_rec": a.dt_rec,
           "centre": {"R_pin": sysm.R_pin, "pins": out["pins"], "unpins": out["unpins"], "restarts": out["restarts"], "n_pinned_end": out["n_pinned_end"],
                      "count_pinned_end": float(sysm.n[sysm.pinned].sum()), "caustic": out["caustic"], "ledger_sums": out["ledger_sums"],
                      "n_pinned_at_first_root": (first["n_pinned"] if first is not None else None), "count_pinned_at_first_root": (first["count_pinned"] if first is not None else None),
                      "ledger_head": out["ledger"][:400], "note": CENTRE_NOTE},
           "records": [{k_: v_ for k_, v_ in s.items() if k_ not in ("ranks", "y")} for s in recs]}
    if fold is not None:
        res["fold"] = {"T_over_rs": fold["T_over_rs"], "t": fold["t"], "R": fold["depth_R"], "R_over_rs": fold["depth_R_over_rs"], "R_over_2gE0": fold["depth_R_over_2gE0"],
                       "charge_fraction_inside": fold["depth_R_over_2gE0"], "count_fraction_inside": fold["count_fraction_inside_depth_R"], "E_over_E0": fold["E_over_E0"],
                       "tau_outer_shell_over_rs": fold["tau_outer_shell"] / prof["rs"], "tau_inner_shell_over_rs": fold["tau_inner_shell"] / prof["rs"], "n_pinned": fold.get("n_pinned"),
                       "inversions_from_last_record": fold.get("inversions"),
                       "located_by": "the depth's linear clock between records" if fold.get("linear") else "the integrator's event on the dense output"}
        if mode is not None and fold.get("y") is not None:  # v1d (item 6): per channel the count inside the fold radius and the innermost radius
            Rf = np.abs(sysm.unpack(fold["y"])[0])
            inside = Rf <= fold["depth_R"]
            pc = {}
            for i, kk in enumerate(sysm.kappas):
                has = sysm.n_by_kappa[:, i] > 0.0
                tot_k = float(sysm.n_by_kappa[:, i].sum())
                pc[int(kk)] = {"count_inside": float(sysm.n_by_kappa[inside, i].sum()), "count_total": tot_k,
                               "fraction_inside": float(sysm.n_by_kappa[inside, i].sum()) / tot_k if tot_k > 0 else None,
                               "innermost_R": float(Rf[has].min()) if has.any() else None,
                               "pinned_count": float(sysm.n_by_kappa[sysm.pinned, i].sum())}
            res["fold"]["per_channel"] = pc
            res["fold"]["count_inside_total"] = float(sysm.n[inside].sum())
            log("run %s: at the fold, per channel (count inside R_f %.3f a of total; innermost R): %s" % (spec["label"], fold["depth_R"], "; ".join(
                "kappa %d: %.3f of %.3f, %.3f a" % (kk, v["count_inside"], v["count_total"], v["innermost_R"] if v["innermost_R"] is not None else float("nan")) for kk, v in pc.items())))
    res["rest_sums"] = rest_sums
    res["barrier_mode"] = mode
    res["L2_meta"] = {k_: v_ for k_, v_ in meta.items() if k_.startswith("L2") or k_ in ("mode", "pieces_column_site")}
    if first is not None:
        res["first_record"] = {"T_over_rs": first["T_over_rs"], "outer": first["outer"], "outer_over_2gE0": first["outer_over_2gE0"], "outer_over_2gE_t": first["outer_over_2gE_t"], "outer_over_rs": first["outer_over_rs"],
                               "inner": first["inner"], "applied_outer": first["applied_outer"], "depth": first["depth"], "E_over_E0": first["E_over_E0"], "n_pinned": first["n_pinned"], "count_pinned": first["count_pinned"]}
    if before is not None:
        res["record_before"] = {"T_over_rs": before["T_over_rs"], "depth": before["depth"], "depth_R": before["depth_R"]}
    ca = out["caustic"] or {}
    log("run %s: shells %d (h %g, G %d), E0 %.6f (2gE0 %.4f a = %.4f r_s), release %d iterations (residual %.1e); %s; fold %s at R %s a (%s of 2gE0); first record %s at %s a; E drift %s (net of the ledger %s); pins %d, releases %d, caustic %s; nfev %d; %.0f s; %s" % (
        spec["label"], meta["shells"], meta["h"], meta["G"], sysm.E0, two_gE0, two_gE0 / prof["rs"], info["iterations"], info["velocity_residual"], out["mode"].split(" (")[0],
        fx((res["fold"] or {}).get("T_over_rs"), "%.5f"), fx((res["fold"] or {}).get("R"), "%.3f"), fx((res["fold"] or {}).get("R_over_2gE0"), "%.4f"),
        fx((res["first_record"] or {}).get("T_over_rs"), "%.4f"), fx((res["first_record"] or {}).get("outer"), "%.3f"), fx(res["E_drift_max"], "%.1e"), fx(E_net, "%.1e"),
        out["pins"], out["unpins"], fx(ca.get("T_over_rs"), "%.4f"), out["nfev"], res["seconds"], out["label"]))
    arrays = {"T_over_rs": np.array([s["T_over_rs"] for s in recs]), "E": np.array([s["E"] for s in recs]), "depth": np.array([s["depth"] for s in recs]), "depth_R": np.array([s["depth_R"] for s in recs]),
              "outer": np.array([(s["outer"] if s["outer"] is not None else np.nan) for s in recs]), "inner": np.array([(s["inner"] if s["inner"] is not None else np.nan) for s in recs]),
              "applied_outer": np.array([(s["applied_outer"] if s["applied_outer"] is not None else np.nan) for s in recs]), "inversions": np.array([(s["inversions"] if s["inversions"] is not None else 0) for s in recs]),
              "n_center": np.array([s["n_center"] for s in recs]), "R_50": np.array([(s["R_50"] if s["R_50"] is not None else np.nan) for s in recs]), "n_pinned": np.array([s["n_pinned"] for s in recs]),
              "shell_R0": sysm.R0, "shell_n": sysm.n, "shell_k0": k, "shell_pinned_end": sysm.pinned, "shell_L2": sysm.L2,
              "ledger_t": np.array([e["t"] for e in out["ledger"]]), "ledger_dE_exact": np.array([e["dE_exact"] for e in out["ledger"]])}
    return res, arrays


def build_matrix(sigmas, matrix, have_velocity, a):
    specs = []

    def add(sigma, kernel, dispersion, release_, lapse, step, tau=None):
        lab = "s%g_%s_%s_%s_lapse-%s_%s" % (sigma, kernel, dispersion, release_, lapse, step if step == "adaptive" else "frozen-tau%g" % tau)
        specs.append({"label": lab, "sigma": sigma, "kernel": kernel, "dispersion": dispersion, "release": release_, "lapse": lapse, "step": step, "tau": tau})
    if matrix == "single":
        for s in sigmas:
            add(s, a.kernel, a.dispersion, a.release, a.lapse, a.step, a.tau)
        return specs
    for s in sigmas:
        add(s, "density", "continuum", "rest", "on", "adaptive")
    if matrix == "primary":
        return specs
    for s in sigmas:
        add(s, "density", "continuum", "rest", "off", "adaptive")
    for s in sigmas:
        add(s, "density", "sine", "rest", "on", "adaptive")
    s4 = 4.0 if 4.0 in sigmas else sigmas[0]
    if matrix == "full":
        for s in sigmas:
            for kern in ("radial", "vector"):
                for disp in ("continuum", "sine"):
                    for lap in ("on", "off"):
                        add(s, kern, disp, "rest", lap, "adaptive")
    else:
        add(s4, "radial", "continuum", "rest", "on", "adaptive")
        add(s4, "vector", "continuum", "rest", "on", "adaptive")
    if have_velocity:
        add(s4, "density", "continuum", "read", "on", "adaptive")
    add(s4, "density", "continuum", "rest", "on", "frozen", a.tau)
    add(s4, "density", "continuum", "rest", "on", "frozen", 0.5 * a.tau)
    add(0.25, "density", "continuum", "rest", "on", "adaptive")
    return specs


# ------------------------------------------------------------------------------------------------------------- the md
def compose(args, profs, field_rows, single_rows, results, records_read, fields_T0, out_dir, log):
    rs = profs[0]["rs"] if profs else args.rs
    L = ["# %s: the cold dust comparator on the production's own T = 0 profile (%s; W4 (a))" % (TITLE, CODE), "",
         "The comparator: the pressureless shell system of the model's spherical chart (flat slices; N and v from the 2 constraint rows on the kernel-averaged shell totals, M' = s_rho + w s_J with w = -sqrt(2gM/x) and (ln N)' = -g s_J/(x w), N(infinity) = 1; the applied fields the transpose of the source averaging; each shell's Hamiltonian h = N eps + v q, released at coordinate rest; the shells crossing freely with the charge inside each recomputed from their positions at every right-hand side; the root the zero of N - |v| = N(1 - sqrt(2gM/x)) on the dense fields, the fold clock by the integrator's event). The variants: lapse off (every shell with N = 1 in the Painleve-Gullstrand field of the averaged charge inside it, input 04's control generalized), the frozen-field step of tau, the sine dispersion, the radial and vector kernels, the read release, the narrow kernel. The module docstring carries the full definition.",
         "", CENTRE_NOTE + ".", "", PROVENANCE + ".", "", "## The T = 0 profile", ""]
    for p in profs:
        L.append("- %s (%s; %s; tag ell %s): T %.4f (T/r_s %.4f), r_s %g a, %d sites, kappas %s with weights %s, dust columns per channel %s; the total count %.6f (%d shells above the floor %.2e, %.2e dropped), R_50 %.3f, R_90 %.3f, R_99 %.3f a; velocity: %s%s." % (
            p["path"], p["route"], p["dtype"], p["ell_tag"], p["T"], p["T"] / p["rs"], p["rs"], p["nsites"], p["kappas"], p["weights"], {k: v["columns"] for k, v in p["per_channel"].items()}, p["total"],
            p["shells_above_floor"], p["n_floor"], p["count_dropped"], p["R_50"], p["R_90"], p["R_99"], p["velocity_note"],
            ("; the count-weighted mean |u| %.3e, max |u| %.3e (the coordinate velocity; 0 for a coordinate-rest preparation)" % (p["velocity"]["mean_abs_u_count_weighted"], p["velocity"]["max_abs_u"])) if p["velocity"] else ""))
        fe = fermi_estimate(p, args.m)
        if fe:
            L.append("  the read density's Fermi momentum: p_F/m %.4f (count-weighted; max %.4f), the pressure correction's order (p_F/m)^2/5 = %.4f (%s)." % (fe["pF_over_m_count_weighted"], fe["pF_over_m_max"], fe["pressure_correction_order"], fe["definition"]))
    if len(profs) > 1:
        base = profs[0]["n_site"]
        for p in profs[1:]:
            if p["n_site"].shape == base.shape:
                L.append("- the counts of %s against %s: largest per-site difference %.3e, total difference %.3e." % (os.path.basename(p["path"]), os.path.basename(profs[0]["path"]), float(np.max(np.abs(p["n_site"] - base))), p["total"] - profs[0]["total"]))
    for r in records_read:
        L.append("- records json %s (ell %s, tau %s): E at T = 0 %s (head E0 %s, head R0 %s), R_c %s, min(N - |v|) %s; the first record with a trapping zero: T/r_s %s, outer zero %s a, over the record's 2gE %s (E %s, E/E0 %s), min %s (the record before: T/r_s %s, min %s)." % (
            r["path"], r["ell"], r["tau"], fx(r["E_T0"], "%.6f"), fx(r["head_E0"], "%.6f"), fx(r["head_R0"], "%.3f"), fx(r["R_c_T0"], "%.3f"), fx(r["min_T0"], "%+.5f"), fx(r.get("first_T_over_rs"), "%.4f"),
            fx(r.get("first_outer"), "%.4f"), fx(r.get("first_outer_over_2gE"), "%.4f"), fx(r.get("first_E"), "%.5f"), fx(r.get("first_E_over_E0"), "%.5f"), fx(r.get("first_min"), "%+.5f"), fx(r.get("prev_T_over_rs"), "%.4f"), fx(r.get("prev_min"), "%+.5f")))
    for f in fields_T0:
        if f:
            L.append("- fields npz %s at T %.3f: sum eps_tot %s, sum eps_pairs %s, sum J_tot %s, sum J_pairs %s (the record's shell totals; the comparator's sum n eps and sum n q at rest print per run below)." % (
                f["path"], f["T"], fx(f.get("eps_tot_sum"), "%.6f"), fx(f.get("eps_pairs_sum"), "%.6f"), fx(f.get("J_tot_sum"), "%+.6f"), fx(f.get("J_pairs_sum"), "%+.6f")))
    L += ["", "### The profile by bins of 16 sites (the first profile; count, cumulative fraction, read velocity, the slice's applied fields)", "",
          "| R range (a) | count | cumulative | mean u read | N_applied (slice) at the bin's middle | v_applied (slice) |", "|---|---|---|---|---|---|"]
    if profs:
        p = profs[0]
        R, n = p["R"], p["n_site"]
        cum = np.cumsum(n) / p["total"]
        jmax = int(np.searchsorted(R, p["R_99"] + 32.0))
        for lo in range(0, min(jmax + 16, len(R)), 16):
            hi = min(lo + 16, len(R))
            mid = (lo + hi) // 2
            ub = (p["velocity"]["u_site"][lo:hi] if p["velocity"] else None)
            L.append("| %.1f to %.1f | %.5f | %.5f | %s | %s | %s |" % (R[lo], R[hi - 1], float(n[lo:hi].sum()), float(cum[hi - 1]), fx(float(np.nanmean(ub)), "%+.3e") if (ub is not None and np.any(np.isfinite(ub))) else "-",
                                                                   fx(float(p["N_applied"][mid]), "%.5f") if p["N_applied"] is not None else "-", fx(float(p["v_applied"][mid]), "%+.5f") if p["v_applied"] is not None else "-"))
    L += ["", "## The comparator's T = 0 applied fields against the slice's stored N_applied, v_applied (release at rest; which kernel the engine applies is read off this table by the lead)", "",
          "| slice (ell tag) | sigma/a | kernel | E_0 | RMS dN inside R_99 | max |dN| | RMS dv | max |dv| | RMS dN all sites | RMS dv all | N_app min: comparator; slice | v_app min: comparator; slice | N centre (dense) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in field_rows:
        c = row["cmp"]
        L.append("| %s (%s) | %g | %s | %.6f | %.3e | %.3e | %.3e | %.3e | %.3e | %.3e | %.5f; %.5f | %+.5f; %+.5f | %.5f |" % (
            os.path.basename(row["slice"]), row["ell_tag"], c["sigma"], c["kernel"], c["E0"], c["rms_N_inside"], c["max_N_inside"], c["rms_v_inside"], c["max_v_inside"], c["rms_N_all"], c["rms_v_all"],
            c["N_app_min_comparator"], c["N_applied_min_slice"], c["v_app_min_comparator"], c["v_applied_min_slice"], c["N_centre_comparator"]))
    L += ["", "## The formation: the comparator beside the records", "",
          "Columns: the fold clock (min(N - |v|) = 0 first; the integrator's event for the adaptive runs, the depth's linear clock for the frozen steps), the first record clock holding a dense root on the %g a record grid (the records' convention), the fold's radius R_f in a, over the nominal r_s %g a and over 2gE_0 (the comparator's own charge; at a dense root this is also the fraction of E inside it), the count fraction inside, the first record's outer root and its radius over 2gE_0, the depth min(N - |v|) at the record before the first root, E(T_f)/E_0 - 1 (0 to the integrator's tolerance for the adaptive lapse-on runs; the lapse-off and frozen variants drift), the shell-order inversions and the centre crossings through the first root record, the shells pinned at the centre at the first root (v1b) and the caustic clock (the first pin at a trapped centre), the applied outer root at the first record, the shells, the right-hand sides and the seconds; beside them the records' first root (%s) and, where a records json was given, the read values." % (args.dt_rec, rs, QUOTED_SOURCE), "",
          "| run | sigma/a | kernel | dispersion | release | lapse | step | fold T/r_s | first record T/r_s | R_f (a) | R_f/r_s | R_f/(2gE_0) = charge inside | count inside | first record's root (a); /(2gE_0) | depth before | E(T_f)/E_0 - 1 | inversions; centre crossings | pinned at first root (shells; count); caustic T/r_s | applied outer at first record | shells | nfev | s | records: first root T/r_s; R/(2gE) quoted | read |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        sp, fo, fr, rb = r["spec"], r["fold"] or {}, r["first_record"] or {}, r["record_before"] or {}
        q = QUOTED_RECORDS.get(float(sp["sigma"]))
        rd = next((x for x in records_read if x.get("sigma_from_name") == float(sp["sigma"]) or (x.get("ell") is not None and float(x["ell"]) == float(sp["sigma"]))), None)
        ce = r.get("centre") or {}
        ca = ce.get("caustic") or {}
        L.append("| %s | %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s | %s | %d; %d | %s; %s; %s | %s | %d | %d | %.0f | %s | %s |" % (
            r["label"], sp["sigma"], sp["kernel"], sp["dispersion"], sp["release"], sp["lapse"], ("adaptive" if sp["step"] == "adaptive" else "frozen tau %g a" % sp["tau"]),
            fx(fo.get("T_over_rs"), "%.5f") if fo else ("none through %g r_s" % r["T_end_over_rs"]), fx(fr.get("T_over_rs"), "%.4f"), fx(fo.get("R"), "%.3f"), fx(fo.get("R_over_rs"), "%.4f"), fx(fo.get("R_over_2gE0"), "%.4f"),
            fx(fo.get("count_fraction_inside"), "%.4f"), fx(fr.get("outer"), "%.3f"), fx(fr.get("outer_over_2gE0"), "%.4f"), fx(rb.get("depth"), "%+.5f"), fx((fo.get("E_over_E0") - 1.0) if fo.get("E_over_E0") is not None else None, "%+.2e"),
            r["inversions_before_first_root"], r["centre_crossings_before_first_root"], fx(ce.get("n_pinned_at_first_root"), "%d"), fx(ce.get("count_pinned_at_first_root"), "%.3f"), fx(ca.get("T_over_rs"), "%.4f"),
            fx(fr.get("applied_outer"), "%.3f"), r["meta"]["shells"], r["nfev"], r["seconds"],
            ("%.3f; %.2f" % q) if q else "-", ("%.4f; %.4f" % (rd["first_T_over_rs"], rd["first_outer_over_2gE"])) if (rd and rd.get("first_T_over_rs") is not None) else "-"))
    L += ["", "Every run: E_0, the release (iterations, the velocity residual, shells without a rest momentum), k_0 range, N at the centre and max |w| at T = 0, the depth at T = 0, the charge's largest drift over the records (raw, and net of the centre ledger), negative-mass clamps, the lapse recursion's state, the surface and centre shells' proper times at the fold over r_s, the centre's ledger (pins, releases, restarts, centre crossings with the exact jump sum), the caustic, and the end:", ""]
    for r in results:
        fo = r["fold"] or {}
        ce = r.get("centre") or {}
        ls = ce.get("ledger_sums") or {}
        ca = ce.get("caustic") or {}
        L.append("- %s: E_0 %.6f (2gE_0 %.4f a = %.4f r_s; the binding E_0/(N m) %.5f), release %d iterations, residual %.1e, %d shells without rest; k_0 in [%.5f, %.5f]; N(centre) %.5f, max |w| %.5f, depth %.5f at T = 0; E drift max %s (end %s; net of the ledger %s); negative-mass clamps max %d; lapse ok %s; at the fold tau(surface shell)/r_s %s, tau(centre shell)/r_s %s; pins %s, releases %s, restarts %s, pinned at the end %s (count %s), centre crossings %s with the exact jump sum %s (first order %s), the pin displacements' sum %s; the caustic at T/r_s %s (|v_app(0)| %s against N_app(0) %s); %s; %s." % (
            r["label"], r["E0"], r["two_gE0"], r["two_gE0_over_rs"], r["E0"] / (r["meta"]["count"] * args.m), r["release"]["iterations"], r["release"]["velocity_residual"], r["release"]["shells_without_rest"], r["k0_min"], r["k0_max"],
            r["N_centre_T0"], r["max_u_T0"], r["depth_T0"], fx(r["E_drift_max"], "%.2e"), fx(r["E_drift_at_end"], "%+.2e"), fx(r.get("E_drift_net_max"), "%.2e"), r["neg_mass_points_max"], r["lapse_ok_all"], fx(fo.get("tau_outer_shell_over_rs"), "%.4f"), fx(fo.get("tau_inner_shell_over_rs"), "%.4f"),
            fx(ce.get("pins"), "%d"), fx(ce.get("unpins"), "%d"), fx(ce.get("restarts"), "%d"), fx(ce.get("n_pinned_end"), "%d"), fx(ce.get("count_pinned_end"), "%.3f"), fx(ls.get("crossings"), "%d"), fx(ls.get("dE_crossings_exact"), "%+.3e"), fx(ls.get("dE_crossings_first_order"), "%+.3e"), fx(ls.get("dE_pins_exact"), "%+.3e"),
            fx(ca.get("T_over_rs"), "%.4f"), fx(abs(ca["v0"]) if ca.get("v0") is not None else None, "%.4f"), fx(ca.get("N0"), "%.4f"), r["mode"], r.get("end_label", "status %d" % r["status"])))
    L += ["", "## The point-exterior single shell (input 04's control with the profile's charge: the outermost shell with all the charge inside it, N = 1)", "",
          "| release radius R_0 (a) | charge E (comparator's E_0) | r_g = 2gE (a) | r_g/r_s | t(r_g)/r_s | tau(r_g)/r_s | t/r_g | flat clock to the centre t(0)/r_s | canonical check t, tau (a) | comparator's test shell t, tau (a) | sine-dispersion test shell t/r_s | P20 table |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in single_rows:
        L.append("| %s: %.4f | %.6f | %.4f | %.4f | %.6f | %.6f | %.6f | %.6f | %s, %s | %s, %s | %s | %s |" % (
            s["label"], s["R0"], s["E"], s["rg"], s["rg"] / rs, s["t"] / rs, s["tau"] / rs, s["t"] / s["rg"], s["t_centre"] / rs, fx(s["t_can"], "%.6f"), fx(s["tau_can"], "%.6f"), fx(s["t_shell"], "%.6f"), fx(s["tau_shell"], "%.6f"),
            fx((s["t_sine"] / rs) if s.get("t_sine") else None, "%.6f"), s.get("table", "-")))
    L += ["", "## The root's history after formation (every record to the end; the adaptive lapse-on runs; the shells pinned at the centre per record, v1b)", "",
          "| run | T/r_s | depth min(N - |v|) | dense outer (a) | inner | outer/(2gE_0) | outer/(2gE(t)) | applied outer | E/E_0 - 1 | centre crossings | inversions in the step | pinned (shells; count) | R_50 (a) | max |w| | N centre |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r["spec"]["lapse"] != "on" or r["spec"]["step"] != "adaptive":
            continue
        for s in r["records"]:
            if s["outer"] is None and s["depth"] > 0.05:
                continue
            L.append("| %s | %.4f | %+.5f | %s | %s | %s | %s | %s | %+.2e | %d | %s | %d; %.3f | %s | %.4f | %s |" % (
                r["label"], s["T_over_rs"], s["depth"], fx(s["outer"], "%.3f"), fx(s["inner"], "%.3f"), fx(s["outer_over_2gE0"], "%.4f"), fx(s["outer_over_2gE_t"], "%.4f"), fx(s["applied_outer"], "%.3f"), s["E_over_E0"] - 1.0, s["n_center"],
                fx(s["inversions"], "%d"), s.get("n_pinned", 0), s.get("count_pinned", 0.0), fx(s["R_50"], "%.2f"), s["max_u"], fx(s["N_centre"], "%.5f")))
    L += ["", "## Notes", "",
          "- the units: a = 1, g = %g, m a = %g; T/r_s with the slice's r_s %g a; every radius in a and over 2gE_0 (the comparator's conserved charge; the records' column divides by the record's own E(t), which the engine's collar charge drifts, so the 2 normalizations are printed side by side where both exist);" % (args.g, args.m, rs),
          "- the recursion route: %s; the slice reader: %s; the integrator: DOP853 at rtol %g, atol %g; the kernel windows %g sigma; the wall-clock budget per run %g s, the momentum guard k_max %g (the continuum dispersion), --after-caustic %s (v1b);" % (RECURSION_ROUTE, IO_NOTE, args.rtol, args.atol, 7.0, args.budget_seconds, args.k_max, args.after_caustic),
          "- at a dense root 2gM(R) = R, so the charge inside the root over E is the radius over 2gE by identity: the columns coincide for the comparator and are printed once;",
          "- the centre (v1b): " + CENTRE_NOTE + "; a run's end label says whether it completed, stopped at the caustic, at the grid guard, at k_max, at the restart cap or at the budget;",
          "- the outermost shell of an averaged profile is not on the point-exterior quadrature of the full charge (v1c, self-test 3b): its applied fields read the charge inside it as E minus half its own smeared mass minus the inner tails beyond it, and a clock below 1, so its clock lags the quadrature by about half that deficit; the quadrature is the exact comparator only beyond the kernel reach (the probe of self-test 3b-i);",
          "- W4 (b), the engine from T = 0 to 14 with the pair sources off and on the conserving step, is the engine's GPU run and not this script's;",
          "- every number above is printed by the script; the reading is the lead's."]
    return L


# ------------------------------------------------------------------------------------------------------ the self-test
def ball_system(Rsh, n, sigma, h, g, m, probe_R=None):
    """v1c: the uniform ball's shell system (the density kernel, the continuum dispersion, the lapse on) with an optional massless probe shell
    (n = 0) appended at probe_R, the grid sized for it."""
    R0 = np.asarray(Rsh, float)
    nn = np.asarray(n, float)
    if probe_R is not None:
        R0 = np.concatenate([R0, [float(probe_R)]])
        nn = np.concatenate([nn, [0.0]])
    x_max = 1.5 * float(R0.max()) + 7.0 * float(sigma) + 8.0
    grid = Grid(h, x_max)
    solver = FieldSolver(grid, Kernel(float(sigma), "density"), g, lapse=True)
    sysm = ShellSystem(nn, R0, Dispersion("continuum", m), solver, g, m, 64.0, R_pin=min(1e-3, 0.25 * h))
    return sysm, {"h": h, "G": grid.G, "x_max": x_max, "x_guard": x_max - 7.0 * float(sigma) - 2.0 * h}


def self_test(a):
    log = Logger(os.path.join(a.out, "dust_comparator_v1c_selftest.log"))
    g, m = float(a.g), float(a.m)
    items = []

    def rec(name, passed, detail):
        items.append({"name": name, "pass": bool(passed), "detail": detail})
        log("%s: %s: %s" % ("PASS" if passed else "FAIL", name, detail))
    # 1 the quadrature against input 04's P20 table and canonical check
    for (rg, R0), (t_tab, tau_tab) in P20_TABLE.items():
        t, tau = quadrature(rg, R0, rg)
        rec("1 quadrature P20 table r_g %g R_0 %g" % (rg, R0), abs(t / 64.0 - t_tab) < 2e-9 and abs(tau / 64.0 - tau_tab) < 2e-9,
            "t/64a %.10f against %.10f, tau/64a %.10f against %.10f" % (t / 64.0, t_tab, tau / 64.0, tau_tab))
    t, tau = quadrature(59.05, 256.0, 59.05)
    tc, tauc = canonical(59.05, 256.0, 59.05)
    rec("1 canonical check r_g 59.05", tc is not None and abs(tc / t - 1) < 1e-8 and abs(tauc / tau - 1) < 1e-8 and abs(tc - P20_CANONICAL[0]) < 1e-6,
        "canonical t %.10f tau %.10f against the quadrature's %.10f %.10f and input 04's %.10f %.10f" % (tc, tauc, t, tau, P20_CANONICAL[0], P20_CANONICAL[1]))
    # 2 the test shell by the comparator's dynamics
    E = 59.05 / (2.0 * g)
    t_s, tau_s, k0, t_centre = run_test_shell(E, 256.0, 59.05, g, m, Dispersion("continuum", m))
    t_centre64 = quadrature(64.0, 256.0, 0.0)[0]  # v1c: the flat clock to the centre at r_g 64 a, the review's 13.42
    rec("2 test shell in the point exterior (comparator dynamics)", t_s is not None and abs(t_s / t - 1) < 1e-8 and abs(tau_s / tau - 1) < 1e-8,
        "t %.10f tau %.10f against the quadrature's %.10f %.10f; k_0 %.6f; the flat clock to the centre %.6f r_s at r_g 59.05 a and %.6f r_s at r_g 64 a (the review's 13.42 is the latter, the nominal charge)" % (
            t_s or 0, tau_s or 0, t, tau, k0, t_centre / 64.0, t_centre64 / 64.0))
    t_sine, tau_sine, _, _ = run_test_shell(E, 256.0, 59.05, g, m, Dispersion("sine", m))
    log("     the sine-dispersion test shell: t/r_s %.6f (continuum %.6f; X4's dispersion-corrected value is stated near 13.17), tau/r_s %.6f" % (t_sine / 64.0, t_s / 64.0, tau_sine / 64.0))
    # 3 the uniform ball, run to its fold (v1b), with a massless probe beyond the kernel reach (v1c)
    NB, R0 = 640.0, 256.0
    S = 256 if a.self_test_fine else 128
    sigma = 0.5 if a.self_test_fine else 1.0
    dR = R0 / S
    Rsh = (np.arange(S) + 0.5) * dR
    n = NB * Rsh ** 2 / np.sum(Rsh ** 2)

    def dM(r, M):
        return [3.0 * NB * m * r * r / R0 ** 3 * math.sqrt(max(1.0 - 2.0 * g * M[0] / r, 0.0))]
    ex = solve_ivp(dM, (1e-9, R0), [0.0], method="DOP853", rtol=1e-12, atol=1e-14)
    E_exact = float(ex.y[0, -1])
    x_ser = 2.0 * g * E_exact / R0
    series = 1.0 - 0.3 * x_ser - (3.0 / 56.0) * x_ser ** 2
    i_surf = int(np.argmax(Rsh))
    R_surf = float(Rsh[i_surf])
    R_probe = R_surf + 16.0 * sigma  # v1c: beyond the kernel reach (7 sigma each way) of every massive shell: its window and the ball's deposit are disjoint
    sysm, meta = ball_system(Rsh, n, sigma, 0.5 * sigma, g, m, probe_R=R_probe)
    i_probe = sysm.S - 1
    k, info, F = release(sysm, "rest")
    sysm.E0 = F["E"]
    E0 = float(F["E"])
    rec("3a uniform ball: the dust constraint's charge", abs(E0 / E_exact - 1.0) < 1e-3,
        "comparator E_0 %.6f (S %d shells, sigma %g, h %g; a massless probe shell at R_surf + 16 sigma = %.1f a adds no charge) against the sharp ball's ODE %.6f (relative %.2e); the binding E/(N_B m) %.6f against the series 1 - (3/10)x - (3/56)x^2 = %.6f at x = r_s/R_0 %.4f; release %d iterations, residual %.1e" % (
            E0, S, sigma, meta["h"], R_probe, E_exact, E0 / E_exact - 1.0, E0 / (NB * m), series, x_ser, info["iterations"], info["velocity_residual"]))
    rg = 2.0 * g * E0
    tq, tauq = quadrature(rg, R_surf, rg)
    x = sysm.solver.grid.x
    # v1c: the surface shell's release fields, its first-order lag prediction and the kernel-read deficits
    eps_s, q_s, N_s = float(F["eps"][i_surf]), float(F["q"][i_surf]), float(F["N_app"][i_surf])
    w_s = float(F["v_app"][i_surf]) / N_s
    force_s = float(F["dN"][i_surf]) * eps_s + float(F["dv"][i_surf]) * q_s
    a_s = -force_s * N_s * m * m / eps_s ** 3
    a_0 = -(g * E0 / R_surf ** 2) * (1.0 - rg / R_surf)
    ratio_a = a_s / a_0
    L_t_pred = 1.0 - math.sqrt(max(ratio_a, 0.0))
    eps_0 = m / math.sqrt(1.0 - rg / R_surf)
    L_tau_pred = N_s * m / eps_s - (m / eps_0) * math.sqrt(max(ratio_a, 0.0))
    M_K0 = float(sysm.kernel_read(F, F["M"])[i_surf])
    delta_K0, nu_K0 = 1.0 - M_K0 / E0, 1.0 - N_s
    NMx2 = np.zeros_like(x)
    NMx2[1:] = F["Nbar"][1:] * F["M"][1:] / x[1:] ** 2
    ratio_ident = float(sysm.kernel_read(F, NMx2)[i_surf]) * N_s * (1.0 - w_s * w_s) / ((E0 / R_surf ** 2) * (1.0 - rg / R_surf))
    mu0, _ = sysm.shell_masses(F)
    Rabs0 = np.abs(sysm.R0)
    Phi0 = cumulative_kernels(sysm, Rabs0)
    tail_nb0 = 1.0 - float(Phi0[i_surf - 1, i_surf]) if i_surf >= 1 else 0.0
    log("     the surface shell at release: the kernel reads M_K/E_0 = %.5f (deficit %.4e: its own mu_s/E_0 %.4e, half of it outside; the neighbour at 2 sigma has %.4f of its mass beyond R_s) and N_app %.6f (deficit %.3e); a/a_0 = %.6f from the release forces, %.6f from the identity <N M/x^2>_K N_app (1 - w_app^2)/[(E_0/R_s^2)(1 - 2gE_0/R_s)]; the predicted early lag (t - t_q)/t = %.4e, (tau - tau_q)/t = %.4e" % (
        M_K0 / E0, delta_K0, mu0[i_surf] / E0, tail_nb0, N_s, nu_K0, ratio_a, ratio_ident, L_t_pred, L_tau_pred))
    t_ball = 1.12 * tq
    y0 = sysm.pack(sysm.R0, k, np.zeros(sysm.S))
    log("     3b to 3e: the ball to its fold (terminal) or to %.3f r_s, records every 8 a, rtol 1e-10, budget %.0f s" % (t_ball / 64.0, a.budget_test))
    out = integrate(sysm, y0, 0.0, t_ball, np.arange(0.0, t_ball + 1e-9, 8.0), 1e-10, 1e-12, meta["x_guard"], a.budget_test, fold_terminal=True, k_max=20.0, log=log)
    recs, states, fold, ledger = out["records"], out["states"], out["fold"], out["ledger"]
    T_inv = next((s["t"] for s in recs if (s["inversions"] or 0) > 0), None)
    inv_fold = int((fold or {}).get("inversions") or 0)
    inv_total = int(sum((s["inversions"] or 0) for s in recs)) + inv_fold
    pre = [(s, yv) for s, yv in zip(recs, states) if s["t"] > 0.0 and (T_inv is None or s["t"] < T_inv - 1e-9)]
    # 3b-i the probe (v1c): Birkhoff beyond the kernel reach, exact up to the vacuum averaging
    probe = []
    for s, yv in zip(recs, states):
        if s["t"] < 64.0:
            continue
        R_, k_, tau_ = sysm.unpack(yv)
        Rp = float(abs(R_[i_probe]))
        if Rp >= R_probe:
            continue
        tq_R, tauq_R = quadrature(rg, R_probe, Rp)
        probe.append((s["t"], Rp, abs(tq_R - s["t"]) / s["t"], abs(tauq_R - float(tau_[i_probe])) / s["t"]))
    if probe:
        Rp_min = min(p[1] for p in probe)
        eps_vac = sigma ** 2 / (8.0 * Rp_min ** 2)
        bound_p = 5.0 * eps_vac + 2e-6
        worst_p = max(max(p[2], p[3]) for p in probe)
        rec("3b-i uniform ball: a massless probe beyond the kernel reach moves on the point-exterior quadrature of the full charge (Birkhoff in the chart)", worst_p < bound_p,
            "the probe released at rest at R_surf + 16 sigma = %.1f a (the kernel reach 7 sigma each way: its window and the ball's deposit are disjoint, so it reads M = E_0 and N = 1 exactly); largest |t_quad(R_p) - t|/t and |tau_quad(R_p) - tau_p|/t over the %d records from 1 r_s to the fold: %.2e against the bound 5 sigma^2/(8 R_p,min^2) + 2e-6 = %.2e at R_p,min %.2f a (the density kernel's vacuum average of x^(-1/2) reduces the flow by sigma^2/(8R^2) and the force by 5 times that, so the lag is at most 5 times it; the integrator's share at rtol 1e-10 below 2e-7); the probe's radius at the fold %.2f a; %s" % (
                R_probe, len(probe), worst_p, bound_p, Rp_min, probe[-1][1], out["label"]))
    else:
        rec("3b-i uniform ball: a massless probe beyond the kernel reach moves on the point-exterior quadrature of the full charge (Birkhoff in the chart)", False, "no probe record from 1 r_s on (%s)" % out["label"])
    # 3b-ii the outermost massive shell's lag against its first-order prediction (v1c); its late lag printed with the kernel-read deficits
    devs = []
    for s, yv in pre:
        R_, k_, tau_ = sysm.unpack(yv)
        Rs_ = float(abs(R_[i_surf]))
        if Rs_ >= R_surf:
            continue
        tq_R, tauq_R = quadrature(rg, R_surf, Rs_)
        devs.append((s["t"], Rs_, (s["t"] - tq_R) / s["t"], (float(tau_[i_surf]) - tauq_R) / s["t"], yv))
    early = [d for d in devs if 0.03 * tq <= d[0] <= 0.1 * tq]

    def deficits_at(yv):
        R_, k_, _ = sysm.unpack(yv)
        Fl_ = sysm.fields(R_, k_)
        srt = np.sort(np.abs(R_[:S]))
        return 1.0 - float(sysm.kernel_read(Fl_, Fl_["M"])[i_surf]) / E0, 1.0 - float(Fl_["N_app"][i_surf]), float(srt[-1] - srt[-2])
    late_txt = []
    for lab, sel_ in (("the last record with t <= 0.7 t_q", [d for d in devs if d[0] <= 0.7 * tq]), ("the last record before the fold", devs)):
        if sel_:
            d = sel_[-1]
            dk, nk, gap = deficits_at(d[4])
            late_txt.append("at %s (T/r_s %.3f, %.2f t_q): %.3e in t and %.3e in tau with the kernel-read deficits M_K/E_0 %.4f and N_app %.5f, the spacing below the surface shell %.2f a" % (lab, d[0] / 64.0, d[0] / tq, d[2], d[3], 1.0 - dk, 1.0 - nk, gap))
    if early:
        err_t = max(abs(d[2] - L_t_pred) for d in early)
        err_tau = max(abs(d[3] - L_tau_pred) for d in early)
        tol_t = 0.05 * abs(L_t_pred) + 5e-6
        tol_tau = 0.05 * abs(L_tau_pred) + 5e-6
        rec("3b-ii uniform ball: the outermost massive shell's early lag behind the point-exterior quadrature equals its first-order prediction from the release fields", err_t <= tol_t and err_tau <= tol_tau,
            "over the %d records with 0.03 t_q <= t <= 0.1 t_q: (t - t_quad(R_s))/t = %.4e against 1 - sqrt(a/a_0) = %.4e (|difference| %.1e, tolerance 0.05 of it + 5e-6 = %.1e), (tau_s - tau_quad(R_s))/t = %.4e against N_app m/eps_s - (m/eps_0) sqrt(a/a_0) = %.4e (|difference| %.1e, tolerance %.1e); a/a_0 = %.5f from the release forces and %.5f from the identity <N M/x^2>_K N_app (1 - w_app^2)/[(E_0/R_s^2)(1 - 2gE_0/R_s)] (the force per eps at rest is g <N M/x^2>_K, the local density cancelling between N' and v'): the kernel reads M_K/E_0 = %.5f at the surface shell (its own smeared mass %.4f of E_0, half of it outside; the neighbour at 2 sigma %.4f of its mass beyond; the spacing %.2f a against sigma %g a) and N_app %.6f there, so a/a_0 is about 1 - 0.7 delta_K - 2 nu_K and the lag about delta_K/2; the second-order terms are O(delta (R_0 - R)/R_0), under 3 percent of the lag in the window; the lag grows as the shells crowd: %s (v1b's tolerance 2e-3 took the surface shell for a shell outside the kernel's support, which it is not)" % (
                len(early), early[-1][2], L_t_pred, err_t, tol_t, early[-1][3], L_tau_pred, err_tau, tol_tau, ratio_a, ratio_ident, M_K0 / E0, mu0[i_surf] / E0, tail_nb0, dR, sigma, N_s, "; ".join(late_txt) or "no later record"))
    else:
        rec("3b-ii uniform ball: the outermost massive shell's early lag behind the point-exterior quadrature equals its first-order prediction from the release fields", False,
            "no record in the window 0.03 to 0.1 t_q before the first inversion (first at T/r_s %s; %s)" % (fx((T_inv / 64.0) if T_inv is not None else None, "%.4f"), out["label"]))
    # 3c E through the fold
    drift_raw = max([abs(s["E_over_E0"] - 1.0) for s in recs] + ([abs(fold["E_over_E0"] - 1.0)] if fold else []))
    drift_net = net_drift(recs + ([fold] if fold else []), ledger, E0)
    rec("3c uniform ball: E conserved through the fold", drift_net < 1e-7,
        "largest |E/E_0 - 1| over the records and the fold %.2e (net of the centre ledger %.2e; %d ledger entries: pins %d, releases %d, centre crossings %d); %d right-hand sides, %.0f s" % (
            drift_raw, drift_net, len(ledger), out["ledger_sums"]["pins"], out["ledger_sums"]["unpins"], out["ledger_sums"]["crossings"], out["nfev"], out["seconds"]))
    # 3d the sharp enclosed masses (v1c): the LTB energy integrals, the kernel-read drift decomposed, the pins listed
    pins_txt = []
    for e in ledger:
        if e["kind"] == "pin":
            pins_txt.append("shells %s (R_0 %s a, n %s particles, %.2e of the count) at T/r_s %.4f (%s centre: |v_app(0)| %.4f against N_app(0) %.4f)" % (
                e["shells"], [("%.1f" % sysm.R0[i]) for i in e["shells"]], [("%.2e" % sysm.n[i]) for i in e["shells"]], e["count"] / NB, e["T_over_rs"], "trapped" if e["trapped"] else "untrapped", abs(e["v0"]), e["N0"]))
    if pre:
        s_last, y_last = pre[-1]
        R_, k_, _ = sysm.unpack(y_last)
        Fl = sysm.fields(R_, k_)
        Rabs_l = np.abs(R_)
        mul, _ = sysm.shell_masses(Fl)
        Phil = cumulative_kernels(sysm, Rabs_l)
        th0, thl = sharp_theta(Rabs0), sharp_theta(Rabs_l)
        Msh0, Mshl = mu0 @ th0, mul @ thl
        Msm0, Msml = mu0 @ Phi0, mul @ Phil
        Md0 = np.interp(Rabs0, x, F["M"])
        Mdl = np.interp(Rabs_l, x, Fl["M"])
        massive = sysm.n > 0.0
        selm = massive & (Msh0 > 0.1 * E0)
        dev_sharp = float(np.max(np.abs(Mshl[selm] / Msh0[selm] - 1.0)))
        dev_dense_all = np.abs(Mdl / Md0 - 1.0)
        i_w = int(np.argmax(np.where(selm, dev_dense_all, -1.0)))
        dev_dense = float(dev_dense_all[i_w])
        part_sharp = (Mshl[i_w] - Msh0[i_w]) / Md0[i_w]
        part_overlap = ((Msml[i_w] - Mshl[i_w]) - (Msm0[i_w] - Msh0[i_w])) / Md0[i_w]
        part_resid = ((Mdl[i_w] - Msml[i_w]) - (Md0[i_w] - Msm0[i_w])) / Md0[i_w]
        resid0 = float(np.max(np.abs((Md0 - Msm0)[massive]))) / E0
        residl = float(np.max(np.abs((Mdl - Msml)[massive]))) / E0
        du_l = np.gradient(Fl["u"], x)
        size_l = sigma ** 2 * np.abs(np.interp(Rabs_l, x, du_l)) * np.abs(np.interp(Rabs_l, x, Fl["s_J"])) / E0
        big = sysm.n > 1e-6 * NB
        mu_drift = float(np.max(np.abs(mul[big] / mu0[big] - 1.0)))
        srt0, srtl = np.sort(Rabs0[:S]), np.sort(Rabs_l[:S])
        ok_3d = dev_sharp < 2e-3 and residl < 5e-3 and resid0 < 5e-3
        rec("3d uniform ball: every shell's sharp enclosed mass (the LTB energy integrals of the shells inside it plus half its own) kept before the first crossing", ok_3d,
            "largest |M_sharp(T)/M_sharp(0) - 1| over the %d shells with M > 0.1 E_0 at T/r_s %.4f (the last record before the first crossing%s): %.2e (tolerance 2e-3: mu_j = n_j(eps_j + <w>_j q_j) = n_j m e_j is exactly conserved for sharp dust shells and exchanged pairwise between smeared neighbours at a rate g int K_i K_j/(xu) n_i n_j [q_j R-dot_i e_i - q_i R-dot_j e_j] m, first order in their differences across a kernel width and zero for a homologous flow, the self-term cancelling exactly; a part in 1e-4 to 1e-3 of the enclosed mass over the fall); the largest per-shell |mu_j(T)/mu_j(0) - 1| over the shells above 1e-6 of the count %.2e; sum_j mu_j - E %.1e at release and %.1e at T. The kernel-read M(R_i(T))/M(R_i(0)) - 1 (v1b's quantity, largest %.2e at the shell R_0 %.1f a with M %.3f of E_0) decomposes as the sharp part %+.2e + the overlap of the neighbours' smeared masses sum_j mu_j [Phi_j(R_i) - theta_ji] %+.2e + the identity residual %+.2e: M(x) - sum_j mu_j Phi_j(x) is at most %.2e of E_0 at release and %.2e at T over the shells against its derived size sigma^2 |u'| s_J, at most %.2e (tolerance 5e-3; the overlap is a few 1e-4 at the release spacing 2 sigma and percent-level once the shells crowd below sigma: the spacing between the 2 outermost shells %.2f a at release and %.2f at T, the smallest spacing in the top quarter %.2f a); inversions through the fold %d (the fold's own count against the last record %d); centre crossings %d; pins %d: %s (inside every counted shell before and after; their displacement to R = 0 changes E by %.1e, the ledger)" % (
                int(selm.sum()), s_last["t"] / 64.0, (" at T/r_s %.4f" % (T_inv / 64.0)) if T_inv is not None else ("; no inversion among the records" + (", %d between the last record and the fold" % inv_fold if inv_fold else "")),
                dev_sharp, mu_drift, float(np.sum(mu0)) - E0, float(np.sum(mul)) - float(Fl["E"]), dev_dense, sysm.R0[i_w], Md0[i_w] / E0, part_sharp, part_overlap, part_resid,
                resid0, residl, float(np.max(size_l[massive])), srt0[-1] - srt0[-2], srtl[-1] - srtl[-2], float(np.min(np.diff(srtl[3 * S // 4:]))) if S >= 8 else float("nan"),
                inv_total, inv_fold, out["ledger_sums"]["crossings"], out["pins"], "; ".join(pins_txt) if pins_txt else "none", out["ledger_sums"]["dE_pins_exact"]))
    else:
        rec("3d uniform ball: every shell's sharp enclosed mass (the LTB energy integrals of the shells inside it plus half its own) kept before the first crossing", False,
            "no record before the first inversion (first at T/r_s %s); pins: %s" % (fx((T_inv / 64.0) if T_inv is not None else None, "%.4f"), "; ".join(pins_txt) if pins_txt else "none"))
    if fold is None:
        rec("3e uniform ball: the first root at or after the outermost shell's point-exterior clock, inside 2gE_0", False, "no fold through %.3f r_s (%s)" % (t_ball / 64.0, out["label"]))
    else:
        Tf, Rf = fold["t"], fold["depth_R"]
        ok = (Tf >= tq * (1.0 - 2e-3)) and (Tf <= 1.1 * tq) and (0.75 <= Rf / rg <= 1.002)
        rec("3e uniform ball: the first root at or after the outermost shell's point-exterior clock, inside 2gE_0", ok,
            "fold T/r_s %.6f against the single-shell quadrature t_q %.6f (ratio %.5f; T_f >= t_q is the theorem's lower bound, T_f <= 1.1 t_q and 0.75 <= R_f/(2gE_0) <= 1 are sanity bounds), R_f %.4f a = %.5f of 2gE_0 (count inside %.4f), E/E_0 - 1 at the fold %+.2e, %d pinned; inversions through the fold %d (the first at T/r_s %s; the fold's own count against the last record %d); the outermost shell's proper time at the fold %.6f r_s (its quadrature to r_g %.6f); the multistream delay T_f - t_q = %+.5f r_s is the ball's own (v1's 'first root at the surface at the Oppenheimer-Snyder clock' was not a property of this release)" % (
                Tf / 64.0, tq / 64.0, Tf / tq, Rf, Rf / rg, fold["count_fraction_inside_depth_R"], fold["E_over_E0"] - 1.0, fold["n_pinned"], inv_total, fx((T_inv / 64.0) if T_inv is not None else None, "%.4f"), inv_fold,
                fold["tau_outer_shell"] / 64.0, tauq / 64.0, (Tf - tq) / 64.0))
    if a.self_test_frozen and fold is not None:
        for tau_f in (8.0, 4.0):
            sysf, metaf = ball_system(Rsh, n, sigma, 0.5 * sigma, g, m)
            kf, _, Ff0 = release(sysf, "rest")
            sysf.E0 = Ff0["E"]
            of = run_frozen(sysf, sysf.pack(sysf.R0, kf, np.zeros(sysf.S)), 1.1 * tq, tau_f, 1e-10, 1e-12, metaf["x_guard"], a.budget_test, k_max=20.0, log=log)
            log("     the frozen step tau %g on the ball: fold (linear) T/r_s %s, E/E_0 - 1 at the fold %s (the adaptive fold %.6f); pins %d; %s" % (
                tau_f, fx((of["fold"] or {}).get("T_over_rs"), "%.6f"), fx(((of["fold"] or {}).get("E_over_E0") or 1.0) - 1.0, "%+.2e"), fold["t"] / 64.0, of["pins"], of["label"]))
    # 4 crossings (v1b: the vector kernel asserted; the density kernel's centre passage with its exact jump; the lapse-off drift printed)
    Rc_ = np.concatenate([[20.0], 40.0 + 5.0 * np.arange(6), 100.0 + 5.0 * np.arange(6)])
    nc_ = np.concatenate([[3.0], np.full(6, 15.0), np.full(6, 15.0)])
    kc_ = np.concatenate([[-0.5], np.full(6, 0.12), np.zeros(6)])
    for kern, lap in (("vector", True), ("density", True), ("density", False)):
        profc = {"R": Rc_, "n_site": nc_, "n_floor": 0.0, "rs": 64.0}
        sysc, metac = build_system(profc, 2.0, kern, "continuum", lap, g, m)
        krest, _, Fc = release(sysc, "rest")
        kinit = np.where(kc_ == 0.0, krest, kc_)
        Fc = sysc.fields(Rc_, kinit)
        sysc.E0 = Fc["E"]
        oc = run_adaptive(sysc, sysc.pack(Rc_, kinit, np.zeros(Rc_.size)), 240.0, 10.0, 1e-10, 1e-12, metac["x_guard"], a.budget_test, k_max=20.0, log=log)
        inv = int(sum((s["inversions"] or 0) for s in oc["records"]))
        cen = int(max(s["n_center"] for s in oc["records"]))
        cr = [e for e in oc["ledger"] if e["kind"] == "crossing"]
        d_raw = max(abs(s["E_over_E0"] - 1.0) for s in oc["records"])
        d_net = net_drift(oc["records"], oc["ledger"], sysc.E0)
        if kern == "vector":
            rec("4a shell crossings and centre crossings conserve E (vector kernel: the flow vanishes at the centre)", inv >= 1 and cen >= 1 and d_raw < 1e-8,
                "inversions %d, centre crossings %d (the ledger's %d, their exact jumps sum %+.2e), largest |E/E_0 - 1| %.2e (E_0 %.4f; nfev %d, %.0f s; pins %d; %s)" % (
                    inv, cen, len(cr), sum(e["dE_exact"] for e in cr), d_raw, sysc.E0, oc["nfev"], oc["seconds"], oc["pins"], oc["label"]))
        elif lap:
            jump = float(sum(e["dE_exact"] for e in cr))
            jump1 = float(sum((e["dE_first_order"] or 0.0) for e in cr))
            rec("4b the scalar kernel's centre passage: E changes by the exact flip of the central momentum source and is conserved otherwise (density kernel)", cen >= 1 and len(cr) >= 1 and d_net < 1e-7,
                "centre crossings %d: the exact jump %+.4e of E_0 %.4f (first order -2 n q v_app(0): %+.4e); raw drift %.2e, net of the ledger %.2e; inversions %d, pins %d (nfev %d, %.0f s; %s)" % (
                    len(cr), jump, sysc.E0, jump1, d_raw, d_net, inv, oc["pins"], oc["nfev"], oc["seconds"], oc["label"]))
        else:
            log("     the lapse-off variant (density kernel) on the same crossings: inversions %d, centre crossings %d, raw drift %.2e, net of the ledger %.2e (no assertion: with N forced to 1 the charge is not the generator); %s" % (
                inv, cen, d_raw, d_net, oc["label"]))
    # 5 the generator identity
    rng = np.random.default_rng(7)
    S5 = 40
    R5 = rng.uniform(20.0, 200.0, S5)
    k5 = rng.uniform(-0.2, 0.3, S5)
    n5 = rng.uniform(0.5, 5.0, S5)
    for kern in ("density", "radial", "vector"):
        prof5 = {"R": R5, "n_site": n5, "n_floor": 0.0, "rs": 64.0}
        sys5, _ = build_system(prof5, 4.0, kern, "continuum", True, g, m)
        F5 = sys5.fields(R5, k5)
        gR = n5 * (F5["sgn"] * F5["dN"] * F5["eps"] + F5["dv"] * F5["q"])
        gk = n5 * (F5["N_app"] * sys5.disp.deps(k5) + F5["sgn"] * F5["v_app"] * sys5.disp.dq(k5))
        worst = 0.0
        scale = max(float(np.max(np.abs(gR))), float(np.max(np.abs(gk))))
        for i in rng.choice(S5, 6, replace=False):
            d = 1e-5
            Rp, Rm = R5.copy(), R5.copy()
            Rp[i] += d
            Rm[i] -= d
            fd = (sys5.fields(Rp, k5)["E"] - sys5.fields(Rm, k5)["E"]) / (2 * d)
            worst = max(worst, abs(fd - gR[i]) / scale)
            kp, km = k5.copy(), k5.copy()
            kp[i] += d
            km[i] -= d
            fd = (sys5.fields(R5, kp)["E"] - sys5.fields(R5, km)["E"]) / (2 * d)
            worst = max(worst, abs(fd - gk[i]) / scale)
        rec("5 generator identity dE = sum n dh (%s kernel)" % kern, worst < 1e-6, "largest |central difference - analytic|/scale %.2e over 6 random shells (dR and dk), E %.6f" % (worst, F5["E"]))
    # 6 kernels and the vacuum lapse
    grid6 = Grid(0.5, 300.0)
    worst = 0.0
    for kern in ("density", "radial"):
        for Rt in (0.3, 5.0, 50.0):
            s = float(np.sum(grid6.omega * Kernel(2.0, kern).K(grid6.x, Rt, "rho")))
            worst = max(worst, abs(s - 1.0))
    solver6 = FieldSolver(grid6, Kernel(2.0, "density"), g, lapse=True)
    M6, u6, N6, neg6, ok6 = solver6.solve(np.zeros(grid6.G + 1), np.zeros(grid6.G + 1))
    rec("6 kernel normalization and the vacuum lapse", worst < 1e-9 and float(np.max(np.abs(N6 - 1.0))) < 1e-12 and ok6, "largest |int K - 1| %.2e; vacuum |N - 1| max %.2e" % (worst, float(np.max(np.abs(N6 - 1.0)))))
    # 7 the trapped centre (v1b): the cause of v1's hang, now the pinned caustic
    R7 = np.array([10.0, 12.0, 14.0, 30.0])
    n7 = np.array([200.0, 200.0, 200.0, 10.0])
    k7 = np.array([0.0, 0.0, 0.0, -0.5])
    for kern in ("density", "vector"):
        prof7 = {"R": R7, "n_site": n7, "n_floor": 0.0, "rs": 64.0}
        sys7, meta7 = build_system(prof7, 1.0, kern, "continuum", True, g, m)
        krest, info7, F7 = release(sys7, "rest")
        kinit = np.where(k7 != 0.0, k7, np.where(np.isfinite(krest), krest, 0.0))
        F7 = sys7.fields(R7, kinit)
        sys7.E0 = F7["E"]
        o7 = run_adaptive(sys7, sys7.pack(R7, kinit, np.zeros(4)), 60.0, 5.0, 1e-9, 1e-11, meta7["x_guard"], a.budget_test, k_max=20.0, log=log)
        d_raw = max(abs(s["E_over_E0"] - 1.0) for s in o7["records"])
        d_net = net_drift(o7["records"], o7["ledger"], sys7.E0)
        ca = o7["caustic"] or {}
        ls = o7["ledger_sums"]
        k_end = float(np.max(np.abs(sys7.unpack(o7["y_end"])[1])))
        if kern == "density":
            rec("7 the trapped centre (density kernel): the cluster's shells are pinned at the centre, the run completes in bounded time, E conserved net of the ledger",
                o7["status"] == 0 and o7["n_pinned_end"] == 4 and o7["caustic"] is not None and d_net < 1e-6,
                "2gE_0 %.2f a against the cluster's radius 14 a (trapped at release: %d shells without a rest momentum); pins %d, releases %d, restarts %d, pinned at the end %d of 4; the caustic (the first pin at a trapped centre, |v_app(0)| %s against N_app(0) %s) at T %s a; E drift raw %.2e, net of the ledger %.2e (the pin displacements' sum %+.2e, the crossings' %+.2e); %d right-hand sides, %.1f s; %s" % (
                    2.0 * g * sys7.E0, info7["shells_without_rest"], o7["pins"], o7["unpins"], o7["restarts"], o7["n_pinned_end"], fx(abs(ca["v0"]) if ca.get("v0") is not None else None, "%.3f"), fx(ca.get("N0"), "%.3f"), fx(ca.get("t"), "%.3f"),
                    d_raw, d_net, ls["dE_pins_exact"], ls["dE_crossings_exact"], o7["nfev"], o7["seconds"], o7["label"]))
        else:
            log("     the vector kernel on the same cluster (the flow vanishes at the centre, no pin): pinned %d, centre crossings %d, |k| at the end %.3g, E drift raw %.2e, net %.2e; %d right-hand sides, %.1f s; %s" % (
                o7["n_pinned_end"], ls["crossings"], k_end, d_raw, d_net, o7["nfev"], o7["seconds"], o7["label"]))
    ok_all = all(i["pass"] for i in items)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "dust_comparator_v1c_selftest.json"), "w") as f:
        json.dump({"code": CODE, "recursion": RECURSION_ROUTE, "reader": IO_NOTE, "items": items, "all_pass": ok_all, "seconds": time.time() - T0}, f, default=jdefault, indent=1)
    with open(os.path.join(a.out, "dust_comparator_v1c_selftest.md"), "w") as f:
        f.write("\n".join(["# %s self-test (%s; recursion %s)" % (TITLE, CODE, RECURSION_ROUTE), ""] + ["- %s: %s: %s" % ("PASS" if i["pass"] else "FAIL", i["name"], i["detail"]) for i in items] +
                          ["", "- all pass: %s; %.0f s." % (ok_all, time.time() - T0)]) + "\n")
    log("self-test %s in %.0f s" % ("PASSED" if ok_all else "FAILED", time.time() - T0))
    return 0 if ok_all else 1


# ------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", action="append", default=[], help="a T0000 slice (repeatable; each sigma uses the slice whose name carries its ell tag, else the first)")
    ap.add_argument("--slice-glob", dest="slice_glob", default=DEFAULT_SLICE_GLOB, help="a recursive glob for the T0000 slices where --slice is not given")
    ap.add_argument("--records", action="append", default=[], help="a rung's records json (repeatable)")
    ap.add_argument("--records-glob", dest="records_glob", default=DEFAULT_RECORDS_GLOB)
    ap.add_argument("--fields", action="append", default=[], help="a rung's fields npz (optional; the T = 0 shell totals printed beside)")
    ap.add_argument("--sigma", default="2,4,8", help="the averaging scales sigma/a")
    ap.add_argument("--matrix", default="default", choices=["single", "primary", "default", "full"])
    ap.add_argument("--kernel", default="density", choices=["density", "radial", "vector"])
    ap.add_argument("--dispersion", default="continuum", choices=["continuum", "sine"])
    ap.add_argument("--release", default="rest", choices=["rest", "read"])
    ap.add_argument("--lapse", default="on", choices=["on", "off"])
    ap.add_argument("--step", default="adaptive", choices=["adaptive", "frozen"])
    ap.add_argument("--tau", type=float, default=8.0, help="the frozen step in a")
    ap.add_argument("--T-end", dest="T_end", type=float, default=16.0, help="the end clock in r_s")
    ap.add_argument("--dt-rec", dest="dt_rec", type=float, default=8.0, help="the record spacing in a (the engine's 8 a)")
    ap.add_argument("--h", type=float, default=None, help="the grid spacing in a (default min(a/2, sigma/2))")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-11)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--rs", type=float, default=64.0, help="the nominal r_s in a where no slice gives it")
    ap.add_argument("--R0-head", dest="R0_head", type=float, default=256.0, help="the production's head release radius for the single-shell control")
    ap.add_argument("--page-code", dest="page_code", default=DEFAULT_PAGE_CODE)
    ap.add_argument("--n-floor", dest="n_floor", type=float, default=1e-7, help="shells below this fraction of the total count are dropped")
    ap.add_argument("--no-velocity", dest="no_velocity", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "DUST_comparator_v1d_out"))
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    ap.add_argument("--self-test-fine", dest="self_test_fine", action="store_true", help="the ball at 256 shells, sigma 0.5")
    ap.add_argument("--self-test-frozen", dest="self_test_frozen", action="store_true", help="also the frozen steps on the ball (print only)")
    ap.add_argument("--read-only", dest="read_only", action="store_true", help="read the profiles, compare the T = 0 fields, no run")
    # v1b
    ap.add_argument("--budget-seconds", dest="budget_seconds", type=float, default=900.0, help="v1b: the wall-clock budget per comparator run (the records so far stand, the end is labelled)")
    ap.add_argument("--budget-test", dest="budget_test", type=float, default=300.0, help="v1b: the wall-clock budget per self-test integration")
    ap.add_argument("--after-caustic", dest="after_caustic", default="continue", choices=["continue", "stop"],
                    help="v1b: continue past the central caustic with the arriving shells pinned at the centre (default) or stop the run there")
    ap.add_argument("--k-max", dest="k_max", type=float, default=20.0, help="v1b: the momentum guard for the continuum dispersion (a shell's |k| reaching it ends the run, labelled; 0 disables)")
    # v1d
    ap.add_argument("--modes", default="", help="v1d: comma list of control, columns, census, constant (the barrier allocations); empty runs the delivered matrix")
    ap.add_argument("--k-max-barrier", dest="k_max_barrier", type=float, default=1000.0, help="v1d: the momentum guard for the barrier runs")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.self_test:
        return self_test(a)
    log = Logger(os.path.join(a.out, "dust_comparator_v1d.log"))
    log("%s: recursion %s; reader %s; numpy %s; budget per run %g s, k_max %g, after the caustic: %s" % (CODE, RECURSION_ROUTE, IO_NOTE, np.__version__, a.budget_seconds, a.k_max, a.after_caustic))
    slices = list(a.slice) or sorted(glob.glob(a.slice_glob, recursive=True))
    if not slices:
        log("no T0000 slice found (--slice or --slice-glob %s): nothing to read" % a.slice_glob)
        return 2
    records = list(a.records) or sorted(glob.glob(a.records_glob, recursive=True))
    sigmas = [float(s) for s in a.sigma.split(",") if s.strip()]
    profs = [read_slice_profile(p, a.m, a.page_code, a.n_floor, log, velocity=not a.no_velocity) for p in slices]
    records_read = []
    for p in records:
        try:
            records_read.append(read_records(p, a.rs))
        except Exception as ex:  # noqa: BLE001
            log("records json %s not read (%s: %s)" % (p, type(ex).__name__, ex))
    fields_T0 = [read_fields_T0(p, log) for p in a.fields]

    def prof_for(sigma):
        for p in profs:
            if p["ell_tag"] is not None and float(p["ell_tag"]) == float(sigma):
                return p
        return profs[0]
    modes = [x.strip() for x in a.modes.split(",") if x.strip()]
    for md in modes:
        if md not in BARRIER_MODES:
            raise SystemExit("unknown mode %s" % md)
    # the T = 0 fields under the 3 kernels at the 3 scales per slice
    field_rows = []
    for p in (profs if not modes else []):
        for s in sorted(set(sigmas) | ({float(p["ell_tag"])} if p["ell_tag"] else set())):
            for kern in ("density", "radial", "vector"):
                try:
                    c = compare_T0_fields(p, s, kern, a.g, a.m, log)
                except Exception as ex:  # noqa: BLE001
                    log("T = 0 field comparison failed at sigma %g, %s (%s: %s)" % (s, kern, type(ex).__name__, ex))
                    c = None
                if c:
                    field_rows.append({"slice": p["path"], "ell_tag": p["ell_tag"], "cmp": {k_: v_ for k_, v_ in c.items() if k_ not in ("N_s", "v_s")}})
                    log("T = 0 fields %s sigma %g %s: E0 %.6f, RMS dN %.3e dv %.3e inside R_99" % (os.path.basename(p["path"]), s, kern, c["E0"], c["rms_N_inside"], c["rms_v_inside"]))
    # the single-shell control with the profile's charge
    single_rows = []
    p0 = profs[0]
    sysm0, _ = build_system(p0, 4.0, "density", "continuum", True, a.g, a.m)
    k0_, _, F0 = release(sysm0, "rest")
    E_prof = F0["E"]
    for lab, R0c, Ec in (() if modes else (("head R0, the profile's E_0", a.R0_head, E_prof), ("R_99 of the profile, E_0", p0["R_99"], E_prof), ("P20 control", 256.0, 59.05 / (2.0 * a.g)), ("P20 control", 256.0, 64.0 / (2.0 * a.g)))):
        rg = 2.0 * a.g * Ec
        if rg >= R0c:
            continue
        t, tau = quadrature(rg, R0c, rg)
        tc, tauc = canonical(rg, R0c, rg)
        ts, taus, _, tcen = run_test_shell(Ec, R0c, rg, a.g, a.m, Dispersion("continuum", a.m), rtol=1e-11)
        tsin, _, _, _ = run_test_shell(Ec, R0c, rg, a.g, a.m, Dispersion("sine", a.m), rtol=1e-11)
        tab = P20_TABLE.get((round(rg, 2), R0c))
        single_rows.append({"label": lab, "R0": R0c, "E": Ec, "rg": rg, "t": t, "tau": tau, "t_centre": tcen, "t_can": tc, "tau_can": tauc, "t_shell": ts, "tau_shell": taus, "t_sine": tsin,
                            "table": ("%.10f, %.10f" % tab) if tab else "-"})
        log("single shell %s: R0 %.3f, r_g %.4f (%.4f r_s), t/r_s %.6f, tau/r_s %.6f, the flat clock to the centre %.6f r_s; canonical %s; the comparator's test shell t %s; sine %s" % (
            lab, R0c, rg, rg / p0["rs"], t / p0["rs"], tau / p0["rs"], tcen / p0["rs"], fx(tc, "%.6f"), fx(ts, "%.6f"), fx((tsin / p0["rs"]) if tsin else None, "%.6f")))
    results = []
    if not a.read_only:
        have_vel = any(p["velocity"] is not None for p in profs)
        if modes:  # v1d
            specs = [{"label": "s%g_%s_%s_%s" % (s_, md, a.dispersion, a.release), "sigma": s_, "kernel": a.kernel, "dispersion": a.dispersion, "release": a.release,
                      "lapse": a.lapse, "step": a.step, "tau": a.tau, "mode": md} for s_ in sigmas for md in modes]
        else:
            specs = build_matrix(sigmas, a.matrix, have_vel, a)
        log("%d runs: %s" % (len(specs), [s["label"] for s in specs]))
        for spec in specs:
            prof = prof_for(spec["sigma"])
            try:
                res, arrays = run_spec(spec, prof, a, log)
            except Exception as ex:  # noqa: BLE001
                log("run %s FAILED (%s: %s)" % (spec["label"], type(ex).__name__, ex))
                continue
            res["profile"] = os.path.basename(prof["path"])
            results.append(res)
            with open(os.path.join(a.out, "%s.json" % spec["label"]), "w") as f:
                json.dump(res, f, default=jdefault, indent=1)
            np.savez(os.path.join(a.out, "%s.npz" % spec["label"]), **arrays)
    L = compose(a, profs, field_rows, single_rows, results, records_read, fields_T0, a.out, log)
    with open(os.path.join(a.out, "%s.md" % TITLE), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(a.out, "dust_comparator_v1d_all.json"), "w") as f:
        json.dump({"code": CODE, "title": TITLE, "args": vars(a), "recursion": RECURSION_ROUTE, "reader": IO_NOTE, "quoted_records": {str(k): v for k, v in QUOTED_RECORDS.items()}, "quoted_source": QUOTED_SOURCE,
                   "centre_note": CENTRE_NOTE,
                   "columns": [{k_: v_ for k_, v_ in c.items() if k_ != "n"} | {"count": float(c["n"].sum())} for p in profs for c in p["columns"]],
                   "profiles": [{k_: v_ for k_, v_ in p.items() if k_ not in ("R", "n_site", "N_applied", "v_applied", "velocity", "columns")} | {"velocity_summary": ({k_: v_ for k_, v_ in p["velocity"].items() if k_ not in ("I_cut", "u_cut", "u_site")} if p["velocity"] else None)} for p in profs],
                   "T0_field_rows": field_rows, "single_shell": single_rows, "records_read": records_read, "fields_T0": fields_T0,
                   "results": [{k_: v_ for k_, v_ in r.items() if k_ != "records"} for r in results], "seconds": time.time() - T0}, f, default=jdefault, indent=1)
    for p in profs:
        np.savez(os.path.join(a.out, "profile_%s.npz" % os.path.basename(p["path"])[:-4]), R=p["R"], n_site=p["n_site"], N_applied=(p["N_applied"] if p["N_applied"] is not None else np.zeros(0)),
                 v_applied=(p["v_applied"] if p["v_applied"] is not None else np.zeros(0)), u_site=(p["velocity"]["u_site"] if p["velocity"] else np.zeros(0)), I_cut=(p["velocity"]["I_cut"] if p["velocity"] else np.zeros(0)))
    log("%s: %d runs composed in %s (%.0f s)" % (TITLE, len(results), a.out, time.time() - T0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
