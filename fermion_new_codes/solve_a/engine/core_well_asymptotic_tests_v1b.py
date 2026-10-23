#!/usr/bin/env python
"""core_well_asymptotic_tests_v1b.py (reviewer A for the solve builder, the busi lane; inbox busi #194, the lead's order, verbatim from
REVIEWA_asymptotic_push_v1: the cheap stored-data tests T1 to T8, box CPU, no engine run, no slice read, nothing but the stored readout).

VERSION 1b (v1b) of core_well_asymptotic_tests_v1.py; every changed line is marked v1b. Changes: (1) the self-test's synthetic history had its
flow inverted, w = -sqrt(M/(2 g R)) for -sqrt(2 g M/R), so its stored mass function was 4 M, its lambda 0.49 for 2.0 and its outer root beyond
the 96-site chain, which left T6 without a row and raised the IndexError (the box self-test's FAILs 6 and 7): fixed in write_synthetic_history,
the synthetic's 2 roots checked against the cubic they solve, every pipeline check guarded and named once; (2) the kernel normalization check
integrated from 0.25 a and missed the kernel's mass inside (FAILs 2 and 3): the volume integrand 4 pi r^2 rho_ell, finite at r = 0, is now
integrated from r = 0 by the trapezoid, spectrally accurate for that even analytic integrand, the expectation 1 to 1e-10; (3) the uniform-density
check reached 236 a, 5 ell from the chain's end, where the chain's missing Gaussian tail is 1.5e-7 (FAIL 4): the window ends 6 ell inside the chain
and the chain-end deficit at 236 a is checked against its derived value; (4) T1's lambda at h = 1 is checked against the centred difference's own
leading truncation error lambda (h^2/6)|M'''/M'| (5.32e-3, 1.32e-3, 1.61e-3 at R 16, 32, 64 on the manufactured profile; FAIL 1); (5) T4's
sinusoid checks state the periodogram's resolution at 2.5 cycles (the plain bin bracketing 1/P, the padded peak within its half-power half-width)
and a 50-cycle record recovers 8 r_s to 0.02 (FAIL 5); (6) the red team's amendments (REVIEWA_asym_redteam_v1, applied by the lead to
requirements_new.txt): T1 prints lambda from the unsmoothed site sources labelled as the push's reading (E21), the halo's mean lambda and the edge
lapse N(R_c)/N(root) against the exact law (premise (b)), and the R <= R_50 window's fitted exponent against the measured (lambda - 1)/2; T2 prints
4 a n_rad,max beside 1.222 N_B^(1/3) a and the exclusion floor 0.49 N_B^(1/3) a (E2, E9), the outward branch's onset z_out beside z* (E3), the
directional shares, and carries the reading as amended; T3 prints the energy per shed particle and carries the fields'-work reading (E4, E5); T4
prints the frequency in T_H units and its resolution; T5 carries the amended reading of the signed null energy (E6, E7); T8 carries the halo clause;
T6 and T7 are unchanged by the red team. t8_all's line says so when no rooted slice is read.

Inputs (nothing else is read): the per-slice outputs of core_well_v1.py under <internal>/well
(well_<history>_T<TTTT.TT>.json and .npz; the .md is not parsed) for prod at T/r_s 12, 15, 16, 20, 24, 25, 28, 30, 35, 40 and s8 at 12,
13, 14, 16, 20, 25, 30, 40, their records_prod.json and records_s8.json (the per-record rows T_over_rs, R_t, R_in, E), and, for T2 only,
the rung stand-ins' per-slice outputs under <internal>/well (well_r8_T0013.00, well_r8_T0014.00, and
well_r4_T0013.00, well_r4_T0014.00 where they exist at read time), read under their own label (the json's history_label) and never as
the production's or the c27 member's. The schema is core_well_v1.FORMAT_LINE (SECONDROUTE_CORE_well_v1).

The tests, verbatim (TESTS_VERBATIM below is printed at the head of every section of the output), each followed by the red team's amendment
of its comparator, label or reading (AMENDED below; v1b).

What the stored outputs hold and what they do not (the routes; each is printed in the output beside the numbers it produced):
- T1 needs the records' kernel-averaged rho_ell (FIRSTLAW_records v2.2 section 2.2: the engine's Gaussian-kernel average, standard deviation
  ell, per unit volume of the stored per-site sources). No stored array holds it: the npz's rho_density is the site rho = eps_tot/(4 pi r^2 a),
  unsmoothed. rho_ell is therefore formed by the route T1 allows, the site sources eps_tot convolved with a Gaussian of standard deviation ell
  (the history's sigma_over_a: 4 on prod, 8 on s8), and J_ell the same for J_tot (used only in the derived check columns). The engine's kernel
  form is not among the stored outputs; two forms are offered and the one used is printed: --kernel shell3d (the default: each site's shell
  energy smeared by the three-dimensional isotropic Gaussian, a density per unit volume regular at the centre, whose volume integral is
  exactly sum eps_tot) and --kernel radial1d (the one-dimensional Gaussian along r, reflected at r = 0, over 4 pi r^2). The volume integral
  is checked against sum eps_tot (FIRSTLAW's own check) and the identity's residual is the arbiter. The dense fields at the sites are
  core_well_v1's linear interpolation of the fields npz's stored nodes (513 on prod, 257 on s8), so the centred differences of M_w and ln N
  on the npz grid carry that interpolation's error; the kinks of the interpolant are detected and the node spacing they imply is printed;
  the self-test shows the arithmetic reproduces the centred difference's own truncation error on a manufactured constant-lambda profile (v1b).
  (v1b) The table also prints lambda from the unsmoothed site sources with the dense w (the push's (1.6) construction, labelled as the reading
  it is, E21), the halo interval R_c to the outer root with its secant, its mean lambda 1 + 2 secant and the edge lapse N(R_c)/N(root)
  against exp(-integral ((lambda - 1)/2) d ln M_w) from the pointwise lambda (premise (b)), and the R <= R_50 window's fitted slope of
  ln N against ln N_< (the json's) beside (eps_c - m a)/(2 m a), eps_c/(2 m a) and the measured (lambda - 1)/2 (requirements_new.txt).
- T2: n_rad(kappa) is the json's channels[].n_dust (the dust columns per channel); the channel's bound R_90 channels[].killing.bound.R_90;
  the inner-shell share channels[].shells[0].frac_abs_p_above (recomputed from the npz's k<kappa>_shell00_nk on the grid k with p_reduced as
  the check); z* solves |w| |cos 2z| = q(z)/eps(z) with q = sin z cos z and eps = sqrt((m a)^2 + sin^2 z) (the symbols behind CORE_well
  section 7's kinematic note, a p = 0.32 at m a 0.2) at the slice's inner-root |w| (dense; the applied w there beside) and at the largest
  |w| inside the outer root. (v1b) z_out, the onset of the outward branch for outward momenta, q/eps = |w| cos 2z below pi/4 (0.32 rad at
  |w| = 1; E3), prints beside z*, with the shares of outward momenta p > z_out and inward momenta p < -z*; R_50 and R_90 print against
  4 a n_rad,max, 1.222 N_B^(1/3) a and the exclusion floor 0.49 N_B^(1/3) a (0.39 with all 4 components), the inner root labelled a root
  and not compared with the floor (E2); the reading prints as amended (E9).
- T3: killing.bins_weighted [{label, N, E}] of every json; the field work from the two slices' applied N, v (the dense fields beside) and the
  mid-slice sources (eps_tot, J_tot; the dust's part eps_tot - eps_pairs, J_tot - J_pairs beside); the record's own M1_balance.field_work
  per step at the two ends beside. (v1b) The (m, b] bin's mean Killing energy per particle in units of m prints beside; the reading is the
  fields' work on the orbitals (E4, E5).
- T4: records_<history>.json rows; the slices' fountain block for the bound omega/m and r_turn; the literal integral 2 int dr/(1 - sqrt(r_s/r))
  is printed where finite and reported divergent where the root lies at or inside r_s (the integrand's pole at r = r_s), with the
  Painleve-Gullstrand geodesic fall from rest at r_turn to r_s beside (derived) and T6's kinematic bound as the core legs' lower bound.
  (v1b) The period's angular frequency 2 pi/P over T_H = 1/(4 pi r_s) prints beside, and the resolution of a 2.5-cycle window is stated.
- T5: the stress p_r is held by no stored array, neither per species nor total, and the stress module's companion rows are not among the
  stored outputs: the stress split prints "not available"; the energy-flux split uses currents[dense_outer].weighted dust_FM, dev_FM, pairs_FM
  and the momentum split dust_FM_low/high/cross (non-additive at the cut: the cross term is printed). (v1b) The reading prints as amended.
- T6: N_dense_sites, w_dense between the record's trapping zeros; N_applied, w_applied between the applied roots; each between the other's roots
  beside; the front speed from the jsons' fountain.R_90_all at 20 and 40.
- T7: N_dense_sites, N_applied, n_density, w_dense at R = 1 to 8 a.
- T8: M_w_dense (M_w_applied beside) against the record's E.
Outputs: <out>/ASYMPTOTIC_tests_v1b.md (one table per slice for T1 and T2; one table per history with one row per slice for T3 to T8, the
sequence tests), ASYMPTOTIC_tests_v1b.json (every printed number), ASYMPTOTIC_tests_v1b_arrays.npz (the per-site arrays of T1),
asymptotic_v1b.log; --self-test writes <out>/selftest/ASYMPTOTIC_selftest_v1b.md and .json and exits 1 on any failed check. CPU only; numpy
only; no engine import; no plotting; no dates (elapsed seconds only)."""
import os
import sys


def _pre_threads(default="8"):
    """the BLAS thread count from --threads (8 where not given: the lead's rule for this run), set before numpy is imported."""
    n = None
    argv = sys.argv[1:]
    for i, x in enumerate(argv):
        if x == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif x.startswith("--threads="):
            n = x.split("=", 1)[1]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if n is not None:
            os.environ[var] = str(int(n))
        else:
            os.environ.setdefault(var, default)


_pre_threads()
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

CODE = "core_well_asymptotic_tests_v1b.py"  # v1b
TITLE = "ASYMPTOTIC_tests_v1b"  # v1b
WELL_DIR = "<internal>/well"
RUNG_DIR = "<internal>/well"
OUT_DIR = "<internal>/asymptotic_v1b"  # v1b
CLOCKS = {"prod": (12, 15, 16, 20, 24, 25, 28, 30, 35, 40), "s8": (12, 13, 14, 16, 20, 25, 30, 40)}
T2_TARGETS = (("prod", 15.0, "well"), ("s8", 14.0, "well"), ("r8", 13.0, "rung"), ("r8", 14.0, "rung"), ("r4", 13.0, "rung"), ("r4", 14.0, "rung"))
T7_NAMED = {"prod": (20, 24, 25, 28, 30, 35, 40), "s8": (13, 14, 16, 40)}
T4_RECORDS_FROM, T4_WINDOW = 18.375, (20.0, 40.0)
N_RAD_DECLARED = {1: 5, 2: 4, 3: 3, 4: 2, 5: 2}
PI4 = math.pi / 4.0
T1_FIVE = ("inner root", "R_50", "R_c", "R_90", "outer root")
T8_LINE = "the mass function peaks at the root below E on all 13 rooted slices: the halo-cancellation closure does not occur at (0.2, 64)"
C_R = (18.0 / math.pi ** 2) ** (1.0 / 3.0)  # v1b: 1.2218, the isotropic ball's radius 1.222 N_B^(1/3) a at 16 states per momentum
FLOOR_2 = (3.0 / (8.0 * math.pi)) ** (1.0 / 3.0)  # v1b: 0.4924 N_B^(1/3) a, the exclusion floor at 2 positive-band states per site (red team E2)
FLOOR_4 = (3.0 / (16.0 * math.pi)) ** (1.0 / 3.0)  # v1b: 0.3908 N_B^(1/3) a with all 4 native components (red team E2)
T0 = time.time()

TESTS_VERBATIM = {
    "T1": "T1. The exact clock law and λ(r). On every stored slice of prod and s8: from the dense fields' M_w(R) and the records' kernel-averaged ρ_ℓ (FIRSTLAW v2.2's rows; if only the site ρ is at hand, convolve it with the engine's Gaussian at the history's ℓ), print at the inner root, R_50, R_c, R_90, the outer root: λ = 4πR²ρ_ℓ/M_w′ (M_w′ by centred differences on the npz grid), (λ − 1)/2, the secant d ln N/d ln M_w between consecutive named radii, and the residual of (ln N)′M_w − 2πR²ρ_ℓ + M_w′/2 (the solver's identity, expected at the solver's residual 1e-6 to 1e-11). Readout: a table per slice, plus the comparison of (λ − 1)/2 with the secant in ln M_w; the identity's residual as the row check. Discriminates: constant-λ closure (secants track (λ − 1)/2) against a varying λ; the same-source value ε_c/ζ against the measured λ.",
    "T2": "T2. The channel-count onset. Per channel at the compression minima (prod 15; s8 14; the rung reruns' 13 and 14 under their label): n_rad(κ) (5, 4, 3, 2, 2), 4a n_rad(κ) against R_50 and against the channel's own bound R_90, and the channel's inner-shell share at |ap_r| > π/4 (CORE_well 7). Readout: a 5-row table; the prediction of the Liouville picture is that the share is concentrated in channels with 4a n_rad(κ) > R_50 (κ = 1: 20 a), and a comparable share in κ = 5 (8 a) reads as the field-driven drift of (2.3). From the per-slice npz's k<κ>_shell<ss>_nk: the per-channel radial momentum distribution of the inner shell, its edge against π/4 and against the |w|-dependent threshold z* solving |w||cos 2z*| = |q|/ε at the slice's inner-root |w|.",
    "T3": "T3. The burst as the rebound's work. From the json's killing.bins_weighted at every slice: the dust's total Killing energy Σ_bins E and the (m, b] energy; the differences between consecutive slices (12→15→16→20→24→...→40) against the (m, b] energy at 20; and, where consecutive stored slices are 1 r_s apart (15→16), the field work Σ_sites (ΔN ρ + Δv J) from the two slices' applied fields and the mid-slice sources. Readout: a table of E_K,dust(T), E_(m,b](T), ΔE_K per interval, the 15→16 work estimate. Discriminates the quasi-static selection (E_K constant, the (m, b] content a fixed subset) from the dynamical lift (E_K rises with the (m, b] content).",
    "T4": "T4. The breathing period. From records_prod.json and records_s8.json: R_t(T), R_in(T), E(T) on every rooted record from 18.375 to 40; their discrete Fourier transforms over 20 to 40; the dominant period in r_s and in 4πr_s; R_50 from the 7 stored slices 20 to 40 beside. Readout: the period with its half-width, against the free-fall round trip to r_turn computed from the slice's mean bound ω/m (CORE_well 9), 2 ∫ dr/(1 − √(r_s/r)) on the PG exterior from r_turn to the root plus the core transit. Discriminates the fountain's period from a core-dynamical one.",
    "T5": "T5. The root's null flux split. At the rooted slices where the records' kernel-averaged stress is at hand (FIRSTLAW's companion rows), T_ll at the dense outer root split into the dust's part and the reference-subtracted sea's part (the dust F_M and pairs F_M of CORE_well 10.3 give the energy-flux split; the stress split only if the stress module writes per-species p_r). Readout: per slice, T_ll total, dust, sea, sign; the fraction of |T_ll| carried by content at |ap_r| > π/4 at the cut (the momentum split columns). Discriminates (5.2)'s reading: the sea's negative null energy against the dust's positive one.",
    "T6": "T6. Darkness to probes, kinematic. On every rooted slice: ∫_{R_in}^{R_t} dr/[N(r)(1 + |w(r)|)] on the dense and applied fields, in a and in r_s. Readout: one number per slice per field; compared with the slice's own root-to-root crossing time of the ejecta (the unbound front's 19.3 a per r_s). No fit.",
    "T7": "T7. The regular centre. From the npz: N_dense_sites and n_density at R = 1 to 8 a on the 7 named prod slices and the 4 rooted s8 slices; ln N − ln N(1a) against R² (expected linear by (1.2)). Readout: the slope and r² of the fit over 1 to 8 a. Already among CORE_well's pendings; its reading is (1.2)'s.",
    "T8": "T8. The halo closure (1.8). Already decidable from the printed tables: M_w(R) monotone and below E at the root on every rooted slice. Print the one line \"the mass function peaks at the root below E on all 13 rooted slices: the halo-cancellation closure does not occur at (0.2, 64)\".",
    "common": "Readout spec common to T1 to T7: per slice, one markdown table; every value labelled past the record's own limit; complex64 floor where it applies; the dense-against-applied difference as the systematic; delivered beside CORE_well v1.2 with 1 line in DELIVERIES.md; the hreduce lane recomputes T1 and T3 on 3 slices.",
}

AMENDED = {  # v1b: the red team of the push (REVIEWA_asym_redteam_v1) as the lead applied it to requirements_new.txt; printed under each test's verbatim statement
    "T1": "As amended (red team E21 and premise (b); requirements_new.txt's clock-law clause and collapse bullet): the exact clock law is a slice identity of the 3 constraint rows with no closure and no quasi-static assumption, on the dense fields an identity of the solver on the kernel-averaged sources, a row check and not a prediction; the push's lambda 1.3 at the inner root and 6.8 at R_c at T/r_s 15 were read with the unsmoothed site rho, J and the dense w, and print here as that labelled reading beside lambda from rho_ell; the edge lapse N(R_c) = N(root) exp(-integral_(R_c)^(root) ((lambda - 1)/2) d ln M_w) prints with the halo's mean lambda (6.9 at 15 from N 0.98968 and 0.41914 and M_w 103.5283 and 77.2123, the red team's number); the R <= R_50 window's fitted exponent prints against (eps_c - m a)/(2 m a), eps_c/(2 m a) and the measured (lambda - 1)/2, the requirements' comparator.",  # v1b
    "T2": "As amended (red team E2, E3, E9; requirements_new.txt's collapse, interior and radiation bullets): the onset of shedding at R_c = 4 a n_rad,max (20 a for the production's channel set; 1.222 N_B^(1/3) a = 10.5 a for the isotropic ball at 16 states per momentum) is derived for the cold Fermi ball at the Pauli bound in the deeply trapped limit |w| -> infinity, where the counter-advected set {|w| |cos 2ak| > |q|/eps} is {|ak| > pi/4}; at the computed |w| = 1.2 it begins at ak about 1.0 rad and the outward branch at a p = 0.32; the computed cores reach lattice momenta before that radius by the field-driven drift of the rows (15.1 percent of the inner shell's dust above pi/4 at T/r_s 15 with R_50 = 28.9 a, 19.9 percent of kappa = 1 and 16.6 percent of kappa = 5, whose 4 a n_rad is 8 a), so the onset is the cold ball's bound and not a description of the computed collapse; the exclusion floor is R >= 0.49 N_B^(1/3) a for 2 positive-band states per site (0.39 with all 4 components), not the push's 0.236; the compression radius prints against 1.222 N_B^(1/3) a, 4 a n_rad,max and the floor; the records' inner root is a root, not the matter's radius, and is not compared with the floor.",  # v1b
    "T3": "As amended (red team E4, E5; requirements_new.txt's radiation bullet): the dust's Killing energy at 15 is 40 from the json's bins, not the push's 45; the Killing energy per particle is set as the particle leaves the core and conserved beyond the fountain's reach, inside which the fields breathe, so the work on the orbitals between 16 and 20 is the generic case at the computed size and not an exception; the reading prints as the fields' work on the orbitals between 16 and 20 (the 13 units of (m, b] energy at 20); the energy per shed particle, 1.12 to 1.22 m, prints beside.",  # v1b
    "T4": "The definition is unchanged by the red team; requirements_new.txt cites it as the half-mass radius breathing with a period of order 8 r_s from 20 to 40, busi T4 on the records' roots, and reads the modulation's frequency at infinity in units of T_H (a few to ten T_H), which prints as 2 pi/P over T_H = 1/(4 pi r_s); the turning radius from the mean bound omega/m is not a bound on the top columns' reach after 20 (CORE_well 9.1; red team E4), so the round trip printed is the mean energy's; the window 20 to 40 holds 2.5 cycles of an 8 r_s period, so the period prints with its half-power half-width and no finer reading (the self-test's finding).",  # v1b
    "T5": "As amended (red team E6, E7; requirements_new.txt's persistence clause): the positivity 8 pi N r^2 integral f (eps - q_r)^2/eps > 0 is a theorem for a positive band-diagonal filling, which holds for the dust even on the outward branch, so the outgoing dust does not recede the root within the rows; the records' T_ll is of the reference-subtracted sources, signed, a different object, so FIRSTLAW's negative T_ll at 163 of 225 dense roots (the sigma/a 2 history, under 4 stress conditions the records do not close, the flux form of W_ext open by an order of magnitude) is no contradiction of the theorem, and 'the thermal flux's signature' is an interpretation that does not print; the push's discriminating reading of (5.2) is withdrawn as a discrimination: the table prints the energy-flux split with its signs as FIRSTLAW reads such signs, the stress split not available, and F_M, an energy flux, is not T_ll.",  # v1b
    "T6": "No change by the red team: the push's D8 stands; the number is a kinematic lower bound on the crossing time in Killing time, printed with no darkness sentence (CORE_well's must-not-print list).",  # v1b
    "T7": "No change by the red team: (1.2) stands; the reading is (1.2)'s.",  # v1b
    "T8": "The definition is unchanged by the red team ((1.8) was not among its errors); premise (b) reads the halo as carrying the cancellation at the computed size in the sense of the halo's mean lambda 6.9 at 15 (printed by T1), while M_w stays monotone and below E: the (1.8) closure tested here is M falling to E inside r_s with M(R_c) of order (eps_c/zeta) E.",  # v1b
}


class Logger:
    """prints with the elapsed seconds and appends the same line to a log file (no dates anywhere)."""

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


def fx(x, fmt="%.6g"):
    """a formatted number or '-' for None."""
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and not math.isfinite(x):
            return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
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


def wait_for_load(max_load, log):
    """waits while the 1-minute load average exceeds max_load percent of the CPUs (the box runs GPU jobs beside this; slices144_io_v1's rule)."""
    if max_load is None:
        return
    ncpu = os.cpu_count() or 1
    while True:
        la = os.getloadavg()[0]
        pct = 100.0 * la / ncpu
        if pct < max_load:
            return
        log("the box load %.1f on %d CPUs is %.0f percent (guard %.0f): waiting 60 s" % (la, ncpu, pct, max_load))
        time.sleep(60)


# ------------------------------------------------------------------------------------------------------------ the readers
def slice_paths(d, history, T):
    base = os.path.join(d, "well_%s_T%07.2f" % (history, float(T)))
    return base + ".json", base + ".npz", base + ".md"


class Slice:
    """one per-slice output of core_well_v1: its json and every array of its npz (a few MB per slice)."""

    def __init__(self, d, history, T):
        self.pj, self.pz, self.pm = slice_paths(d, history, T)
        with open(self.pj) as f:
            self.J = json.load(f)
        with np.load(self.pz) as Z:
            self.A = {k: np.asarray(Z[k]) for k in Z.files}
        self.history = history
        self.T = float(self.J.get("T_over_rs", T))
        self.label_text = str(self.J.get("history_label", history))
        self.R = self.A["R"].astype(float)
        self.N_d, self.v_d, self.w_d = (self.A[k].astype(float) for k in ("N_dense_sites", "v_dense_sites", "w_dense"))
        self.N_a, self.v_a, self.w_a = (self.A[k].astype(float) for k in ("N_applied", "v_applied", "w_applied"))
        self.M_w_d, self.M_w_a = self.A["M_w_dense"].astype(float), self.A["M_w_applied"].astype(float)
        self.eps, self.Jt = self.A["eps_tot"].astype(float), self.A["J_tot"].astype(float)
        self.eps_pairs = self.A["eps_pairs"].astype(float) if "eps_pairs" in self.A else None
        self.J_pairs = self.A["J_pairs"].astype(float) if "J_pairs" in self.A else None
        self.n_dens = self.A["n_density"].astype(float)
        self.rho_site = self.A["rho_density"].astype(float) if "rho_density" in self.A else self.eps / (4.0 * math.pi * self.R ** 2)
        self.rho_over_n = self.A["rho_over_n"].astype(float) if "rho_over_n" in self.A else None
        self.L_N_d = self.A["L_N_dense"].astype(float) if "L_N_dense" in self.A else None
        self.m, self.g, self.rs = float(self.J["m"]), float(self.J["g"]), float(self.J["rs"])
        self.ell = float(self.J["sigma_over_a"])
        rec = self.J.get("record") or {}
        self.E, self.two_gE, self.R_c = rec.get("E"), rec.get("two_gE"), rec.get("R_c")
        ro = self.J.get("roots") or {}
        self.Ro_d, self.Ri_d, self.Ro_a, self.Ri_a = (ro.get(k) for k in ("dense_outer", "dense_inner", "applied_outer", "applied_inner"))
        du = self.J.get("dust") or {}
        self.R_50, self.R_90 = du.get("R_50"), du.get("R_90")
        self.c64 = bool(self.J.get("complex64"))
        self.rooted = self.Ro_d is not None
        self.key = "%s_T%07.2f" % (history, self.T)

    def named_radii(self):
        out = []
        for lab, x, five in (("inner root", self.Ri_d, True), ("applied inner root", self.Ri_a, False), ("R_c", self.R_c, True), ("R_50", self.R_50, True),
                             ("R_90", self.R_90, True), ("applied outer root", self.Ro_a, False), ("outer root", self.Ro_d, True), ("2gE", self.two_gE, False)):
            if x is not None and self.R[0] <= x <= self.R[-1]:
                out.append({"name": lab, "R": float(x), "t1_named": five})
        out.sort(key=lambda q: q["R"])
        return out

    def gate_line(self):
        g = self.J.get("gate") or {}
        return "gate sigma |d_r v| = %s at R %s a against %s" % (fx(g.get("sigma_abs_dv_max"), "%.4f"), fx(g.get("at_R"), "%.1f"), fx(g.get("threshold"), "%g"))

    def limit_label(self):
        return str(self.J.get("limit_label", "limit label not in the json"))

    def short_label(self):
        g = self.J.get("gate") or {}
        s = "past limit" if g.get("exceeded", True) else "within limit"
        return s + ("; c64 floor" if self.c64 else "")

    def floor_clause(self):
        if not self.c64:
            return "complex128 slice (no complex64 floor)"
        fl = self.J.get("floor") or {}
        return "complex64 slice: the floor %s relative on the sources and %s on the orthonormality applies to every value read from its columns" % (fl.get("source_rel_range"), fx(fl.get("orthonormality_defect"), "%.1e"))

    def header(self):
        return ("Label: %s (%s). %s. Dense fields: %s. Applied fields: the slice's N_applied, v_applied (the record's accepted site fields). "
                "The dense-against-applied difference is the systematic." % (self.limit_label(), self.gate_line(), self.floor_clause(), (self.J.get("fields_source") or {}).get("dense")))


def read_records(d, history):
    p = os.path.join(d, "records_%s.json" % history)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------- shared arithmetic
def interp_safe(x, R, arr):
    if x is None or arr is None:
        return None
    arr = np.asarray(arr, float)
    ok = np.isfinite(arr)
    if ok.sum() < 2:
        return None
    return float(np.interp(float(x), R[ok], arr[ok]))


def trapz(xs, fs):
    xs = np.asarray(xs, float)
    fs = np.asarray(fs, float)
    return float(0.5 * np.sum((xs[1:] - xs[:-1]) * (fs[1:] + fs[:-1])))


def integrate_between(R, f, a, b):
    """int_a^b f dr with f given at the sites (linear between them); None where a limit is absent or the limits are inverted."""
    if a is None or b is None or b <= a:
        return None
    f = np.asarray(f, float)
    ok = np.isfinite(f)
    inner = R[(R > a) & (R < b) & ok]
    xs = np.concatenate([[float(a)], inner, [float(b)]])
    fs = np.interp(xs, R[ok], f[ok])
    return trapz(xs, fs)


def fit_lin(x, y):
    """the least-squares line y = s x + c: (s, c, r2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.vstack([x, np.ones_like(x)]).T
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    sst = float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), float(coef[1]), (1.0 - float(np.sum(resid ** 2)) / sst) if sst > 0 else None


def fit_origin(x, y):
    """the least-squares slope of y = s x through the origin and the uncentred r2 = 1 - SSE/sum y^2."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = float(x @ y / (x @ x)) if float(x @ x) > 0 else None
    if s is None:
        return None, None
    syy = float(y @ y)
    return s, (1.0 - float(np.sum((y - s * x) ** 2)) / syy) if syy > 0 else None


def node_diagnostic(R, N, rel=1e-9):
    """where the interpolated dense field has kinks (the second difference away from zero) the stored nodes sit; between them it is linear.
    The kink count, the spacing between consecutive kinks inside 64 a and over the chain (a), or 'every site' where no linear segment exists."""
    N = np.asarray(N, float)
    d2 = np.abs(N[2:] - 2.0 * N[1:-1] + N[:-2])
    thr = rel * max(float(np.max(np.abs(N))), 1e-300)
    kinks = R[1:-1][d2 > thr]
    linear_sites = int(np.sum(d2 <= thr))
    out = {"kinks": int(kinks.size), "linear_interior_sites": linear_sites, "threshold_abs": thr}
    if kinks.size >= 2:
        sp = np.diff(kinks)
        inside = kinks[:-1] <= 64.0
        out["spacing_inside_64a"] = {"min": float(sp[inside].min()), "median": float(np.median(sp[inside])), "max": float(sp[inside].max())} if inside.any() else None
        out["spacing_chain"] = {"min": float(sp.min()), "median": float(np.median(sp)), "max": float(sp.max())}
        out["first_kinks_a"] = [float(x) for x in kinks[:8]]
    if linear_sites == 0:
        out["reading"] = "the dense field varies at every site: no linear segment detected (the nodes lie at or below the site spacing)"
    else:
        out["reading"] = "piecewise linear between kinks a median %s a apart inside 64 a (the stored nodes' spacing there, derived from the interpolant)" % fx((out.get("spacing_inside_64a") or {}).get("median"), "%.2f")
    return out


# ------------------------------------------------------------------------------------------------------- the kernel (T1's route)
def rho_ell_shell3d(r, Rn, src, ell):
    """each site's shell energy src_n at radius R_n smeared by the three-dimensional isotropic Gaussian of standard deviation ell per axis:
    rho(r) = sum_n src_n [exp(-(r - R_n)^2/2 ell^2) - exp(-(r + R_n)^2/2 ell^2)] / (4 pi r R_n sqrt(2 pi) ell); its volume integral is sum src_n
    exactly and it is regular at r = 0 (the image term)."""
    r = np.asarray(r, float)[:, None]
    Rn = np.asarray(Rn, float)[None, :]
    K = (np.exp(-(r - Rn) ** 2 / (2.0 * ell * ell)) - np.exp(-(r + Rn) ** 2 / (2.0 * ell * ell))) / (4.0 * math.pi * r * Rn * math.sqrt(2.0 * math.pi) * ell)
    return K @ np.asarray(src, float)


def rho_ell_radial1d(r, Rn, src, ell):
    """the one-dimensional alternative: the shell energies smoothed along r by the line Gaussian (reflected at r = 0) and divided by 4 pi r^2;
    its volume integral is sum src_n exactly; it does not preserve a uniform density (an O(ell^2/r^2) bias)."""
    r = np.asarray(r, float)[:, None]
    Rn = np.asarray(Rn, float)[None, :]
    gk = (np.exp(-(r - Rn) ** 2 / (2.0 * ell * ell)) + np.exp(-(r + Rn) ** 2 / (2.0 * ell * ell))) / (math.sqrt(2.0 * math.pi) * ell)
    return (gk @ np.asarray(src, float)) / (4.0 * math.pi * r[:, 0] ** 2)


KERNELS = {"shell3d": (rho_ell_shell3d, "the three-dimensional isotropic Gaussian smearing of each site's shell (standard deviation ell per axis; a density per unit volume, regular at the centre; the volume integral equals the site sum exactly)"),
           "radial1d": (rho_ell_radial1d, "the one-dimensional Gaussian along r of standard deviation ell, reflected at r = 0, over 4 pi r^2 (the volume integral equals the site sum exactly; an O(ell^2/r^2) bias on a uniform density)")}


def kernel_average(kernel, r, Rn, src, ell, chunk=512):
    fn = KERNELS[kernel][0]
    r = np.asarray(r, float)
    out = np.empty(r.size)
    for i in range(0, r.size, chunk):
        out[i:i + chunk] = fn(r[i:i + chunk], Rn, src, ell)
    return out


def shell_integrand(kernel, r, Rn, src, ell, chunk=512):  # v1b
    """(v1b) the kernel average's volume integrand 4 pi r^2 rho_ell(r), finite at r = 0 (0 for shell3d, where it is r [G(r - R_n) - G(r + R_n)]/(sqrt(2 pi) ell R_n)
    per unit source; [G(r - R_n) + G(r + R_n)]/(sqrt(2 pi) ell) for radial1d): an even analytic function of r, so the trapezoid from r = 0 is spectrally accurate."""  # v1b
    r = np.asarray(r, float)  # v1b
    Rn = np.asarray(Rn, float)[None, :]  # v1b
    src = np.asarray(src, float)  # v1b
    out = np.empty(r.size)  # v1b
    for i in range(0, r.size, chunk):  # v1b
        rr = r[i:i + chunk][:, None]  # v1b
        Gm = np.exp(-(rr - Rn) ** 2 / (2.0 * ell * ell))  # v1b
        Gp = np.exp(-(rr + Rn) ** 2 / (2.0 * ell * ell))  # v1b
        if kernel == "shell3d":  # v1b
            K = rr * (Gm - Gp) / (math.sqrt(2.0 * math.pi) * ell * Rn)  # v1b
        else:  # v1b
            K = (Gm + Gp) / (math.sqrt(2.0 * math.pi) * ell)  # v1b
        out[i:i + chunk] = K @ src  # v1b
    return out  # v1b


def normalization_check(kernel, Rn, src, ell):
    """sum src against the volume integral of the kernel average on a fine grid from r = 0 to R_end + 8 ell in 0.25 a steps (v1b: from 0, where the volume
    integrand is finite; v1 began at 0.25 a and missed the kernel's mass inside it) and on the site grid by the trapezoid."""  # v1b
    Rn = np.asarray(Rn, float)
    rf = np.arange(0.0, Rn[-1] + 8.0 * ell + 1e-9, 0.25)  # v1b
    I_fine = trapz(rf, shell_integrand(kernel, rf, Rn, src, ell))  # v1b
    I_site = trapz(Rn, 4.0 * math.pi * Rn ** 2 * kernel_average(kernel, Rn, Rn, src, ell))
    s = float(np.sum(src))
    return {"sum_sources": s, "integral_fine_grid": I_fine, "defect_fine_rel": ((I_fine - s) / s) if s else None, "integral_site_trapezoid": I_site, "defect_site_rel": ((I_site - s) / s) if s else None,
            "fine_grid": "0 to R_end + 8 ell in 0.25 a steps from r = 0, where 4 pi r^2 rho_ell is finite: the trapezoid of that even analytic integrand is spectrally accurate (v1b; the kernel's own normalization)",  # v1b
            "site_grid": "the npz sites by the trapezoid (what a site-grid consumer sees: it misses r < 1 a and beyond the chain end)"}


def chain_end_tail_defect(r, R_end, ell):  # v1b
    """(v1b) the relative deficit 1 - rho_ell/n0 of the discrete site sum of a uniform density at r near the chain's end R_end (sites at 1, 2, ..., R_end): the
    Gaussian mass beyond the last site, integral_(R_end + 1/2)^infinity (R/r) phi((R - r)/ell) dR/ell = Q(s0) + (ell/r) phi(s0), s0 = (R_end + 1/2 - r)/ell, plus the
    Euler-Maclaurin term F'(R_end + 1/2)/24 of the plain sum (sum_n F(n) = integral_(a - 1/2)^(b + 1/2) F - [F'(b + 1/2) - F'(a - 1/2)]/24 + ...), F' negative there;
    derived, with the image term and the lower end exact for the even integrand. At r = 236, R_end = 256, ell = 4: 1.507e-7."""  # v1b
    s0 = (R_end + 0.5 - r) / ell  # v1b
    phi = math.exp(-0.5 * s0 * s0) / math.sqrt(2.0 * math.pi)  # v1b
    Q = 0.5 * math.erfc(s0 / math.sqrt(2.0))  # v1b
    Fp = (phi / ell) * (1.0 / r - ((R_end + 0.5) / r) * (s0 / ell))  # v1b
    return Q + (ell / r) * phi + Fp / 24.0  # v1b


def m3_over_m1(R, rc):  # v1b
    """(v1b) M'''/M' of the manufactured profile M = E R^3/(R^3 + c), c = r_c^3: (2 c^2 - 32 R^3 c + 20 R^6)/((R^3 + c)^2 R^2); the centred difference's leading
    truncation error of M' at spacing h is (h^2/6) M''' (the next term O(h^4)), so lambda = 4 pi R^2 rho/M' is off by lambda (h^2/6) M'''/M'."""  # v1b
    c = rc ** 3  # v1b
    return (2.0 * c * c - 32.0 * R ** 3 * c + 20.0 * R ** 6) / ((R ** 3 + c) ** 2 * R ** 2)  # v1b


def laplacian_ratio(R, rc):  # v1b
    """(v1b) (nabla^2 rho)/rho for rho proportional to (R^3 + c)^(-2), c = r_c^3 (the manufactured constant-lambda profile's density): (30 R^4 - 24 R c)/(R^3 + c)^2;
    the three-dimensional Gaussian of per-axis standard deviation ell biases rho by (ell^2/2) nabla^2 rho at leading order."""  # v1b
    c = rc ** 3  # v1b
    return (30.0 * R ** 4 - 24.0 * R * c) / (R ** 3 + c) ** 2  # v1b


# ------------------------------------------------------------------------------------------------------------------- T1
def t1_core(R, M, N, rho_ell):
    """lambda = 4 pi R^2 rho_ell/M' (M' by centred differences on the grid), (lambda - 1)/2, (ln N)' and the identity's residual
    (ln N)' M - 2 pi R^2 rho_ell + M'/2 with its size relative to the largest of the three terms."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dM = np.gradient(np.asarray(M, float), R)
        lnNp = np.gradient(np.log(np.asarray(N, float)), R)
        lam = 4.0 * math.pi * R ** 2 * rho_ell / dM
        t1, t2, t3 = lnNp * M, 2.0 * math.pi * R ** 2 * rho_ell, 0.5 * dM
        res = t1 - t2 + t3
        scale = np.maximum(np.maximum(np.abs(t1), np.abs(t2)), np.abs(t3))
        res_rel = res / scale
    return {"dM": dM, "lnNp": lnNp, "lambda": lam, "half": 0.5 * (lam - 1.0), "res": res, "res_rel": res_rel}


def _weighted_half(R, lnM, half, R1, R2):
    """the d ln M_w-weighted mean of (lambda - 1)/2 over (R1, R2]: equal to the secant d ln N/d ln M_w where the identity holds pointwise."""
    ok = np.isfinite(lnM) & np.isfinite(half)
    inner = R[(R > R1) & (R < R2) & ok]
    xs = np.concatenate([[R1], inner, [R2]])
    lm = np.interp(xs, R[ok], lnM[ok])
    hf = np.interp(xs, R[ok], half[ok])
    dl = np.diff(lm)
    den = float(np.sum(dl))
    if den == 0.0:
        return None
    return float(np.sum(0.5 * (hf[1:] + hf[:-1]) * dl) / den)


def t1_slice(S, a, arrays, cache):
    R = S.R
    kern = a.kernel
    rho_ell = kernel_average(kern, R, R, S.eps, S.ell)
    J_ell = kernel_average(kern, R, R, S.Jt, S.ell)
    cache[S.key] = rho_ell
    norm = normalization_check(kern, R, S.eps, S.ell)
    D = t1_core(R, S.M_w_d, S.N_d, rho_ell)
    Ap = t1_core(R, S.M_w_a, S.N_a, rho_ell)
    J_site = S.Jt / (4.0 * math.pi * R ** 2)  # v1b: the unsmoothed site current density J_tot/(4 pi R^2 a)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_src_d = rho_ell / (rho_ell + S.w_d * J_ell)
        lam_src_a = rho_ell / (rho_ell + S.w_a * J_ell)
        lam_site_d = S.rho_site / (S.rho_site + S.w_d * J_site)  # v1b: the push's (1.6) construction with the unsmoothed site sources and the dense w (E21: a labelled reading)
        src_d = 4.0 * math.pi * R ** 2 * (rho_ell + S.w_d * J_ell)
        R1_rel_d = (D["dM"] - src_d) / np.maximum(np.abs(D["dM"]), np.abs(src_d))
        R3_d = S.w_d * D["lnNp"] + 4.0 * math.pi * S.g * R * J_ell
        R3_rel_d = R3_d / np.maximum(np.abs(S.w_d * D["lnNp"]), np.abs(4.0 * math.pi * S.g * R * J_ell))
        res_L_rel = None
        if S.L_N_d is not None:
            sgn = np.sign(np.gradient(S.N_d, R))
            sgn[sgn == 0] = 1.0
            lnNp_L = sgn / S.L_N_d
            tL = lnNp_L * S.M_w_d
            res_L = tL - 2.0 * math.pi * R ** 2 * rho_ell + 0.5 * D["dM"]
            res_L_rel = res_L / np.maximum(np.maximum(np.abs(tL), 2.0 * math.pi * R ** 2 * np.abs(rho_ell)), 0.5 * np.abs(D["dM"]))
        same_src = (S.rho_over_n / S.m) if S.rho_over_n is not None else None
    lim = S.Ro_d if S.Ro_d is not None else (S.R_90 if S.R_90 is not None else R[-1])
    inside = R <= lim

    def maxrel(arr):
        if arr is None:
            return None
        sel = inside & np.isfinite(arr)
        return float(np.max(np.abs(arr[sel]))) if sel.any() else None

    with np.errstate(divide="ignore", invalid="ignore"):  # v1b
        lnM_d, lnM_a = np.log(S.M_w_d), np.log(S.M_w_a)
        lnN_d, lnN_a = np.log(S.N_d), np.log(S.N_a)
    rows = []
    for q in S.named_radii():
        x = q["R"]
        row = dict(q)
        row.update({"rho_ell": interp_safe(x, R, rho_ell), "rho_site": interp_safe(x, R, S.rho_site), "M_w_dense": interp_safe(x, R, S.M_w_d), "dM_w_dense": interp_safe(x, R, D["dM"]),
                    "lambda_dense": interp_safe(x, R, D["lambda"]), "half_dense": interp_safe(x, R, D["half"]), "lnN_prime_dense": interp_safe(x, R, D["lnNp"]),
                    "residual_dense": interp_safe(x, R, D["res"]), "residual_rel_dense": interp_safe(x, R, D["res_rel"]),
                    "M_w_applied": interp_safe(x, R, S.M_w_a), "dM_w_applied": interp_safe(x, R, Ap["dM"]), "lambda_applied": interp_safe(x, R, Ap["lambda"]), "half_applied": interp_safe(x, R, Ap["half"]),
                    "residual_applied": interp_safe(x, R, Ap["res"]), "residual_rel_applied": interp_safe(x, R, Ap["res_rel"]),
                    "lambda_sources_dense": interp_safe(x, R, lam_src_d), "lambda_sources_applied": interp_safe(x, R, lam_src_a),
                    "lambda_site_sources_dense": interp_safe(x, R, lam_site_d),  # v1b
                    "same_source_lambda_site": interp_safe(x, R, same_src), "N_dense": interp_safe(x, R, S.N_d), "N_applied": interp_safe(x, R, S.N_a),
                    "residual_rel_dense_lnN_from_L_N": interp_safe(x, R, res_L_rel)})
        row["lambda_dense_minus_applied"] = (row["lambda_dense"] - row["lambda_applied"]) if (row["lambda_dense"] is not None and row["lambda_applied"] is not None) else None
        rows.append(row)
    five = [r_ for r_ in rows if r_["t1_named"]]
    intervals = []
    for i in range(len(five) - 1):
        p, q = five[i], five[i + 1]
        dd = math.log(q["M_w_dense"]) - math.log(p["M_w_dense"])
        da = math.log(q["M_w_applied"]) - math.log(p["M_w_applied"])
        iv = {"from": p["name"], "to": q["name"], "R_from": p["R"], "R_to": q["R"],
              "secant_dense": ((math.log(q["N_dense"]) - math.log(p["N_dense"])) / dd) if dd != 0 else None,
              "secant_applied": ((math.log(q["N_applied"]) - math.log(p["N_applied"])) / da) if da != 0 else None,
              "weighted_half_dense": _weighted_half(R, lnM_d, D["half"], p["R"], q["R"]), "weighted_half_applied": _weighted_half(R, lnM_a, Ap["half"], p["R"], q["R"]),
              "half_at_ends_dense": [p["half_dense"], q["half_dense"]], "delta_ln_M_w_dense": dd, "delta_ln_N_dense": math.log(q["N_dense"]) - math.log(p["N_dense"])}
        iv["secant_dense_minus_applied"] = (iv["secant_dense"] - iv["secant_applied"]) if (iv["secant_dense"] is not None and iv["secant_applied"] is not None) else None
        intervals.append(iv)
        p["secant_to_next_dense"], p["secant_to_next_applied"], p["weighted_half_to_next_dense"] = iv["secant_dense"], iv["secant_applied"], iv["weighted_half_dense"]
    # v1b: the halo R_c to the outer root (red team premise (b)): its secant, the mean lambda 1 + 2 secant and the edge lapse against the exact law
    halo = {"available": False, "reason": "R_c absent, no dense outer root, or R_c at or outside the root"}  # v1b
    if S.R_c is not None and S.Ro_d is not None and S.R_c < S.Ro_d:  # v1b
        N_c, N_t = interp_safe(S.R_c, R, S.N_d), interp_safe(S.Ro_d, R, S.N_d)  # v1b
        M_c, M_t = interp_safe(S.R_c, R, S.M_w_d), interp_safe(S.Ro_d, R, S.M_w_d)  # v1b
        Na_c, Na_t = interp_safe(S.R_c, R, S.N_a), interp_safe(S.Ro_d, R, S.N_a)  # v1b
        Ma_c, Ma_t = interp_safe(S.R_c, R, S.M_w_a), interp_safe(S.Ro_d, R, S.M_w_a)  # v1b
        if all(v_ is not None and v_ > 0 for v_ in (N_c, N_t, M_c, M_t)) and M_t != M_c:  # v1b
            dlm = math.log(M_t) - math.log(M_c)  # v1b
            sec = (math.log(N_t) - math.log(N_c)) / dlm  # v1b
            wh = _weighted_half(R, lnM_d, D["half"], S.R_c, S.Ro_d)  # v1b
            sec_a = ((math.log(Na_t) - math.log(Na_c)) / (math.log(Ma_t) - math.log(Ma_c))) if (all(v_ is not None and v_ > 0 for v_ in (Na_c, Na_t, Ma_c, Ma_t)) and Ma_t != Ma_c) else None  # v1b
            halo = {"available": True, "R_c": float(S.R_c), "R_root": float(S.Ro_d), "N_R_c_dense": N_c, "N_root_dense": N_t, "M_w_R_c_dense": M_c, "M_w_root_dense": M_t, "delta_ln_M_w_dense": dlm,  # v1b
                    "secant_dense": sec, "mean_lambda_dense": 1.0 + 2.0 * sec, "secant_applied": sec_a, "mean_lambda_applied": (1.0 + 2.0 * sec_a) if sec_a is not None else None,  # v1b
                    "weighted_half_dense": wh, "edge_lapse_measured": N_c / N_t, "edge_lapse_law": (math.exp(-wh * dlm) if wh is not None else None),  # v1b
                    "edge_lapse_law_over_measured": ((math.exp(-wh * dlm) / (N_c / N_t)) if wh is not None else None), "M_w_R_c_over_root": M_c / M_t}  # v1b
    # v1b: the R <= R_50 window: the json's fitted exponent against the two forms and the measured (lambda - 1)/2 (requirements_new.txt's comparator)
    cfw = ((S.J.get("core_fit") or {}).get("windows") or {}).get("R50") or {}
    win = {"available": False, "reason": "no R_50"}  # v1b
    if S.R_50 is not None:  # v1b
        N_0, M_0 = float(S.N_d[0]), float(S.M_w_d[0])  # v1b
        N_5, M_5 = interp_safe(S.R_50, R, S.N_d), interp_safe(S.R_50, R, S.M_w_d)  # v1b
        cum = S.A["dust_enclosed"].astype(float) if "dust_enclosed" in S.A else None  # v1b
        c_0 = float(cum[0]) if cum is not None else None  # v1b
        c_5 = interp_safe(S.R_50, R, cum) if cum is not None else None  # v1b
        if all(v_ is not None and v_ > 0 for v_ in (N_0, M_0, N_5, M_5)) and M_5 != M_0:  # v1b
            dlm = math.log(M_5) - math.log(M_0)  # v1b
            win = {"available": True, "R_from": float(R[0]), "R_50": float(S.R_50), "secant_lnM_dense": (math.log(N_5) - math.log(N_0)) / dlm, "weighted_half_dense": _weighted_half(R, lnM_d, D["half"], R[0], S.R_50),  # v1b
                   "d_lnM_w_over_d_lnNenc": (dlm / (math.log(c_5) - math.log(c_0))) if (c_0 is not None and c_5 is not None and c_0 > 0 and c_5 > 0 and c_5 != c_0) else None,  # v1b
                   "json_slope_lnN_vs_lnNenc_dense": (cfw.get("fit_dense") or {}).get("slope"), "json_slope_lnN_vs_lnNenc_applied": (cfw.get("fit_applied") or {}).get("slope"),  # v1b
                   "epsilon_c": cfw.get("epsilon_c_total"), "exponent_eps_minus_m_over_2m": cfw.get("exponent_eps_minus_m_over_2m"), "exponent_eps_over_2m": cfw.get("exponent_eps_over_2m")}  # v1b
            if win["json_slope_lnN_vs_lnNenc_dense"] is not None and win["d_lnM_w_over_d_lnNenc"]:  # v1b
                win["json_slope_converted_to_lnM_w"] = win["json_slope_lnN_vs_lnNenc_dense"] / win["d_lnM_w_over_d_lnNenc"]  # v1b
            else:  # v1b
                win["json_slope_converted_to_lnM_w"] = None  # v1b
    rec = S.J.get("record") or {}
    solver = rec.get("solver") or {}
    out = {"slice": S.key, "history": S.history, "history_label": S.label_text, "T_over_rs": S.T, "label": S.short_label(), "limit_label": S.limit_label(), "complex64": S.c64, "ell": S.ell,
           "rho_ell_route": "no stored array holds rho_ell: the site sources eps_tot convolved with the %s kernel (%s) at ell = %g; J_ell the same for J_tot (derived check columns only)" % (kern, KERNELS[kern][1], S.ell),
           "normalization": norm, "node_diagnostic_dense_N": node_diagnostic(R, S.N_d), "node_diagnostic_dense_v": node_diagnostic(R, S.v_d),
           "record_solver_residuals": {"residual_W": solver.get("residual_W"), "residual_invN": solver.get("residual_invN"), "module": solver.get("module")},
           "inside_radius_for_maxima": float(lim), "max_abs_residual_rel_inside": {"dense": maxrel(D["res_rel"]), "applied": maxrel(Ap["res_rel"]), "dense_lnN_from_stored_L_N": maxrel(res_L_rel)},
           "row_R1_max_rel_inside_dense": maxrel(R1_rel_d), "row_R3_max_rel_inside_dense": maxrel(R3_rel_d),
           "same_source_lambda_window_R50": (cfw.get("epsilon_c_total") / S.m) if cfw.get("epsilon_c_total") is not None else None,
           "halo": halo, "window": win,  # v1b
           "label_as_amended": "a slice identity of the solver on the kernel-averaged sources, a row check and not a prediction (red team; requirements_new.txt); the site-source lambda column is the push's (1.6) reading, labelled (E21)",  # v1b
           "radii": rows, "intervals": intervals}
    for k_, v_ in (("rho_ell", rho_ell), ("J_ell", J_ell), ("lambda_dense", D["lambda"]), ("lambda_applied", Ap["lambda"]), ("residual_dense", D["res"]), ("residual_rel_dense", D["res_rel"]),
                   ("residual_applied", Ap["res"]), ("residual_rel_applied", Ap["res_rel"]), ("lambda_sources_dense", lam_src_d), ("lambda_site_sources_dense", lam_site_d), ("dM_w_dense", D["dM"]), ("lnN_prime_dense", D["lnNp"])):  # v1b
        arrays["T1_%s_%s" % (S.key, k_)] = np.asarray(v_, float)
    return out


def t1_md(S, out):
    hl = out.get("halo") or {}  # v1b
    wn = out.get("window") or {}  # v1b
    if hl.get("available"):  # v1b
        halo_text = ("secant d ln N/d ln M_w %s dense (%s applied), the halo's mean lambda 1 + 2 secant %s (%s applied; the red team's 6.9 at 15), the edge lapse N(R_c)/N(root) %s against the exact law's "
                     "exp(-integral ((lambda - 1)/2) d ln M_w) %s from the pointwise lambda(rho_ell) (ratio %s; the difference is the identity's integrated residual over the halo), M_w(R_c)/M_w(root) %s" % (
                         fx(hl["secant_dense"], "%.4f"), fx(hl["secant_applied"], "%.4f"), fx(hl["mean_lambda_dense"], "%.3f"), fx(hl["mean_lambda_applied"], "%.3f"), fx(hl["edge_lapse_measured"], "%.5f"),
                         fx(hl["edge_lapse_law"], "%.5f"), fx(hl["edge_lapse_law_over_measured"], "%.4f"), fx(hl["M_w_R_c_over_root"], "%.4f")))  # v1b
    else:  # v1b
        halo_text = "no halo interval (%s)" % hl.get("reason")  # v1b
    if wn.get("available"):  # v1b
        win_text = ("the json's fitted slope of ln N against ln N_< over R <= R_50 %s dense, %s applied, at eps_c %s, against (eps_c - m a)/(2 m a) %s and eps_c/(2 m a) %s; the exact law's d ln M_w-weighted mean of "
                    "(lambda - 1)/2 over %g a to R_50 %s (dense), the secant d ln N/d ln M_w over the same window %s, d ln M_w/d ln N_< over it %s, so the count slope converted to ln M_w is %s" % (
                        fx(wn["json_slope_lnN_vs_lnNenc_dense"], "%.4f"), fx(wn["json_slope_lnN_vs_lnNenc_applied"], "%.4f"), fx(wn["epsilon_c"], "%.5f"), fx(wn["exponent_eps_minus_m_over_2m"], "%.4f"), fx(wn["exponent_eps_over_2m"], "%.4f"),
                        wn["R_from"], fx(wn["weighted_half_dense"], "%.4f"), fx(wn["secant_lnM_dense"], "%.4f"), fx(wn["d_lnM_w_over_d_lnNenc"], "%.4f"), fx(wn["json_slope_converted_to_lnM_w"], "%.4f")))  # v1b
    else:  # v1b
        win_text = "no window (%s)" % wn.get("reason")  # v1b
    L = ["### T1: %s at T/r_s %.3f" % (out["history_label"], out["T_over_rs"]), "", S.header(), "",
         "rho_ell route: %s. Normalization: sum eps_tot %.6f; the kernel's volume integral on the fine grid from r = 0 %.6f (relative defect %s), on the site grid by the trapezoid %.6f (%s). "
         "The record's own solver residuals: residual_W %s, residual_invN %s (%s). Largest |identity residual| relative to its largest term inside R <= %.3f a: dense %s, applied %s, dense with (ln N)' from the stored L_N %s; "
         "the rows' own residuals there (derived): R1 M_w' - 4 pi R^2 (rho_ell + w J_ell) %s, R3 w (ln N)' + 4 pi g R J_ell %s. The dense N: %s; the dense v: %s. The same-source lambda of the R <= R_50 window, eps_c/zeta = eps_c/(m a): %s. Label as amended: %s." % (
             out["rho_ell_route"], out["normalization"]["sum_sources"], out["normalization"]["integral_fine_grid"], fx(out["normalization"]["defect_fine_rel"], "%.2e"), out["normalization"]["integral_site_trapezoid"], fx(out["normalization"]["defect_site_rel"], "%.2e"),
             fx(out["record_solver_residuals"]["residual_W"], "%.2e"), fx(out["record_solver_residuals"]["residual_invN"], "%.2e"), out["record_solver_residuals"]["module"], out["inside_radius_for_maxima"],
             fx(out["max_abs_residual_rel_inside"]["dense"], "%.3e"), fx(out["max_abs_residual_rel_inside"]["applied"], "%.3e"), fx(out["max_abs_residual_rel_inside"]["dense_lnN_from_stored_L_N"], "%.3e"),
             fx(out["row_R1_max_rel_inside_dense"], "%.3e"), fx(out["row_R3_max_rel_inside_dense"], "%.3e"), out["node_diagnostic_dense_N"]["reading"], out["node_diagnostic_dense_v"]["reading"], fx(out["same_source_lambda_window_R50"], "%.4f"), out["label_as_amended"]), "",  # v1b
         "The R <= R_50 window (v1b; requirements_new.txt's comparator): %s. The halo R_c to the outer root (v1b; red team premise (b)): %s." % (win_text, halo_text), "",  # v1b
         "| radius | R (a) | rho_ell (a^-3) | M_w dense | M_w' dense | lambda = 4 pi R^2 rho_ell/M_w' | (lambda - 1)/2 | secant d ln N/d ln M_w to the next named radius | d ln M_w-weighted mean of (lambda - 1)/2 over that interval | identity residual (absolute; relative) | lambda from the sources rho_ell/(rho_ell + w J_ell) (derived) | lambda from the unsmoothed site sources with the dense w (the push's (1.6) reading, E21; v1b) | same-source eps_c/zeta = rho/(m n) at the site | applied: lambda; (lambda - 1)/2; secant; residual (relative) | lambda dense - applied | label |",  # v1b
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # v1b
    for q in out["radii"]:
        L.append("| %s%s | %.3f | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s | %s | %s | %s; %s; %s; %s | %s | %s |" % (  # v1b
            q["name"], "" if q["t1_named"] else " (extra)", q["R"], fx(q["rho_ell"], "%.4e"), fx(q["M_w_dense"], "%.4f"), fx(q["dM_w_dense"], "%.4f"), fx(q["lambda_dense"], "%.4f"), fx(q["half_dense"], "%.4f"),
            fx(q.get("secant_to_next_dense"), "%.4f"), fx(q.get("weighted_half_to_next_dense"), "%.4f"), fx(q["residual_dense"], "%+.3e"), fx(q["residual_rel_dense"], "%.2e"), fx(q["lambda_sources_dense"], "%.4f"),
            fx(q["lambda_site_sources_dense"], "%.4f"),  # v1b
            fx(q["same_source_lambda_site"], "%.4f"), fx(q["lambda_applied"], "%.4f"), fx(q["half_applied"], "%.4f"), fx(q.get("secant_to_next_applied"), "%.4f"), fx(q["residual_rel_applied"], "%.2e"),
            fx(q["lambda_dense_minus_applied"], "%+.4f"), out["label"]))
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------- T2
def q_over_eps(z, m):
    s = np.sin(z)
    return s * np.cos(z) / np.sqrt(m * m + s * s)


def z_star(absw, m):
    """the smallest z in (pi/4, pi/2] with |w| |cos 2z| >= q(z)/eps(z), q = sin z cos z, eps = sqrt(m^2 + sin^2 z): the edge of the counter-advected set of inward momenta (2.2)."""
    if absw is None or absw <= 0:
        return None
    zs = np.linspace(PI4, 0.5 * math.pi, 20001)[1:]
    f = absw * np.abs(np.cos(2.0 * zs)) - q_over_eps(zs, m)
    idx = np.nonzero(f >= 0)[0]
    if idx.size == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(zs[0])
    lo, hi = float(zs[i - 1]), float(zs[i])
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if absw * abs(math.cos(2.0 * mid)) - float(q_over_eps(mid, m)) >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def z_out(absw, m):  # v1b
    """(v1b) the smallest z in (0, pi/4) with q(z)/eps(z) >= |w| cos 2z, q = sin z cos z, eps = sqrt(m^2 + sin^2 z): the onset of the outward branch for outward momenta,
    dx/dt = N q/eps - N |w| cos 2z > 0 (CORE_well 7's kinematic note: 0.32 rad at |w| = 1 for m a 0.2; the red team's E3 with the massless symbol: 0.33 rad at |w| = 1.2)."""  # v1b
    if absw is None or absw <= 0:  # v1b
        return None  # v1b
    zs = np.linspace(0.0, PI4, 20001)[1:]  # v1b
    f = q_over_eps(zs, m) - absw * np.cos(2.0 * zs)  # v1b
    idx = np.nonzero(f >= 0)[0]  # v1b
    if idx.size == 0:  # v1b
        return None  # v1b
    i = int(idx[0])  # v1b
    if i == 0:  # v1b
        return float(zs[0])  # v1b
    lo, hi = float(zs[i - 1]), float(zs[i])  # v1b
    for _ in range(80):  # v1b
        mid = 0.5 * (lo + hi)  # v1b
        if float(q_over_eps(mid, m)) - absw * math.cos(2.0 * mid) >= 0:  # v1b
            hi = mid  # v1b
        else:  # v1b
            lo = mid  # v1b
    return 0.5 * (lo + hi)  # v1b


def momentum_edge(nk, p, thresholds):
    """nk on the grid (fftfreq order; (1/M) sum nk the shell's dust number), p the reduced-zone momentum of each grid point: the shares above each
    threshold in |p| (the producer's weights, unclipped), the quantiles p_50, p_90, p_99 of |p|, <cos 2p> and <|p|> (weights clipped at 0);
    (v1b) the directional shares, p > threshold and p < -threshold, p > 0 taken as outward on the chain's index (core_kinematics_v2's reduced zone)."""  # v1b
    nk = np.asarray(nk, float)
    p = np.asarray(p, float)
    M = nk.size
    tot = float(nk.sum())
    ap = np.abs(p)
    out = {"number": tot / M, "negative_weight": float(-nk[nk < 0].sum() / M), "grid_points": int(M)}
    out["shares_above"] = {name: (float(nk[ap > thr].sum() / tot) if tot else None) for name, thr in thresholds.items()}
    out["shares_outward_above"] = {name: (float(nk[p > thr].sum() / tot) if tot else None) for name, thr in thresholds.items()}  # v1b
    out["shares_inward_above"] = {name: (float(nk[p < -thr].sum() / tot) if tot else None) for name, thr in thresholds.items()}  # v1b
    w = np.clip(nk, 0.0, None)
    sw = float(w.sum())
    if sw > 0:
        o = np.argsort(ap, kind="stable")
        cw = np.cumsum(w[o]) / sw
        aps = ap[o]

        def quant(qq):
            i = int(np.searchsorted(cw, qq))
            return float(aps[min(i, M - 1)])

        out.update({"p_50": quant(0.5), "p_90": quant(0.9), "p_99": quant(0.99), "cos2p": float(np.sum(w * np.cos(2.0 * p)) / sw), "mean_abs_p": float(np.sum(w * ap) / sw)})
    return out


def t2_slice(S, a):
    R = S.R
    m = S.m
    inner = S.Ri_d
    w_in_d = abs(interp_safe(inner, R, S.w_d)) if inner is not None else None
    w_in_a = abs(interp_safe(inner, R, S.w_a)) if inner is not None else None
    lim = S.Ro_d if S.Ro_d is not None else (S.R_50 if S.R_50 is not None else R[-1])
    sel = R <= lim
    aw = np.abs(S.w_d[sel])
    i_max = int(np.argmax(aw))
    w_max, R_wmax = float(aw[i_max]), float(R[sel][i_max])
    zs = {"inner_root_dense_w": z_star(w_in_d, m), "inner_root_applied_w": z_star(w_in_a, m), "max_w_inside": z_star(w_max, m), "w_1.2": z_star(1.2, m)}
    zo = {"inner_root_dense_w": z_out(w_in_d, m), "inner_root_applied_w": z_out(w_in_a, m), "max_w_inside": z_out(w_max, m), "w_1.0": z_out(1.0, m), "w_1.2": z_out(1.2, m)}  # v1b
    thr = {"pi/4": PI4}
    for nm in ("inner_root_dense_w", "inner_root_applied_w", "max_w_inside"):
        if zs[nm] is not None:
            thr["z*_" + nm] = zs[nm]
    for nm in ("inner_root_dense_w", "max_w_inside"):  # v1b
        if zo[nm] is not None:  # v1b
            thr["z_out_" + nm] = zo[nm]  # v1b
    p = S.A["p_reduced"].astype(float)
    R50 = S.R_50
    chans = []
    for ch in S.J.get("channels", []):
        if not ch.get("n_dust"):
            continue
        kap = int(ch["kappa"])
        n_rad = int(ch["n_dust"])
        bound = ((ch.get("killing") or {}).get("bound") or {})
        sh0 = (ch.get("shells") or [{}])[0]
        key = "k%d_shell00_nk" % kap
        edge = momentum_edge(S.A[key], p, thr) if key in S.A else None
        row = {"kappa": kap, "n_rad": n_rad, "n_rad_declared": N_RAD_DECLARED.get(kap), "four_a_n_rad": 4.0 * n_rad, "R_50": R50, "four_a_n_rad_minus_R_50": (4.0 * n_rad - R50) if R50 is not None else None,
               "bound_R_90": bound.get("R_90"), "four_a_n_rad_over_bound_R_90": (4.0 * n_rad / bound["R_90"]) if bound.get("R_90") else None, "bound_mean_omega_over_m": bound.get("mean_omega_over_m"),
               "share_json": sh0.get("frac_abs_p_above"), "cos2k_json": sh0.get("cos2k"), "shell_number_json": sh0.get("number"), "edge": edge,
               "share_check_npz_minus_json": (edge["shares_above"]["pi/4"] - sh0["frac_abs_p_above"]) if (edge and edge["shares_above"].get("pi/4") is not None and sh0.get("frac_abs_p_above") is not None) else None}
        chans.append(row)
    # v1b: the radii against the regime's bounds (requirements_new.txt's collapse bullet; red team E2, E9)
    tot_w = (S.J.get("dust") or {}).get("total_weighted")  # v1b
    N_B = float(tot_w) if tot_w is not None else float(sum(16.0 * c_["kappa"] * c_["n_rad"] for c_ in chans))  # v1b
    n_rad_max = max((c_["n_rad"] for c_ in chans), default=None)  # v1b
    cube = N_B ** (1.0 / 3.0) if N_B > 0 else 0.0  # v1b
    regime = {"N_B": N_B, "N_B_source": "the json's dust.total_weighted" if tot_w is not None else "derived: sum over the channels of 16 kappa n_dust (no total_weighted in the json)",  # v1b
              "ball_radius_1222": C_R * cube, "four_a_n_rad_max": (4.0 * n_rad_max) if n_rad_max is not None else None, "floor_2_states_per_site": FLOOR_2 * cube, "floor_4_components": FLOOR_4 * cube,  # v1b
              "R_50_over_four_a_n_rad_max": (R50 / (4.0 * n_rad_max)) if (R50 is not None and n_rad_max) else None, "R_50_over_ball": (R50 / (C_R * cube)) if (R50 is not None and cube > 0) else None,  # v1b
              "R_50_over_floor_2": (R50 / (FLOOR_2 * cube)) if (R50 is not None and cube > 0) else None, "R_90": S.R_90, "inner_root_dense": inner,  # v1b
              "note": "the record's inner root is a root, not the matter's radius, and is not compared with the floor (E2)"}  # v1b
    return {"slice": S.key, "history": S.history, "history_label": S.label_text, "T_over_rs": S.T, "label": S.short_label(), "limit_label": S.limit_label(), "complex64": S.c64, "inner_root_R_dense": inner,
            "abs_w_inner_root_dense": w_in_d, "abs_w_at_dense_inner_root_applied": w_in_a, "abs_w_max_inside": w_max, "R_of_abs_w_max": R_wmax, "z_star": zs, "z_out": zo, "pi_over_4": PI4, "R_50": R50, "R_90": S.R_90,  # v1b
            "shell0_R_range": [float(R[0]), float(R[int(S.A["shell_edges_sites"][1]) - 1])] if "shell_edges_sites" in S.A else None, "channels": chans, "regime": regime,  # v1b
            "sign_convention": "p > 0 taken as outward on the chain's index (core_kinematics_v2's reduced zone p = k mod pi); both halves print so the reader may swap them",  # v1b
            "prediction_verbatim": "the prediction of the Liouville picture is that the share is concentrated in channels with 4a n_rad(κ) > R_50 (κ = 1: 20 a), and a comparable share in κ = 5 (8 a) reads as the field-driven drift of (2.3)",
            "reading_as_amended": AMENDED["T2"]}  # v1b


def t2_md(S, out):
    rg = out["regime"]  # v1b
    zo = out["z_out"]  # v1b
    L = ["### T2: %s at T/r_s %.3f" % (out["history_label"], out["T_over_rs"]), "", S.header(), "",
         "The slice's inner root (dense) %s a with |w| dense %s (applied w there %s); the largest |w| inside the outer root %s at R %s a. z* solving |w| |cos 2z*| = q/eps (q = sin z cos z, eps = sqrt((m a)^2 + sin^2 z), m a = %g): at the dense inner-root |w| %s rad, at the applied w there %s rad, at the largest |w| inside %s rad; at |w| = 1.2 %s rad; pi/4 = %.4f rad. R_50 %s a, R_90 %s a. The inner shell is R %s a (the npz's shell 0). %s." % (
             fx(out["inner_root_R_dense"], "%.3f"), fx(out["abs_w_inner_root_dense"], "%.4f"), fx(out["abs_w_at_dense_inner_root_applied"], "%.4f"), fx(out["abs_w_max_inside"], "%.4f"), fx(out["R_of_abs_w_max"], "%.1f"), S.m,
             fx(out["z_star"]["inner_root_dense_w"], "%.4f"), fx(out["z_star"]["inner_root_applied_w"], "%.4f"), fx(out["z_star"]["max_w_inside"], "%.4f"), fx(out["z_star"]["w_1.2"], "%.4f"), PI4, fx(out["R_50"], "%.3f"), fx(out["R_90"], "%.3f"),
             ("%.0f to %.0f" % tuple(out["shell0_R_range"])) if out["shell0_R_range"] else "-", out["prediction_verbatim"]), "",
         "The outward branch's onset z_out solving q/eps = |w| cos 2z below pi/4 (v1b; CORE_well 7: 0.32 rad at |w| = 1; red team E3): at the dense inner-root |w| %s rad, at the applied w there %s rad, at the largest |w| inside %s rad, at |w| = 1 %s rad, at |w| = 1.2 %s rad. Sign convention of the directional shares: %s." % (
             fx(zo["inner_root_dense_w"], "%.4f"), fx(zo["inner_root_applied_w"], "%.4f"), fx(zo["max_w_inside"], "%.4f"), fx(zo["w_1.0"], "%.4f"), fx(zo["w_1.2"], "%.4f"), out["sign_convention"]), "",  # v1b
         "Radii against the regime's bounds (v1b; requirements_new.txt's collapse bullet, red team E2 and E9): N_B %s (%s); 1.222 N_B^(1/3) a = %s a (the isotropic ball at 16 states per momentum); 4 a n_rad,max = %s a (the channel set's onset); the exclusion floor 0.49 N_B^(1/3) a = %s a (2 positive-band states per site; %s a with all 4 components at 0.39); R_50 %s a is %s of 4 a n_rad,max, %s of 1.222 N_B^(1/3) a and %s of the floor; R_90 %s a; the record's inner root %s a: %s." % (
             fx(rg["N_B"], "%.1f"), rg["N_B_source"], fx(rg["ball_radius_1222"], "%.2f"), fx(rg["four_a_n_rad_max"], "%.0f"), fx(rg["floor_2_states_per_site"], "%.2f"), fx(rg["floor_4_components"], "%.2f"), fx(out["R_50"], "%.3f"),
             fx(rg["R_50_over_four_a_n_rad_max"], "%.2f"), fx(rg["R_50_over_ball"], "%.2f"), fx(rg["R_50_over_floor_2"], "%.2f"), fx(rg["R_90"], "%.3f"), fx(rg["inner_root_dense"], "%.3f"), rg["note"]), "",  # v1b
         "Reading as amended: %s" % out["reading_as_amended"], "",  # v1b
         "| kappa | n_rad (dust columns) | 4a n_rad (a) | R_50 (a) | 4a n_rad - R_50 (a) | bound R_90 of the channel (a) | 4a n_rad over bound R_90 | inner-shell share at |ap| > pi/4 (json) | recomputed from the npz n(k) minus json | share at |ap| > z*(inner-root |w|) | share at |ap| > z*(max |w|) | outward p > z_out(inner-root |w|); inward p < -z*(inner-root |w|) (v1b) | p_50; p_90; p_99 of |ap| (rad) | <cos 2ap> (npz; json) | shell dust number | label |",  # v1b
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # v1b
    for c in out["channels"]:
        e = c["edge"] or {}
        sh = e.get("shares_above") or {}
        so, si = e.get("shares_outward_above") or {}, e.get("shares_inward_above") or {}  # v1b
        L.append("| %d | %d (declared %s) | %.1f | %s | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s; %s; %s | %s; %s | %s | %s |" % (  # v1b
            c["kappa"], c["n_rad"], c["n_rad_declared"], c["four_a_n_rad"], fx(c["R_50"], "%.3f"), fx(c["four_a_n_rad_minus_R_50"], "%+.3f"), fx(c["bound_R_90"], "%.3f"), fx(c["four_a_n_rad_over_bound_R_90"], "%.3f"),
            fx(c["share_json"], "%.4f"), fx(c["share_check_npz_minus_json"], "%+.1e"), fx(sh.get("z*_inner_root_dense_w"), "%.4f"), fx(sh.get("z*_max_w_inside"), "%.4f"),
            fx(so.get("z_out_inner_root_dense_w"), "%.4f"), fx(si.get("z*_inner_root_dense_w"), "%.4f"),  # v1b
            fx(e.get("p_50"), "%.3f"), fx(e.get("p_90"), "%.3f"), fx(e.get("p_99"), "%.3f"),
            fx(e.get("cos2p"), "%+.4f"), fx(c["cos2k_json"], "%+.4f"), fx(e.get("number"), "%.4f"), out["label"]))
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------- T3
def _bin(bins, label):
    for b_ in bins:
        if b_.get("label") == label:
            return b_
    return {}


def field_work(S0, S1, which):
    """sum over the sites of (Delta N eps + Delta v J) between two stored slices with the mid-slice sources (the trapezoid in the clock),
    the two one-end brackets, and the dust's part (eps_tot - eps_pairs, J_tot - J_pairs)."""
    N0, v0 = (S0.N_a, S0.v_a) if which == "applied" else (S0.N_d, S0.v_d)
    N1, v1 = (S1.N_a, S1.v_a) if which == "applied" else (S1.N_d, S1.v_d)
    dN, dv = N1 - N0, v1 - v0
    out = {"fields": which, "from": S0.key, "to": S1.key}
    for part, e0, e1, j0, j1 in (("total", S0.eps, S1.eps, S0.Jt, S1.Jt),
                                 ("dust", (S0.eps - S0.eps_pairs) if S0.eps_pairs is not None else None, (S1.eps - S1.eps_pairs) if S1.eps_pairs is not None else None,
                                  (S0.Jt - S0.J_pairs) if S0.J_pairs is not None else None, (S1.Jt - S1.J_pairs) if S1.J_pairs is not None else None)):
        if e0 is None or e1 is None:
            out[part] = None
            continue
        em, jm = 0.5 * (e0 + e1), 0.5 * (j0 + j1)
        wN, wv = float(np.sum(dN * em)), float(np.sum(dv * jm))
        cum = np.cumsum(dN * em + dv * jm)
        out[part] = {"dN_rho": wN, "dv_J": wv, "work": wN + wv, "bracket_sources_at_start": float(np.sum(dN * e0 + dv * j0)), "bracket_sources_at_end": float(np.sum(dN * e1 + dv * j1)),
                     "work_inside_R_50_of_end": interp_safe(S1.R_50, S1.R, cum), "work_inside_outer_root_of_end": interp_safe(S1.Ro_d, S1.R, cum)}
    return out


def t3_history(slices, a):
    sl = sorted(slices, key=lambda s: s.T)
    rows, prev = [], None
    for S in sl:
        bins = (S.J.get("killing") or {}).get("bins_weighted") or []
        EK = float(sum(b_["E"] for b_ in bins))
        mb = _bin(bins, "m < omega <= b")
        neg = _bin(bins, "-b < omega <= -m").get("E", 0.0) + _bin(bins, "-m < omega <= 0").get("E", 0.0)
        r = {"T_over_rs": S.T, "slice": S.key, "label": S.short_label(), "E_K_dust": EK, "E_mb": mb.get("E"), "N_mb": mb.get("N"), "E_negative_bins": neg, "N_total": float(sum(b_["N"] for b_ in bins)),
             "mean_omega_over_m_mb": (mb.get("E") / (S.m * mb.get("N"))) if (mb.get("E") is not None and mb.get("N")) else None,  # v1b: the energy per shed particle in units of m
             "bins": bins}
        if prev is not None:
            r.update({"dT_rs": S.T - prev["T_over_rs"], "dE_K": EK - prev["E_K_dust"], "dE_mb": (mb.get("E") - prev["E_mb"]) if (mb.get("E") is not None and prev["E_mb"] is not None) else None,
                      "dN_mb": (mb.get("N") - prev["N_mb"]) if (mb.get("N") is not None and prev["N_mb"] is not None) else None})
        rows.append(r)
        prev = r
    ref = [r for r in rows if abs(r["T_over_rs"] - 20.0) < 1e-6]
    E_mb_20 = ref[0]["E_mb"] if ref else None
    for r in rows:
        r["dE_K_over_E_mb_20"] = (r["dE_K"] / E_mb_20) if (r.get("dE_K") is not None and E_mb_20) else None
    dk = [r["dE_K"] for r in rows if r.get("dE_K") is not None and r.get("dE_mb") is not None]
    dm = [r["dE_mb"] for r in rows if r.get("dE_K") is not None and r.get("dE_mb") is not None]
    corr = float(np.corrcoef(dk, dm)[0, 1]) if len(dk) >= 3 and np.std(dk) > 0 and np.std(dm) > 0 else None
    works = []
    for i in range(len(sl) - 1):
        S0, S1 = sl[i], sl[i + 1]
        if abs((S1.T - S0.T) - 1.0) < 1e-6:
            rec0, rec1 = (S0.J.get("record") or {}).get("M1_balance") or {}, (S1.J.get("record") or {}).get("M1_balance") or {}
            w_ = {"interval": "%g -> %g" % (S0.T, S1.T), "label": "%s / %s" % (S0.short_label(), S1.short_label()), "applied": field_work(S0, S1, "applied"), "dense": field_work(S0, S1, "dense"),
                  "record_field_work_per_step_at_ends": [rec0.get("field_work"), rec1.get("field_work")],
                  "record_field_work_x8_at_ends_derived": [(8.0 * rec0["field_work"]) if rec0.get("field_work") is not None else None, (8.0 * rec1["field_work"]) if rec1.get("field_work") is not None else None],
                  "dE_K": [r for r in rows if r["slice"] == S1.key][0].get("dE_K"), "dE_mb": [r for r in rows if r["slice"] == S1.key][0].get("dE_mb")}
            works.append(w_)
    return {"history": sl[0].history if sl else None, "E_mb_at_20": E_mb_20, "rows": rows, "pearson_dE_K_vs_dE_mb": corr, "work": works,
            "definitions": "E_K,dust = the sum of the six bins' E of killing.bins_weighted (the dust's Killing energy against the slice's own h_m at the applied fields); E_(m,b] the bin's E and E_(m,b]/(m N_(m,b]) its mean Killing energy per particle in units of m (the energy per shed particle, 1.12 to 1.22 m on the requirements' reading; v1b); the differences are between consecutive stored slices; the work is sum over the sites of (Delta N eps_tot + Delta v J_tot) with the mid-slice sources, the dust's part with eps_tot - eps_pairs and J_tot - J_pairs, the one-end brackets beside; the record's M1_balance.field_work is per 0.125 r_s step at the two ends (x8 per r_s, derived); the reading as amended (red team E4, E5; v1b): the Killing energy per particle is set as the particle leaves the core and conserved beyond the fountain's reach, inside which the fields breathe, so a rise of E_K with the (m, b] content is the fields' work on the orbitals, the generic case at the computed size, and the push's 45 at 15 is the bins' 40"}  # v1b


def t3_md(hk, out, label_text):
    L = ["### T3: %s" % label_text, "", out["definitions"] + ". E_(m,b] at 20: %s. Pearson correlation of Delta E_K with Delta E_(m,b] over the intervals (derived): %s." % (fx(out["E_mb_at_20"], "%.6f"), fx(out["pearson_dE_K_vs_dE_mb"], "%.3f")), "",
         "| T/r_s | E_K,dust = sum of the 6 bins' E | E_(m,b] | N_(m,b] | E_(m,b]/(m N_(m,b]) (per shed particle, in m; v1b) | E of the negative bins | Delta E_K from the previous stored slice | Delta E_(m,b] | Delta N_(m,b] | Delta T (r_s) | Delta E_K over E_(m,b](20) | label |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # v1b
    for r in out["rows"]:
        L.append("| %.3f | %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["T_over_rs"], r["E_K_dust"], fx(r["E_mb"], "%.6f"), fx(r["N_mb"], "%.6f"), fx(r.get("mean_omega_over_m_mb"), "%.4f"), fx(r["E_negative_bins"], "%+.3e"), fx(r.get("dE_K"), "%+.4f"), fx(r.get("dE_mb"), "%+.6f"), fx(r.get("dN_mb"), "%+.4f"),  # v1b
                                                                                     fx(r.get("dT_rs"), "%.3f"), fx(r.get("dE_K_over_E_mb_20"), "%+.3f"), r["label"]))
    L += ["", "The field work over the stored intervals 1 r_s apart:", "",
          "| interval | fields | sum Delta N eps (total sources) | sum Delta v J | work (total) | one-end brackets (sources at the start; at the end) | dust part: Delta N eps; Delta v J; work | work inside R_50; inside the outer root (end slice) | record field_work per step at the ends (x8, derived) | Delta E_K over the interval | Delta E_(m,b] | label |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for w_ in out["work"]:
        for fld in ("applied", "dense"):
            q = w_[fld]
            t_, d_ = q["total"], q["dust"] or {}
            L.append("| %s | %s | %+.4f | %+.4f | %+.4f | %+.4f; %+.4f | %s; %s; %s | %s; %s | %s, %s (%s, %s) | %s | %s | %s |" % (
                w_["interval"], fld, t_["dN_rho"], t_["dv_J"], t_["work"], t_["bracket_sources_at_start"], t_["bracket_sources_at_end"], fx(d_.get("dN_rho"), "%+.4f"), fx(d_.get("dv_J"), "%+.4f"), fx(d_.get("work"), "%+.4f"),
                fx(t_["work_inside_R_50_of_end"], "%+.4f"), fx(t_["work_inside_outer_root_of_end"], "%+.4f"), fx(w_["record_field_work_per_step_at_ends"][0], "%+.4f"), fx(w_["record_field_work_per_step_at_ends"][1], "%+.4f"),
                fx(w_["record_field_work_x8_at_ends_derived"][0], "%+.3f"), fx(w_["record_field_work_x8_at_ends_derived"][1], "%+.3f"), fx(w_["dE_K"], "%+.4f"), fx(w_["dE_mb"], "%+.6f"), w_["label"]))
    if not out["work"]:
        L.append("| none: no two stored slices of this history are 1 r_s apart | | | | | | | | | | | |")
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------- T4
def _fwhm(freqs, P, j):
    half = P[j] / 2.0
    lo = j
    while lo > 0 and P[lo] > half:
        lo -= 1
    hi = j
    while hi < P.size - 1 and P[hi] > half:
        hi += 1

    def cross(i0, i1):
        if P[i1] == P[i0]:
            return float(freqs[i1])
        return float(freqs[i0] + (half - P[i0]) * (freqs[i1] - freqs[i0]) / (P[i1] - P[i0]))

    fl = cross(lo, lo + 1) if lo < j else float(freqs[j])
    fr = cross(hi - 1, hi) if hi > j else float(freqs[j])
    return fl, fr


def dft_period(T, y, T0, T1, dt=0.125, pad=16, detrend=False):
    """the discrete Fourier transform of y(T) over [T0, T1] on the records' 0.125 r_s grid (missing records interpolated and counted); the mean
    removed (or a line, detrend); the dominant bin, the zero-padded refinement of the peak and its half-power width in period."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    sel = (T >= T0 - 1e-9) & (T <= T1 + 1e-9) & np.isfinite(y)
    if sel.sum() < 8:
        return {"available": False, "reason": "fewer than 8 finite samples in the window", "samples_in_window": int(sel.sum())}
    Ts, ys = T[sel], y[sel]
    n = int(round((T1 - T0) / dt)) + 1
    grid = T0 + dt * np.arange(n)
    on = np.array([bool(np.any(np.abs(Ts - g_) < 1e-6)) for g_ in grid])
    yg = np.interp(grid, Ts, ys)
    if detrend:
        X = np.vstack([grid - grid.mean(), np.ones(n)]).T
        coef = np.linalg.lstsq(X, yg, rcond=None)[0]
        y0 = yg - X @ coef
        trend = float(coef[0])
    else:
        y0 = yg - yg.mean()
        trend = None
    f = np.fft.rfftfreq(n, dt)
    P = np.abs(np.fft.rfft(y0)) ** 2
    P[0] = 0.0
    k = int(np.argmax(P))
    df = float(f[1] - f[0])
    period = (1.0 / f[k]) if f[k] > 0 else None
    top = np.argsort(P)[::-1][:3]
    npad = n * pad
    fp = np.fft.rfftfreq(npad, dt)
    Pp = np.abs(np.fft.rfft(y0, npad)) ** 2
    Pp[fp < 0.999 / (T1 - T0)] = 0.0
    j = int(np.argmax(Pp))
    fl, fr = _fwhm(fp, Pp, j)
    period_pad = (1.0 / fp[j]) if fp[j] > 0 else None
    hw = (0.5 * (1.0 / fl - 1.0 / fr)) if (fl > 0 and fr > 0) else None
    return {"available": True, "samples_in_window": int(sel.sum()), "grid_points": n, "grid_points_interpolated": int((~on).sum()), "dt_rs": dt, "detrended": detrend, "linear_trend_per_rs": trend,
            "mean": float(yg.mean()), "min": float(yg.min()), "max": float(yg.max()), "peak_to_peak": float(np.ptp(yg)), "bin_width_per_rs": df,
            "plain": {"f_peak": float(f[k]), "period_rs": period, "period_over_4pi_rs": (period / (4.0 * math.pi)) if period else None, "bin_half_width_period_rs": (0.5 * period * period * df) if period else None,
                      "variance_fraction_in_peak_bin": float(P[k] / P.sum()) if P.sum() > 0 else None,
                      "top_bins": [{"f": float(f[i]), "period_rs": (1.0 / f[i]) if f[i] > 0 else None, "power_fraction": float(P[i] / P.sum()) if P.sum() > 0 else None} for i in top]},
            "padded": {"pad": pad, "f_peak": float(fp[j]), "period_rs": period_pad, "period_over_4pi_rs": (period_pad / (4.0 * math.pi)) if period_pad else None, "half_width_period_rs": hw, "half_power_f": [fl, fr],
                       "omega_over_T_H": (8.0 * math.pi ** 2 / period_pad) if period_pad else None}}  # v1b: the angular frequency 2 pi/P (P in r_s) over T_H = 1/(4 pi r_s)


def rooted_stretches(T, mask):
    out, start, last = [], None, None
    for t, ok in zip(T, mask):
        if ok and start is None:
            start = t
        if not ok and start is not None:
            out.append([start, last])
            start = None
        if ok:
            last = t
    if start is not None:
        out.append([start, last])
    return out


def pg_ingoing_time(e, rs, r_turn, nq=4000):
    """int_{r_s}^{r_turn} dr (e^2 + beta^2)/((e + beta S) S), beta^2 = r_s/r, S = sqrt(e^2 - 1 + beta^2): the Painleve-Gullstrand time of the radial geodesic with Killing
    energy e m falling from rest at r_turn = r_s/(1 - e^2) to r = r_s (derived from ds^2 = -dt^2 + (dr + sqrt(r_s/r) dt)^2: dr/dt = -(1 - beta^2) S/(e - beta S), rewritten
    without the 0/0 at r_s); finite; the outgoing leg of the same geodesic has dr/dt -> 0 at r_s and no finite PG time from r_s."""
    if e is None or not (0.0 < e < 1.0) or rs is None or r_turn is None or r_turn <= rs:
        return None
    th = (np.arange(nq) + 0.5) / nq * (0.5 * math.pi)
    r = rs + (r_turn - rs) * np.sin(th) ** 2
    dr = 2.0 * (r_turn - rs) * np.sin(th) * np.cos(th) * (0.5 * math.pi / nq)
    beta2 = rs / r
    S = np.sqrt(np.maximum(e * e - 1.0 + beta2, 1e-300))
    return float(np.sum((e * e + beta2) / ((e + np.sqrt(beta2) * S) * S) * dr))


def literal_null_roundtrip(R_t, r_turn, rs, nq=4000):
    """2 int_{R_t}^{r_turn} dr/(1 - sqrt(r_s/r)) as T4 states it: a number where R_t > r_s; divergent where the root lies at or inside r_s (the pole at r = r_s)."""
    if R_t is None or r_turn is None or rs is None:
        return None, "a limit is absent"
    if r_turn <= R_t:
        return None, "r_turn %.3f a lies at or inside the root %.3f a" % (r_turn, R_t)
    if R_t <= rs:
        return None, "divergent: the root %.3f a lies at or inside r_s %.3f a, where 1/(1 - sqrt(r_s/r)) has a non-integrable pole; not a number" % (R_t, rs)
    r = R_t + (r_turn - R_t) * (np.arange(nq) + 0.5) / nq
    return float(2.0 * np.sum(1.0 / (1.0 - np.sqrt(rs / r))) * (r_turn - R_t) / nq), "finite"


def t6_values(S):
    R = S.R
    with np.errstate(divide="ignore", invalid="ignore"):
        f_d = 1.0 / (S.N_d * (1.0 + np.abs(S.w_d)))
        f_a = 1.0 / (S.N_a * (1.0 + np.abs(S.w_a)))
    return {"dense": integrate_between(R, f_d, S.Ri_d, S.Ro_d), "applied": integrate_between(R, f_a, S.Ri_a, S.Ro_a),
            "applied_fields_between_dense_roots": integrate_between(R, f_a, S.Ri_d, S.Ro_d), "dense_fields_between_applied_roots": integrate_between(R, f_d, S.Ri_a, S.Ro_a)}


def t4_history(hk, REC, slices, a):
    out = {"history": hk, "records_available": bool(REC and REC.get("available"))}
    if not out["records_available"]:
        out["reason"] = "records_%s.json absent or not available" % hk
        return out
    rows = sorted(REC["rows"], key=lambda r: float(r["T_over_rs"]))
    T = np.array([float(r["T_over_rs"]) for r in rows])
    Rt = np.array([float(r["R_t"]) if r.get("R_t") is not None else np.nan for r in rows])
    Rin = np.array([float(r["R_in"]) if r.get("R_in") is not None else np.nan for r in rows])
    E = np.array([float(r["E"]) if r.get("E") is not None else np.nan for r in rows])
    gate = np.array([float(r["gate_sigma_abs_dv_max"]) if r.get("gate_sigma_abs_dv_max") is not None else np.nan for r in rows])
    sel18 = T >= T4_RECORDS_FROM - 1e-9
    rooted = np.isfinite(Rt)
    out["rooted_records_from_18.375"] = int(np.sum(rooted & sel18))
    out["rooted_stretches_from_18.375"] = rooted_stretches(T[sel18], rooted[sel18])
    out["records_past_limit_in_window"] = int(np.sum((T >= T4_WINDOW[0] - 1e-9) & (T <= T4_WINDOW[1] + 1e-9) & (gate > 0.1)))
    out["records_in_window"] = int(np.sum((T >= T4_WINDOW[0] - 1e-9) & (T <= T4_WINDOW[1] + 1e-9)))
    win = (T >= T4_WINDOW[0] - 1e-9) & (T <= T4_WINDOW[1] + 1e-9)
    n_grid = int(round((T4_WINDOW[1] - T4_WINDOW[0]) / 0.125)) + 1
    series = {}
    for name, y in (("R_t", Rt), ("R_in", Rin), ("E", E)):
        covered = int(np.sum(win & np.isfinite(y)))
        d = {"samples_in_window": covered, "grid_points": n_grid, "coverage": covered / n_grid, "stretches_in_window": rooted_stretches(T[win], np.isfinite(y)[win])}
        if covered == n_grid:
            d["dft_mean_removed"] = dft_period(T, y, T4_WINDOW[0], T4_WINDOW[1], 0.125, a.pad, False)
            d["dft_detrended"] = dft_period(T, y, T4_WINDOW[0], T4_WINDOW[1], 0.125, a.pad, True)
        elif covered >= 0.98 * n_grid:
            d["dft_mean_removed"] = dft_period(T, y, T4_WINDOW[0], T4_WINDOW[1], 0.125, a.pad, False)
            d["dft_detrended"] = dft_period(T, y, T4_WINDOW[0], T4_WINDOW[1], 0.125, a.pad, True)
            d["note"] = "%d of %d grid points interpolated" % (n_grid - covered, n_grid)
        else:
            d["dft"] = "not computed: the series over 20 to 40 has gaps (defined on %s); a transform of a gapped series is not the test as stated" % d["stretches_in_window"]
        series[name] = d
    out["series"] = series
    out["resolution"] = ("the window 20 to 40 r_s holds 20/P cycles of a period P: 2.5 for 8 r_s; the plain bins are 20.125/k r_s (20.1, 10.1, 6.7, 5.0, ...), and the padded peak locates P within its half-power half-width, "
                         "about 0.44 P^2/20 r_s (1.4 to 1.6 r_s at P = 8), its offset from the true period being the short record's leakage and line removal (the self-test's 0.26 r_s at 2.5 cycles); no finer period is read (v1b)")  # v1b
    out["R_50_slices_20_to_40"] = [{"T_over_rs": S.T, "R_50": S.R_50, "label": S.short_label()} for S in sorted(slices, key=lambda s: s.T) if 20.0 - 1e-6 <= S.T <= 40.0 + 1e-6]
    ff = []
    for S in sorted(slices, key=lambda s: s.T):
        if not S.rooted:
            continue
        fo = S.J.get("fountain") or {}
        e = fo.get("mean_omega_over_m_bound")
        t6 = t6_values(S)
        r = {"T_over_rs": S.T, "in_window": bool(20.0 - 1e-6 <= S.T <= 40.0 + 1e-6), "label": S.short_label(), "omega_over_m_bound": e, "R_t_dense": S.Ro_d, "R_in_dense": S.Ri_d,
             "t6_dense_rs": (t6["dense"] / S.rs) if t6["dense"] is not None else None}
        for tag, rs_, rt in (("nominal", S.rs, fo.get("r_turn_rs_nominal")), ("2gE", S.two_gE, fo.get("r_turn_2gE"))):
            val, status = literal_null_roundtrip(S.Ro_d, rt, rs_)
            Tin = pg_ingoing_time(e, rs_, rt)
            q = {"r_s": rs_, "r_turn": rt, "r_turn_recomputed": (rs_ / (1.0 - e * e)) if (e is not None and rs_ is not None and e < 1) else None, "literal_2int_dr_over_1_minus_sqrt": val, "literal_status": status,
                 "pg_ingoing_time_r_turn_to_rs_a": Tin, "pg_ingoing_time_rs": (Tin / S.rs) if Tin is not None else None,
                 "round_trip_estimate_rs": ((2.0 * Tin + 2.0 * t6["dense"]) / S.rs) if (Tin is not None and t6["dense"] is not None) else None}
            r[tag] = q
        ff.append(r)
    out["free_fall"] = ff
    out["free_fall_definitions"] = ("the literal integral 2 int_{R_t}^{r_turn} dr/(1 - sqrt(r_s/r)) is printed where finite and reported divergent where the dense outer root lies at or inside r_s; beside it (derived) the "
                                    "Painleve-Gullstrand geodesic fall from rest at r_turn to r_s for the bound part's mean e = omega/m, and the round-trip estimate 2 x that plus 2 x T6's kinematic lower bound of the "
                                    "root-to-root crossing on the dense fields (a lower bound on the core legs): the sum is a lower bound on the round trip, not the round trip; the turning radius from the mean bound "
                                    "omega/m is not a bound on the top columns' reach after 20 (CORE_well 9.1; red team E4), so the round trip is the mean energy's (v1b)")  # v1b
    return out


def t4_md(hk, out, label_text):
    L = ["### T4: %s" % label_text, ""]
    if not out.get("records_available"):
        L += ["Not available: %s." % out.get("reason"), ""]
        return L
    L += ["Records from 18.375 with a dense root: %d, on the stretches %s (T/r_s). Records in the window 20 to 40: %d, of which %d past the record's own limit (gate above 0.1): every transform below carries the label past the record's own limit on those records. Grid 0.125 r_s, %d points; the plain transform's bin width %.4f per r_s (periods 20/k r_s). Resolution: %s." % (
        out["rooted_records_from_18.375"], out["rooted_stretches_from_18.375"], out["records_in_window"], out["records_past_limit_in_window"], out["series"]["E"]["grid_points"], 1.0 / 20.0, out["resolution"]), "",  # v1b
          "| series | samples in 20 to 40 (of the grid) | stretches | mean; min; max | plain DFT peak: period (r_s); over 4 pi r_s; bin half-width (r_s); variance fraction | top 3 plain bins (period r_s: fraction) | padded refinement: period (r_s); half-power half-width (r_s); 2 pi/P over T_H = 1/(4 pi r_s) (v1b) | the same with a line removed: period; half-width; trend per r_s | label |", "|---|---|---|---|---|---|---|---|---|"]  # v1b
    for name in ("R_t", "R_in", "E"):
        d = out["series"][name]
        if "dft_mean_removed" not in d:
            L.append("| %s | %d (%d) | %s | - | %s | - | - | - | past limit (records) |" % (name, d["samples_in_window"], d["grid_points"], d["stretches_in_window"], d.get("dft", "-")))
            continue
        m_, t_ = d["dft_mean_removed"], d["dft_detrended"]
        L.append("| %s | %d (%d)%s | %s | %.4f; %.4f; %.4f | %s; %s; %s; %s | %s | %s; %s; %s | %s; %s; %s | past limit (records) |" % (  # v1b
            name, d["samples_in_window"], d["grid_points"], (" " + d["note"]) if d.get("note") else "", d["stretches_in_window"], m_["mean"], m_["min"], m_["max"],
            fx(m_["plain"]["period_rs"], "%.3f"), fx(m_["plain"]["period_over_4pi_rs"], "%.4f"), fx(m_["plain"]["bin_half_width_period_rs"], "%.3f"), fx(m_["plain"]["variance_fraction_in_peak_bin"], "%.3f"),
            ", ".join("%s: %s" % (fx(b_["period_rs"], "%.3f"), fx(b_["power_fraction"], "%.3f")) for b_ in m_["plain"]["top_bins"]),
            fx(m_["padded"]["period_rs"], "%.3f"), fx(m_["padded"]["half_width_period_rs"], "%.3f"), fx(m_["padded"]["omega_over_T_H"], "%.2f"), fx(t_["padded"]["period_rs"], "%.3f"), fx(t_["padded"]["half_width_period_rs"], "%.3f"), fx(t_["linear_trend_per_rs"], "%+.4f")))  # v1b
    L += ["", "R_50 of the stored slices 20 to 40 (a; the json's dust.R_50): %s." % ", ".join("%g: %s (%s)" % (q["T_over_rs"], fx(q["R_50"], "%.3f"), q["label"]) for q in out["R_50_slices_20_to_40"]), "", out["free_fall_definitions"] + ".", "",
          "| T/r_s | in the window | omega/m bound | R_in, R_t dense (a) | r_s nominal: r_turn (a); literal 2 int dr/(1 - sqrt(r_s/r)); PG fall r_turn to r_s (r_s); round-trip estimate (r_s) | r_s = 2gE: r_turn (a); literal; PG fall (r_s); round-trip estimate (r_s) | T6 dense (r_s) | label |", "|---|---|---|---|---|---|---|---|"]
    for r in out["free_fall"]:
        nom, tg = r["nominal"], r["2gE"]
        L.append("| %.3f | %s | %s | %s, %s | %s; %s; %s; %s | %s; %s; %s; %s | %s | %s |" % (
            r["T_over_rs"], "yes" if r["in_window"] else "no", fx(r["omega_over_m_bound"], "%.5f"), fx(r["R_in_dense"], "%.3f"), fx(r["R_t_dense"], "%.3f"),
            fx(nom["r_turn"], "%.3f"), (fx(nom["literal_2int_dr_over_1_minus_sqrt"], "%.3f a") if nom["literal_2int_dr_over_1_minus_sqrt"] is not None else nom["literal_status"]), fx(nom["pg_ingoing_time_rs"], "%.4f"), fx(nom["round_trip_estimate_rs"], "%.4f"),
            fx(tg["r_turn"], "%.3f"), (fx(tg["literal_2int_dr_over_1_minus_sqrt"], "%.3f a") if tg["literal_2int_dr_over_1_minus_sqrt"] is not None else tg["literal_status"]), fx(tg["pg_ingoing_time_rs"], "%.4f"), fx(tg["round_trip_estimate_rs"], "%.4f"),
            fx(r["t6_dense_rs"], "%.4f"), r["label"]))
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------- T5
def t5_history(slices, a):
    rows = []
    for S in sorted(slices, key=lambda s: s.T):
        if not S.rooted:
            continue
        cu = (S.J.get("currents") or {})
        w_ = (cu.get("dense_outer") or {}).get("weighted") or {}
        wa = (cu.get("applied_outer") or {}).get("weighted") or {}

        def frac(num, den):
            return (abs(num) / abs(den)) if (num is not None and den not in (None, 0.0)) else None

        r = {"T_over_rs": S.T, "slice": S.key, "label": S.short_label(), "R_t_dense": S.Ro_d, "N_at_cut": (cu.get("dense_outer") or {}).get("N_applied_at_cut"),
             "matter_FM": w_.get("dev_FM"), "dust_FM": w_.get("dust_FM"), "pairs_FM": w_.get("pairs_FM"),
             "signs": {k_: (("+" if w_[k_] > 0 else "-") if w_.get(k_) is not None else "-") for k_ in ("dev_FM", "dust_FM", "pairs_FM")},
             "pairs_share_of_matter_FM": (w_["pairs_FM"] / w_["dev_FM"]) if (w_.get("pairs_FM") is not None and w_.get("dev_FM")) else None,
             "dust_FM_low": w_.get("dust_FM_low"), "dust_FM_high": w_.get("dust_FM_high"), "dust_FM_cross": w_.get("dust_FM_cross"),
             "fraction_high_over_dust": frac(w_.get("dust_FM_high"), w_.get("dust_FM")),
             "fraction_high_plus_cross_over_dust": frac((w_.get("dust_FM_high") or 0.0) + (w_.get("dust_FM_cross") or 0.0), w_.get("dust_FM")) if w_.get("dust_FM") is not None else None,
             "applied_root": {"R": S.Ro_a, "matter_FM": wa.get("dev_FM"), "dust_FM": wa.get("dust_FM"), "pairs_FM": wa.get("pairs_FM"), "fraction_high_over_dust": frac(wa.get("dust_FM_high"), wa.get("dust_FM"))} if wa else None,
             "matter_FM_dense_minus_applied_root": (w_["dev_FM"] - wa["dev_FM"]) if (w_.get("dev_FM") is not None and wa.get("dev_FM") is not None) else None,
             "T_ll_stress_split": "not available (p_r per species is held by no stored array; the stress module's companion rows are not among the stored outputs)",
             "dust_number_current": w_.get("dust_number"), "pairs_number_current": w_.get("pairs_number")}
        rows.append(r)
    return {"rows": rows, "definitions": "F_M is the Killing-energy current over N_bar at the cut, outward positive (the flux of rho + w J; CORE_well 10.3): matter = dev = C - R2 R2^dagger, dust, pairs = matter - dust (the reference-subtracted sea); the momentum split low (|ap| <= pi/4), high, cross is not additive at the cut and only the sum is certified (CORE_well correction 3); T_ll = rho + p_r - 2J needs the stress, which no stored array holds, and F_M, an energy flux, is not T_ll and its sign is not T_ll's sign (v1b); as amended (red team E6, E7; v1b): T_ll of a positive band-diagonal filling is 8 pi N r^2 integral f (eps - q_r)^2/eps >= 0 even on the outward branch, so the outgoing dust does not recede the root within the rows, and FIRSTLAW's negative T_ll at 163 of 225 dense roots is the reference-subtracted sources' on the sigma/a 2 history under 4 stress conditions the records do not close, a different object from the theorem's and not its contradiction; the push's reading of (5.2) prints nowhere as a discrimination"}  # v1b


def t5_md(hk, out, label_text):
    L = ["### T5: %s" % label_text, "", out["definitions"] + ".", "",
         "| T/r_s | R_t dense (a) | N at the cut | matter F_M (dev) | dust F_M | pairs F_M (the sea) | signs (matter; dust; pairs) | pairs share of the matter F_M | dust F_M low; high; cross | |high| over |dust F_M| | |high + cross| over |dust F_M| | at the applied outer root: R (a); matter; dust; pairs; |high|/|dust| | matter F_M dense root - applied root | T_ll stress split (total; dust; sea) | label |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        ap = r["applied_root"]
        L.append("| %.3f | %s | %s | %s | %s | %s | %s; %s; %s | %s | %s; %s; %s | %s | %s | %s | %s | %s | %s |" % (
            r["T_over_rs"], fx(r["R_t_dense"], "%.4f"), fx(r["N_at_cut"], "%.5f"), fx(r["matter_FM"], "%+.6e"), fx(r["dust_FM"], "%+.6e"), fx(r["pairs_FM"], "%+.6e"), r["signs"]["dev_FM"], r["signs"]["dust_FM"], r["signs"]["pairs_FM"],
            fx(r["pairs_share_of_matter_FM"], "%.3f"), fx(r["dust_FM_low"], "%+.4e"), fx(r["dust_FM_high"], "%+.4e"), fx(r["dust_FM_cross"], "%+.4e"), fx(r["fraction_high_over_dust"], "%.3f"), fx(r["fraction_high_plus_cross_over_dust"], "%.3f"),
            ("%s; %s; %s; %s; %s" % (fx(ap["R"], "%.4f"), fx(ap["matter_FM"], "%+.4e"), fx(ap["dust_FM"], "%+.4e"), fx(ap["pairs_FM"], "%+.4e"), fx(ap["fraction_high_over_dust"], "%.3f"))) if ap else "no applied outer root",
            fx(r["matter_FM_dense_minus_applied_root"], "%+.4e"), r["T_ll_stress_split"], r["label"]))
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------- T6
def t6_history(slices, a):
    sl = sorted(slices, key=lambda s: s.T)
    by_T = {round(S.T, 6): S for S in sl}
    S20, S40 = by_T.get(20.0), by_T.get(40.0)
    front = None
    if S20 is not None and S40 is not None:
        r20, r40 = (S20.J.get("fountain") or {}).get("R_90_all"), (S40.J.get("fountain") or {}).get("R_90_all")
        if r20 is not None and r40 is not None:
            front = {"a_per_rs": (r40 - r20) / 20.0, "from": "R_90 of all the dust at 20 (%.3f a) and at 40 (%.3f a), derived" % (r20, r40)}
    rows = []
    for S in sl:
        if not S.rooted:
            continue
        v = t6_values(S)
        r = {"T_over_rs": S.T, "slice": S.key, "label": S.short_label(), "R_in_dense": S.Ri_d, "R_t_dense": S.Ro_d, "R_in_applied": S.Ri_a, "R_t_applied": S.Ro_a,
             "integral_dense_a": v["dense"], "integral_dense_rs": (v["dense"] / S.rs) if v["dense"] is not None else None,
             "integral_applied_a": v["applied"], "integral_applied_rs": (v["applied"] / S.rs) if v["applied"] is not None else None,
             "integral_applied_fields_between_dense_roots_a": v["applied_fields_between_dense_roots"], "integral_dense_fields_between_applied_roots_a": v["dense_fields_between_applied_roots"],
             "dense_minus_applied_a": (v["dense"] - v["applied"]) if (v["dense"] is not None and v["applied"] is not None) else None,
             "dense_minus_applied_fields_same_roots_a": (v["dense"] - v["applied_fields_between_dense_roots"]) if (v["dense"] is not None and v["applied_fields_between_dense_roots"] is not None) else None}
        if front and S.Ri_d is not None:
            r["ejecta_crossing_rs"] = (S.Ro_d - S.Ri_d) / front["a_per_rs"]
            r["bound_over_ejecta_crossing"] = (r["integral_dense_rs"] / r["ejecta_crossing_rs"]) if r["integral_dense_rs"] is not None else None
        rows.append(r)
    return {"front": front, "rows": rows, "definitions": "int_{R_in}^{R_t} dr/[N(r)(1 + |w(r)|)] by the trapezoid on the sites between the roots (the endpoints interpolated), the dense fields between the record's trapping zeros and the applied fields between the applied roots, each between the other's roots beside; the ejecta's crossing time of R_t - R_in at the unbound front's speed; no fit"}


def t6_md(hk, out, label_text):
    fr = out["front"] or {}
    L = ["### T6: %s" % label_text, "", out["definitions"] + ". The front speed: %s a per r_s (%s)." % (fx(fr.get("a_per_rs"), "%.3f"), fr.get("from", "not derivable: a slice at 20 or 40 is missing")), "",
         "| T/r_s | R_in, R_t dense (a) | integral dense (a; r_s) | R_in, R_t applied (a) | integral applied (a; r_s) | applied fields between the dense roots (a) | dense fields between the applied roots (a) | dense - applied (own roots; same roots) (a) | ejecta crossing time of R_t - R_in (r_s) | bound over the ejecta crossing | label |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        L.append("| %.3f | %s, %s | %s; %s | %s, %s | %s; %s | %s | %s | %s; %s | %s | %s | %s |" % (
            r["T_over_rs"], fx(r["R_in_dense"], "%.3f"), fx(r["R_t_dense"], "%.3f"), fx(r["integral_dense_a"], "%.3f"), fx(r["integral_dense_rs"], "%.4f"), fx(r["R_in_applied"], "%.3f"), fx(r["R_t_applied"], "%.3f"),
            fx(r["integral_applied_a"], "%.3f"), fx(r["integral_applied_rs"], "%.4f"), fx(r["integral_applied_fields_between_dense_roots_a"], "%.3f"), fx(r["integral_dense_fields_between_applied_roots_a"], "%.3f"),
            fx(r["dense_minus_applied_a"], "%+.3f"), fx(r["dense_minus_applied_fields_same_roots_a"], "%+.3f"), fx(r.get("ejecta_crossing_rs"), "%.4f"), fx(r.get("bound_over_ejecta_crossing"), "%.3f"), r["label"]))
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------- T7
def t7_history(slices, a, rho_cache):
    rows = []
    named = T7_NAMED.get(slices[0].history, ()) if slices else ()
    for S in sorted(slices, key=lambda s: s.T):
        R = S.R
        idx = [int(np.argmin(np.abs(R - r_))) for r_ in range(1, 9)]
        if any(abs(R[i] - (k_ + 1)) > 1e-9 for k_, i in enumerate(idx)):
            rows.append({"T_over_rs": S.T, "slice": S.key, "status": "the sites 1 to 8 a are not on the grid"})
            continue
        Rs = R[idx]
        x = Rs ** 2
        Nd, Na, nn, wd = S.N_d[idx], S.N_a[idx], S.n_dens[idx], S.w_d[idx]
        yd, ya = np.log(Nd) - math.log(Nd[0]), np.log(Na) - math.log(Na[0])
        sd, cd, r2d = fit_lin(x, yd)
        sa, ca, r2a = fit_lin(x, ya)
        so_d, r2o_d = fit_origin(x - 1.0, yd)
        yn = np.log(nn) - math.log(nn[0]) if np.all(nn > 0) else None
        sn, cn, r2n = fit_lin(x, yn) if yn is not None else (None, None, None)
        H_fit, r2H = fit_origin(Rs, -wd)
        rho0_site = S.rho_site[idx[0]]
        rho0_kern = rho_cache.get(S.key)
        rho0_kern = float(rho0_kern[idx[0]]) if rho0_kern is not None else None
        r = {"T_over_rs": S.T, "slice": S.key, "label": S.short_label(), "named_in_T7": bool(any(abs(S.T - t_) < 1e-6 for t_ in named)), "rooted": S.rooted,
             "N_dense_1_to_8": [float(v) for v in Nd], "N_applied_1_to_8": [float(v) for v in Na], "n_density_1_to_8": [float(v) for v in nn], "w_dense_1_to_8": [float(v) for v in wd],
             "fit_lnN_dense": {"slope": sd, "intercept": cd, "r2": r2d}, "fit_lnN_dense_through_origin_in_R2_minus_1": {"slope": so_d, "r2_uncentred": r2o_d}, "fit_lnN_applied": {"slope": sa, "intercept": ca, "r2": r2a},
             "slope_dense_minus_applied": (sd - sa) if (sd is not None and sa is not None) else None, "fit_ln_n": {"slope": sn, "intercept": cn, "r2": r2n},
             "H_fit_minus_w_over_R": H_fit, "H_fit_r2_uncentred": r2H, "H_from_8pi_g_rho_over_3": {"site_rho_1a": math.sqrt(8.0 * math.pi * S.g * rho0_site / 3.0) if rho0_site > 0 else None,
                                                                                                        "kernel_rho_ell_1a": (math.sqrt(8.0 * math.pi * S.g * rho0_kern / 3.0) if (rho0_kern is not None and rho0_kern > 0) else None)},
             "rho_1a_site": float(rho0_site), "rho_ell_1a_kernel": rho0_kern, "solver_N_centre_record": ((S.J.get("record") or {}).get("solver") or {}).get("N_centre"), "node_reading": node_diagnostic(R, S.N_d)["reading"]}
        rows.append(r)
    return {"rows": rows, "named_set": list(named), "definitions": "y = ln N(R) - ln N(1 a) against x = R^2 over the sites R = 1 to 8 a: the least-squares line y = s x + c (slope, intercept, r2; the dense fields, the applied beside), the line through the origin in x - 1 beside; the same for ln n; H_fit the slope of -w against R through the origin over 1 to 8 a against sqrt(8 pi g rho_0/3) of (1.2) with rho_0 the site rho at 1 a and the kernel rho_ell at 1 a (derived); the record's solver N_centre beside"}


def t7_md(hk, out, label_text):
    L = ["### T7: %s" % label_text, "", out["definitions"] + ". The slices T7 names on this history: %s (the other stored slices beside, flagged)." % out["named_set"], "",
         "| T/r_s | named in T7 | N dense at 1 a; at 8 a | slope of ln N - ln N(1a) against R^2 (dense; intercept; r2) | through the origin in R^2 - 1 (slope; uncentred r2) | applied (slope; r2) | dense - applied slope | n at 1 a; at 8 a (a^-3) | slope of ln n - ln n(1a) against R^2 (r2) | H_fit = -w/R over 1 to 8 a (r2) | H from 8 pi g rho_0/3: site rho(1a); kernel rho_ell(1a) | solver N_centre (record) | the dense N near the centre | label |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        if "status" in r:
            L.append("| %.3f | - | %s | | | | | | | | | | | |" % (r["T_over_rs"], r["status"]))
            continue
        L.append("| %.3f | %s | %.5f; %.5f | %s; %s; %s | %s; %s | %s; %s | %s | %.5f; %.5f | %s (%s) | %s (%s) | %s; %s | %s | %s | %s |" % (
            r["T_over_rs"], "yes" if r["named_in_T7"] else "no", r["N_dense_1_to_8"][0], r["N_dense_1_to_8"][7], fx(r["fit_lnN_dense"]["slope"], "%+.5e"), fx(r["fit_lnN_dense"]["intercept"], "%+.4e"), fx(r["fit_lnN_dense"]["r2"], "%.4f"),
            fx(r["fit_lnN_dense_through_origin_in_R2_minus_1"]["slope"], "%+.5e"), fx(r["fit_lnN_dense_through_origin_in_R2_minus_1"]["r2_uncentred"], "%.4f"), fx(r["fit_lnN_applied"]["slope"], "%+.5e"), fx(r["fit_lnN_applied"]["r2"], "%.4f"),
            fx(r["slope_dense_minus_applied"], "%+.3e"), r["n_density_1_to_8"][0], r["n_density_1_to_8"][7], fx(r["fit_ln_n"]["slope"], "%+.5e"), fx(r["fit_ln_n"]["r2"], "%.4f"), fx(r["H_fit_minus_w_over_R"], "%.5f"), fx(r["H_fit_r2_uncentred"], "%.4f"),
            fx(r["H_from_8pi_g_rho_over_3"]["site_rho_1a"], "%.5f"), fx(r["H_from_8pi_g_rho_over_3"]["kernel_rho_ell_1a"], "%.5f"), fx(r["solver_N_centre_record"], "%.5f"), r["node_reading"], r["label"]))
    L.append("")
    return L


# ------------------------------------------------------------------------------------------------------------------- T8
def t8_all(slices, a):
    rows = []
    for S in sorted(slices, key=lambda s: (s.history, s.T)):
        if not S.rooted:
            continue
        for fields, M, Ro in (("dense", S.M_w_d, S.Ro_d), ("applied", S.M_w_a, S.Ro_a)):
            r = {"slice": S.key, "history": S.history, "T_over_rs": S.T, "fields": fields, "label": S.short_label(), "E": S.E}
            if Ro is None:
                r["status"] = "no applied root"
                rows.append(r)
                continue
            sel = S.R <= Ro
            Mi = np.concatenate([M[sel], [interp_safe(Ro, S.R, M)]])
            Ri = np.concatenate([S.R[sel], [Ro]])
            steps = np.diff(Mi)
            M_root = float(Mi[-1])
            max_dec = float(-min(float(steps.min()), 0.0)) if steps.size else 0.0
            imax = int(np.argmax(Mi))
            r.update({"R_root": float(Ro), "M_w_at_root": M_root, "M_w_at_root_over_E": (M_root / S.E) if S.E else None, "largest_single_step_decrease_inside": max_dec, "largest_decrease_rel": (max_dec / M_root) if M_root else None,
                      "monotone_within_tol": bool(max_dec <= a.t8_tol * M_root), "strictly_non_decreasing": bool(max_dec == 0.0), "max_inside_root": float(Mi[imax]), "R_of_max_inside": float(Ri[imax]),
                      "peaks_at_root_within_tol": bool((float(Mi[imax]) - M_root) <= a.t8_tol * M_root), "below_E_at_root": bool(S.E is not None and M_root < S.E),
                      "max_over_chain": float(np.nanmax(M)), "R_of_max_over_chain": float(S.R[int(np.nanargmax(M))]), "max_over_chain_over_E": (float(np.nanmax(M)) / S.E) if S.E else None,
                      "M_w_at_R_c": interp_safe(S.R_c, S.R, M), "M_w_at_R_c_over_E": (interp_safe(S.R_c, S.R, M) / S.E) if (S.E and S.R_c is not None) else None})
            r["passes"] = bool(r["monotone_within_tol"] and r["peaks_at_root_within_tol"] and r["below_E_at_root"])
            rows.append(r)
    dense = [r for r in rows if r["fields"] == "dense"]
    n_rooted = len(dense)
    all_pass = bool(dense) and all(r["passes"] for r in dense)
    counter = [r for r in dense if not r["passes"]]
    if not dense:  # v1b
        line = "not printed: no rooted slice among the slices read (nothing to decide)"  # v1b
    elif all_pass and n_rooted == 13:  # v1b
        line = T8_LINE
    elif all_pass:
        line = "the mass function peaks at the root below E on all %d rooted slices read (not the 13 of the line: %d read): the verbatim line is not printed" % (n_rooted, n_rooted)
    else:
        line = "not printed: the counter-example(s) on the dense fields: " + "; ".join("%s (%s; monotone within tol %s, largest decrease %.3e; peaks at the root %s, max inside at R %.2f a; below E %s: M_w(root) %.4f against E %s)" % (
            r["slice"], r["label"], r["monotone_within_tol"], r["largest_single_step_decrease_inside"], r["peaks_at_root_within_tol"], r["R_of_max_inside"], r["below_E_at_root"], r["M_w_at_root"], fx(r["E"], "%.4f")) for r in counter)
    return {"rows": rows, "rooted_slices_dense": n_rooted, "all_pass_dense": all_pass, "line": line, "tol": a.t8_tol,
            "definitions": "on each rooted slice: M_w non-decreasing from the first site to the outer root (the largest single-step decrease against the tolerance %g x M_w(root)), its maximum inside the root at the root (within the same tolerance), and M_w(root) below the record's E; the dense fields decide (the record's roots), the applied fields are the systematic; the verbatim line prints only if every rooted slice read passes and the count read is 13; as amended (v1b): %s" % (a.t8_tol, AMENDED["T8"])}  # v1b


def t8_md(out):
    L = ["### T8: the halo closure (1.8) on every rooted slice", "", out["definitions"] + ".", "",
         "| slice | fields | R root (a) | M_w at the root | E (record) | M_w(root)/E | strictly non-decreasing inside | largest single-step decrease (relative) | monotone within tol | maximum inside the root at R (a) | peaks at the root | below E | M_w at R_c (over E) | max over the chain at R (a) (over E) | passes | label |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        if "status" in r:
            L.append("| %s | %s | %s | | | | | | | | | | | | | %s |" % (r["slice"], r["fields"], r["status"], r["label"]))
            continue
        L.append("| %s | %s | %.4f | %.4f | %s | %s | %s | %.3e (%s) | %s | %.4f at %.2f | %s | %s | %s (%s) | %.4f at %.1f (%s) | %s | %s |" % (
            r["slice"], r["fields"], r["R_root"], r["M_w_at_root"], fx(r["E"], "%.6f"), fx(r["M_w_at_root_over_E"], "%.4f"), r["strictly_non_decreasing"], r["largest_single_step_decrease_inside"], fx(r["largest_decrease_rel"], "%.1e"), r["monotone_within_tol"],
            r["max_inside_root"], r["R_of_max_inside"], r["peaks_at_root_within_tol"], r["below_E_at_root"], fx(r["M_w_at_R_c"], "%.4f"), fx(r["M_w_at_R_c_over_E"], "%.4f"), r["max_over_chain"], r["R_of_max_over_chain"], fx(r["max_over_chain_over_E"], "%.4f"), r["passes"], r["label"]))
    L += ["", "Rooted slices read (dense roots): %d; all pass on the dense fields: %s." % (out["rooted_slices_dense"], out["all_pass_dense"]), "", "The line: %s" % out["line"], ""]
    return L


# ------------------------------------------------------------------------------------------------------------------ the run
def load_history(d, hk, clocks, log):
    out = []
    for t in clocks:
        pj, pz, _ = slice_paths(d, hk, t)
        if os.path.isfile(pj) and os.path.isfile(pz):
            S = Slice(d, hk, t)
            log("read %s: T/r_s %.3f, %d sites, %s, %s, rooted %s" % (os.path.basename(pj), S.T, S.R.size, S.J.get("dtype"), S.short_label(), S.rooted))
            out.append(S)
        else:
            log("MISSING %s or its npz: skipped" % pj)
    return out


def run_tests(a, log, well=None, rung=None, out_dir=None, histories=None, clocks=None):
    well = well or a.well
    rung = rung or a.rung
    out_dir = out_dir or a.out
    histories = histories or [h.strip() for h in a.histories.split(",") if h.strip()]
    os.makedirs(out_dir, exist_ok=True)
    tests = [t.strip().upper() for t in a.tests.split(",")] if a.tests.strip().lower() != "all" else ["T%d" % i for i in range(1, 9)]
    wait_for_load(a.max_load, log)
    SL = {}
    REC = {}
    for hk in histories:
        cl = CLOCKS.get(hk, ())
        if clocks is not None:
            cl = clocks.get(hk, cl)
        elif a.clocks.strip().lower() != "all":
            cl = [float(x) for x in a.clocks.split(",") if x.strip()]
        SL[hk] = load_history(well, hk, cl, log)
        REC[hk] = read_records(well, hk)
        log("records_%s.json: %s" % (hk, ("%d rows" % len(REC[hk].get("rows", []))) if REC[hk] else "absent"))
    OUT = {"code": CODE, "title": TITLE, "tests_verbatim": TESTS_VERBATIM, "amended": AMENDED, "inputs": {"well": well, "rung": rung, "histories": histories, "slices_read": {hk: [S.T for S in SL[hk]] for hk in SL}, "records": {hk: bool(REC[hk]) for hk in REC}},  # v1b
           "kernel": {"name": a.kernel, "definition": KERNELS[a.kernel][1]}, "label_rule": "every value read from a stored slice carries the slice's limit label as computed by core_well_v1 (the gate sigma |d_r v| against 0.1); the complex64 floor where the json says complex64; the dense-against-applied difference is the systematic"}
    arrays = {}
    rho_cache = {}
    MD = ["# %s: the asymptotic push's cheap stored-data tests T1 to T8 on the stored slices (%s)" % (TITLE, CODE), "",
          "Inputs: the per-slice outputs of core_well_v1.py under %s (json and npz; the md is not parsed), records_<history>.json there, and for T2 the rung stand-ins under %s read under their own label. "
          "Histories: %s; slices read: %s. Every number below is read or derived from those files by this script; nothing is asserted. %s. T1 and T2 print one table per slice; T3 to T8 are sequence tests and print one table per history with one row per slice. "
          "The kernel of T1's rho_ell route: %s (%s). Each test's verbatim statement is followed by the red team's amendment of its comparator, label or reading as the lead applied it to requirements_new.txt (v1b)." % (well, rung, histories, OUT["inputs"]["slices_read"], OUT["label_rule"], a.kernel, KERNELS[a.kernel][1]), "", TESTS_VERBATIM["common"], ""]  # v1b
    labels = {hk: (SL[hk][0].label_text if SL[hk] else hk) for hk in SL}
    if "T1" in tests:
        MD += ["## T1", "", TESTS_VERBATIM["T1"], "", AMENDED["T1"], ""]  # v1b
        OUT["T1"] = {}
        for hk in histories:
            for S in SL[hk]:
                t_ = time.time()
                o = t1_slice(S, a, arrays, rho_cache)
                OUT["T1"][S.key] = o
                MD += t1_md(S, o)
                log("T1 %s: max |residual| relative inside %.1f a: dense %s, applied %s; normalization defect %s (fine grid); halo mean lambda %s; %.1f s" % (S.key, o["inside_radius_for_maxima"], fx(o["max_abs_residual_rel_inside"]["dense"], "%.2e"), fx(o["max_abs_residual_rel_inside"]["applied"], "%.2e"), fx(o["normalization"]["defect_fine_rel"], "%.1e"), fx((o["halo"] or {}).get("mean_lambda_dense"), "%.3f"), time.time() - t_))  # v1b
    if "T2" in tests:
        MD += ["## T2", "", TESTS_VERBATIM["T2"], "", AMENDED["T2"], ""]  # v1b
        OUT["T2"] = {}
        for hk, T, where in T2_TARGETS:
            d = well if where == "well" else rung
            S = None
            if where == "well":
                S = next((s for s in SL.get(hk, []) if abs(s.T - T) < 1e-6), None)
            if S is None:
                pj, pz, _ = slice_paths(d, hk, T)
                if os.path.isfile(pj) and os.path.isfile(pz):
                    S = Slice(d, hk, T)
                    log("read %s for T2 (%s)" % (os.path.basename(pj), S.label_text))
            if S is None:
                OUT["T2"]["%s_T%07.2f" % (hk, T)] = {"status": "not stored at read time: %s" % slice_paths(d, hk, T)[0]}
                MD += ["### T2: %s at T/r_s %g" % (hk, T), "", "Not stored at read time (%s): nothing printed (the rung's unstored clocks run only after the bridge, CORE_well v1.2 section 18)." % slice_paths(d, hk, T)[0], ""]
                continue
            o = t2_slice(S, a)
            OUT["T2"][S.key] = o
            MD += t2_md(S, o)
            log("T2 %s (%s): shares %s; z* at the inner-root |w| %s; z_out there %s" % (S.key, S.label_text, [fx(c["share_json"], "%.3f") for c in o["channels"]], fx(o["z_star"]["inner_root_dense_w"], "%.4f"), fx(o["z_out"]["inner_root_dense_w"], "%.4f")))  # v1b
    if "T3" in tests:
        MD += ["## T3", "", TESTS_VERBATIM["T3"], "", AMENDED["T3"], ""]  # v1b
        OUT["T3"] = {}
        for hk in histories:
            if not SL[hk]:
                continue
            o = t3_history(SL[hk], a)
            OUT["T3"][hk] = o
            MD += t3_md(hk, o, labels[hk])
            log("T3 %s: E_K,dust %s; work intervals %s" % (hk, [fx(r["E_K_dust"], "%.4f") for r in o["rows"]], [w_["interval"] for w_ in o["work"]]))
    if "T4" in tests:
        MD += ["## T4", "", TESTS_VERBATIM["T4"], "", AMENDED["T4"], ""]  # v1b
        OUT["T4"] = {}
        for hk in histories:
            o = t4_history(hk, REC.get(hk), SL[hk], a)
            OUT["T4"][hk] = o
            MD += t4_md(hk, o, labels[hk])
            if o.get("records_available"):
                for nm in ("R_t", "R_in", "E"):
                    d = o["series"][nm]
                    if "dft_mean_removed" in d:
                        log("T4 %s %s: plain peak period %s r_s (bin half-width %s), padded %s +- %s r_s" % (hk, nm, fx(d["dft_mean_removed"]["plain"]["period_rs"], "%.3f"), fx(d["dft_mean_removed"]["plain"]["bin_half_width_period_rs"], "%.3f"), fx(d["dft_mean_removed"]["padded"]["period_rs"], "%.3f"), fx(d["dft_mean_removed"]["padded"]["half_width_period_rs"], "%.3f")))
                    else:
                        log("T4 %s %s: %s" % (hk, nm, d.get("dft")))
    if "T5" in tests:
        MD += ["## T5", "", TESTS_VERBATIM["T5"], "", AMENDED["T5"], "", "The stress split: not available on every slice (no stored array holds p_r, per species or total); the energy-flux split prints.", ""]  # v1b
        OUT["T5"] = {}
        for hk in histories:
            if not SL[hk]:
                continue
            o = t5_history(SL[hk], a)
            OUT["T5"][hk] = o
            MD += t5_md(hk, o, labels[hk])
    if "T6" in tests:
        MD += ["## T6", "", TESTS_VERBATIM["T6"], "", AMENDED["T6"], ""]  # v1b
        OUT["T6"] = {}
        for hk in histories:
            if not SL[hk]:
                continue
            o = t6_history(SL[hk], a)
            OUT["T6"][hk] = o
            MD += t6_md(hk, o, labels[hk])
            log("T6 %s: dense integrals (r_s) %s" % (hk, [fx(r["integral_dense_rs"], "%.4f") for r in o["rows"]]))
    if "T7" in tests:
        MD += ["## T7", "", TESTS_VERBATIM["T7"], "", AMENDED["T7"], ""]  # v1b
        OUT["T7"] = {}
        for hk in histories:
            if not SL[hk]:
                continue
            for S in SL[hk]:
                if S.key not in rho_cache:
                    rho_cache[S.key] = kernel_average(a.kernel, S.R, S.R, S.eps, S.ell)
            o = t7_history(SL[hk], a, rho_cache)
            OUT["T7"][hk] = o
            MD += t7_md(hk, o, labels[hk])
    if "T8" in tests:
        MD += ["## T8", "", TESTS_VERBATIM["T8"], "", AMENDED["T8"], ""]  # v1b
        all_slices = [S for hk in histories for S in SL[hk]]
        o = t8_all(all_slices, a)
        OUT["T8"] = o
        MD += t8_md(o)
        log("T8: %s" % o["line"])
    OUT["seconds"] = time.time() - T0
    MD += ["## Checks and elapsed", "", "- T1: the kernel's normalization defect on the fine grid from r = 0 per slice (the kernel's volume integral against sum eps_tot; roundoff expected): %s." % ({k_: fx(v_["normalization"]["defect_fine_rel"], "%.1e") for k_, v_ in (OUT.get("T1") or {}).items()}),  # v1b
           "- T2: the recomputed inner-shell share minus the json's per channel (the npz against the json; the producer's convention): %s." % ({k_: [fx(c["share_check_npz_minus_json"], "%.1e") for c in v_.get("channels", [])] for k_, v_ in (OUT.get("T2") or {}).items()}),
           "- %.0f s in all; every number above is printed by the script from the stored outputs; the reading is the lead's." % OUT["seconds"]]
    with open(os.path.join(out_dir, TITLE + ".md"), "w") as f:
        f.write("\n".join(MD) + "\n")
    with open(os.path.join(out_dir, TITLE + ".json"), "w") as f:
        json.dump(OUT, f, default=jdefault, indent=1)
    if arrays:
        np.savez(os.path.join(out_dir, TITLE + "_arrays.npz"), **arrays)
    log("%s: written to %s (%.0f s)" % (TITLE, out_dir, OUT["seconds"]))
    return OUT


# ------------------------------------------------------------------------------------------------------------- the self-test
def write_synthetic_history(d, history="syn", clocks=(10.0, 11.0), nsites=96, M=256, lam=2.0, E=150.0, rc=30.0, ell=4.0, m=0.2, g=0.25, rs=64.0):
    """a tiny synthetic history in the schema the readers use: a constant-lambda mass function M = E R^3/(R^3 + rc^3) (lam = 2: N = sqrt(M/E)), w = -sqrt(2 g M/R)
    so that M_w = R w^2/(2 g) = M exactly (the roots where M = R/(2 g), i.e. R^3 - 2 g E R^2 + rc^3 = 0: 22.73 and 69.39 a at the defaults, both inside the chain; v1b:
    version 1 had w = -sqrt(M/(2 g R)), which stored 4 M as the mass function and put the outer root at 300 a), smooth site sources eps_tot = lambda M' and
    J_tot = (1 - lambda) M'/w, the second clock the first with the fields lifted by a known Delta N; shell-0 momentum distributions, Killing bins, currents and a
    records json with a sinusoidal root of period 8 r_s."""  # v1b
    os.makedirs(d, exist_ok=True)
    R = np.arange(1, nsites + 1, dtype=float)
    Mf = E * R ** 3 / (R ** 3 + rc ** 3)
    dMf = E * 3.0 * rc ** 3 * R ** 2 / (R ** 3 + rc ** 3) ** 2
    rho = lam * dMf / (4.0 * math.pi * R ** 2)
    w = -np.sqrt(2.0 * g * Mf / R)  # v1b: M_w = R w^2/(2 g) = M; version 1's -sqrt(Mf/(2 g R)) gave M_w = M/(2 g)^2 = 4 M
    N = (Mf / E) ** (0.5 * (lam - 1.0))
    v = w * N
    eps = 4.0 * math.pi * R ** 2 * rho
    Jt = (1.0 - lam) * dMf / w
    zeros = []
    f = N + v
    for i in range(nsites - 1):
        if f[i] * f[i + 1] < 0:
            zeros.append(float(R[i] + f[i] / (f[i] - f[i + 1])))
    k = 2.0 * math.pi * np.fft.fftfreq(M)
    p = np.mod(k + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    rows = []
    T_rec = np.arange(18.375, 40.0 + 1e-9, 0.125)
    for T in T_rec:
        rows.append({"T": float(T * rs), "T_over_rs": float(T), "E": float(108.0 - 0.02 * (T - 20.0)), "R_t": float(38.0 + 1.5 * math.sin(2.0 * math.pi * T / 8.0)), "R_in": float(10.0 + 2.0 * math.sin(2.0 * math.pi * T / 8.0 + 1.0)),
                     "gate_sigma_abs_dv_max": 0.15, "limit_label": "past the record's own limit"})
    with open(os.path.join(d, "records_%s.json" % history), "w") as fh:
        json.dump({"available": True, "rows": rows, "history": history, "code": "synthetic"}, fh)
    for j, T in enumerate(clocks):
        dN = 0.01 * j
        Nj, vj = N + dN, v * (1.0 + 0.005 * j)
        Nd, vd = Nj * (1.0 + 1e-3), vj * (1.0 - 1e-3)
        wd, wa = vd / Nd, vj / Nj
        A = {"R": R, "N_applied": Nj, "v_applied": vj, "N_dense_sites": Nd, "v_dense_sites": vd, "w_dense": wd, "w_applied": wa, "M_w_dense": R * wd ** 2 / (2.0 * g), "M_w_applied": R * wa ** 2 / (2.0 * g),
             "eps_tot": eps, "J_tot": Jt, "eps_pairs": 0.1 * eps, "J_pairs": 0.1 * Jt, "n_density": rho / 0.5, "rho_density": rho, "rho_over_n": np.full(nsites, 0.5), "dust_number_site": eps / 0.5,
             "dust_enclosed": np.cumsum(eps / 0.5), "L_N_dense": np.where(np.abs(np.gradient(Nd, R)) > 0, Nd / np.abs(np.gradient(Nd, R)), 1e300), "k": k, "p_reduced": p, "shell_edges_sites": np.array([0, nsites // 2, nsites])}
        chans = []
        for kap, centre, width, num in ((1, 0.0, 0.4, 5.0), (2, 1.0, 0.1, 4.0)):
            nk = np.exp(-(np.abs(p) - centre) ** 2 / (2.0 * width ** 2))
            nk *= num * M / nk.sum()
            A["k%d_shell00_nk" % kap] = nk
            share = float(nk[np.abs(p) > PI4].sum() / nk.sum())
            chans.append({"kappa": kap, "n_dust": int(num), "killing": {"bound": {"R_90": 40.0 + kap, "mean_omega_over_m": 0.35}}, "shells": [{"frac_abs_p_above": share, "cos2k": float(np.sum(nk * np.cos(2 * p)) / nk.sum()), "number": num}]})
        Ro, Ri = (max(zeros), min(zeros)) if len(zeros) >= 2 else (None, None)
        J = {"history": history, "history_label": "the synthetic history (self-test)", "T_over_rs": float(T), "rs": rs, "m": m, "g": g, "sigma_over_a": ell, "dtype": "complex128", "complex64": False, "floor": {},
             "record": {"E": E, "two_gE": 2 * g * E, "R_c": 25.0, "trapping_zeros": [Ri, Ro] if Ro else [], "solver": {"residual_W": 1e-10, "residual_invN": 1e-10, "module": "synthetic", "N_centre": float(N[0])}, "M1_balance": {"field_work": 0.0}},
             "roots": {"dense_outer": Ro, "dense_inner": Ri, "applied_outer": Ro, "applied_inner": Ri}, "gate": {"sigma_abs_dv_max": 0.15, "at_R": 30.0, "threshold": 0.1, "exceeded": True}, "limit_label": "past the record's own limit",
             "fields_source": {"dense": "synthetic (the applied fields times 1 +- 1e-3)"}, "dust": {"R_50": 24.0, "R_90": 48.0}, "core_fit": {"windows": {"R50": {"epsilon_c_total": 0.5}}},
             "killing": {"bins_weighted": [{"label": "omega <= -b", "N": 0.0, "E": 0.0}, {"label": "-b < omega <= -m", "N": 0.0, "E": 0.0}, {"label": "-m < omega <= 0", "N": 0.0, "E": 0.0},
                                           {"label": "0 < omega <= m", "N": 630.0, "E": 40.0 + 5.0 * j}, {"label": "m < omega <= b", "N": 10.0 * j, "E": 2.0 * j}, {"label": "omega > b", "N": 0.0, "E": 0.0}]},
             "fountain": {"mean_omega_over_m_bound": 0.35, "r_turn_rs_nominal": rs / (1 - 0.35 ** 2), "r_turn_2gE": 2 * g * E / (1 - 0.35 ** 2), "R_90_all": 100.0 + 20.0 * j},
             "currents": {"dense_outer": {"R_cut": Ro, "N_applied_at_cut": 0.9, "weighted": {"dev_FM": -0.04, "dust_FM": -0.03, "pairs_FM": -0.01, "dust_FM_low": -0.025, "dust_FM_high": 0.005, "dust_FM_cross": -0.01, "dust_number": -0.1, "pairs_number": 1e-8}},
                          "applied_outer": {"R_cut": Ro, "N_applied_at_cut": 0.9, "weighted": {"dev_FM": -0.041, "dust_FM": -0.031, "pairs_FM": -0.01, "dust_FM_low": -0.025, "dust_FM_high": 0.005, "dust_FM_cross": -0.011}}},
             "channels": chans}
        pj, pz, pm = slice_paths(d, history, T)
        np.savez(pz, **A)
        with open(pj, "w") as fh:
            json.dump(J, fh, default=jdefault)
    return {"zeros": zeros, "lam": lam, "E": E, "R": R, "N": N, "w": w, "dN": 0.01, "eps": eps, "g": g, "rc": rc, "ell": ell, "R_50": 24.0}  # v1b


def selftest(a, log):
    out_dir = os.path.join(a.out, "selftest")
    os.makedirs(out_dir, exist_ok=True)
    checks = []

    def chk(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        log("%s %s: %s" % ("ok  " if ok else "FAIL", name, detail))

    def guarded(name, fn):  # v1b: a pipeline check that raises is a failed check under its own name, never a duplicate of another's
        try:  # v1b
            ok, detail = fn()  # v1b
        except Exception as ex:  # noqa: BLE001  # v1b
            ok, detail = False, "%s: %s (the check itself raised)" % (type(ex).__name__, ex)  # v1b
        chk(name, ok, detail)  # v1b

    # T1: a constant-lambda profile; the identity's residual is zero analytically; the discrete lambda is off by the centred difference's own truncation error, which shrinks as h^2
    lam, E, rc = 2.4, 100.0, 24.0
    res_at = {}
    for h in (1.0, 0.125):
        R = np.arange(1.0, 256.0 + 0.5 * h, h)
        Mf = E * R ** 3 / (R ** 3 + rc ** 3)
        dMf = E * 3.0 * rc ** 3 * R ** 2 / (R ** 3 + rc ** 3) ** 2
        rho = lam * dMf / (4.0 * math.pi * R ** 2)
        N = (Mf / E) ** (0.5 * (lam - 1.0))
        D = t1_core(R, Mf, N, rho)
        pred = {}  # v1b
        for r_ in (16.0, 32.0, 64.0):
            i = int(np.argmin(np.abs(R - r_)))
            res_at[(h, r_)] = (abs(float(D["res_rel"][i])), abs(float(D["lambda"][i]) - lam))
            pred[r_] = lam * h * h / 6.0 * abs(m3_over_m1(r_, rc))  # v1b
        chk("T1 lambda at h = %g against the centred difference's leading truncation error" % h,  # v1b
            all(abs(res_at[(h, r_)][1] - pred[r_]) <= 0.05 * pred[r_] + 1e-9 for r_ in (16.0, 32.0, 64.0)),  # v1b
            "|lambda - %.1f| at R 16, 32, 64: %s against lambda (h^2/6)|M'''/M'| = %s (to 5 percent: the prescribed method's own floor on this profile, not a tolerance; version 1's 5e-3 lay below that floor at R 16)" % (  # v1b
                lam, [fx(res_at[(h, r_)][1], "%.3e") for r_ in (16.0, 32.0, 64.0)], [fx(pred[r_], "%.3e") for r_ in (16.0, 32.0, 64.0)]))  # v1b
        tol_res = {1.0: 5e-3, 0.125: 1e-4}[h]  # v1b
        chk("T1 identity residual at h = %g" % h, all(res_at[(h, r_)][0] < tol_res for r_ in (16.0, 32.0, 64.0)), "relative residual at R 16, 32, 64: %s (below %.0e)" % ([fx(res_at[(h, r_)][0], "%.2e") for r_ in (16.0, 32.0, 64.0)], tol_res))  # v1b
        i1, i2 = int(np.argmin(np.abs(R - 16.0))), int(np.argmin(np.abs(R - 64.0)))
        sec = (math.log(N[i2]) - math.log(N[i1])) / (math.log(Mf[i2]) - math.log(Mf[i1]))
        chk("T1 secant of the power law at h = %g" % h, abs(sec - 0.5 * (lam - 1.0)) < 1e-10, "secant d ln N/d ln M 16 to 64 a = %.12f against (lambda - 1)/2 = %.12f" % (sec, 0.5 * (lam - 1.0)))
    ratio = res_at[(1.0, 32.0)][0] / max(res_at[(0.125, 32.0)][0], 1e-300)
    chk("T1 residual converges as h^2", ratio > 30.0, "residual(h = 1)/residual(h = 1/8) at R 32 = %.1f (64 expected for h^2)" % ratio)
    # the kernel's normalization (the volume integral from r = 0, v1b) and its action on a uniform density
    Rn = np.arange(1.0, 257.0)
    for kern in ("shell3d", "radial1d"):
        for Rsh in (1.0, 40.0):
            src = np.zeros(Rn.size)
            src[int(Rsh) - 1] = 1.0
            nc = normalization_check(kern, Rn, src, 4.0)
            chk("kernel %s normalization, shell at %g a" % (kern, Rsh), abs(nc["defect_fine_rel"]) < 1e-10,  # v1b
                "volume integral from r = 0 on the 0.25 a grid %.12f against 1 (spectrally exact for the even analytic integrand; the site-grid trapezoid from 1 a %.6f, informational: it misses r < 1 a; version 1 began the fine grid at 0.25 a and lost the mass inside)" % (nc["integral_fine_grid"], nc["integral_site_trapezoid"]))  # v1b
    n0 = 0.01
    rho_u = kernel_average("shell3d", Rn, Rn, 4.0 * math.pi * Rn ** 2 * n0, 4.0)
    sel = (Rn >= 20) & (Rn <= 256 - 24)  # v1b: 6 ell inside the chain's end, where the chain's missing Gaussian tail is below 1e-9
    chk("kernel shell3d preserves a uniform density (20 to 232 a, 6 ell inside the chain's end)", float(np.max(np.abs(rho_u[sel] / n0 - 1.0))) < 1e-8,  # v1b
        "max |rho_ell/n0 - 1| on 20 to 232 a = %.2e (version 1's window reached 236 a, 5 ell from the chain's end, where the chain, not the kernel, leaves the Gaussian tail 1.5e-7 unfilled)" % float(np.max(np.abs(rho_u[sel] / n0 - 1.0))))  # v1b
    exp236 = chain_end_tail_defect(236.0, 256.0, 4.0)  # v1b
    meas236 = float(1.0 - rho_u[235] / n0)  # v1b
    chk("kernel shell3d at 236 a: the chain's end, not the kernel", abs(meas236 - exp236) <= 0.1 * exp236,  # v1b
        "1 - rho_ell/n0 at 236 a = %.3e against the chain-end deficit Q((R_end + 1/2 - r)/ell) + (ell/r) phi - |F'(R_end + 1/2)|/24 = %.3e (derived; to 10 percent)" % (meas236, exp236))  # v1b
    rho_r = kernel_average("radial1d", Rn, Rn, 4.0 * math.pi * Rn ** 2 * n0, 4.0)
    chk("kernel radial1d bias on a uniform density", abs(float(rho_r[39] / n0 - 1.0) - 16.0 / 1600.0) < 2e-3, "rho_ell/n0 - 1 at 40 a = %.4f against the O(ell^2/r^2) = 0.0100 bias" % float(rho_r[39] / n0 - 1.0))
    # T2: z*, z_out and the momentum reader
    z12 = z_star(1.2, 0.2)
    chk("T2 z* at |w| = 1.2, m a = 0.2", 0.98 < z12 < 1.04, "z* = %.4f rad against the push's 'z approx 1.0 rad' (2.2)" % z12)
    zb = z_star(1e4, 0.2)
    chk("T2 z* -> pi/4 as |w| -> infinity", 0 < zb - PI4 < 0.02, "z*(|w| = 1e4) - pi/4 = %.4f" % (zb - PI4))
    zo1 = z_out(1.0, 0.2)  # v1b
    chk("T2 z_out at |w| = 1, m a = 0.2 (CORE_well 7's a p = 0.32)", zo1 is not None and 0.31 < zo1 < 0.33, "z_out = %s rad; at |w| = 1.2 %s rad (the massive symbol; the red team's massless 0.33)" % (fx(zo1, "%.4f"), fx(z_out(1.2, 0.2), "%.4f")))  # v1b
    k = 2.0 * math.pi * np.fft.fftfreq(512)
    p = np.mod(k + 0.5 * math.pi, math.pi) - 0.5 * math.pi
    nk = np.where(np.abs(np.abs(p) - 0.5) < 0.01, 1.0, 0.0)
    e = momentum_edge(nk, p, {"pi/4": PI4, "z": z12})
    chk("T2 momentum reader on weight at |p| = 0.5", abs(e["p_90"] - 0.5) < 0.02 and e["shares_above"]["pi/4"] == 0.0, "p_90 %.4f, share above pi/4 %.3f, share above z* %.3f" % (e["p_90"], e["shares_above"]["pi/4"], e["shares_above"]["z"]))
    nk2 = np.where(np.abs(np.abs(p) - 1.2) < 0.01, 1.0, 0.0)
    e2 = momentum_edge(nk2, p, {"pi/4": PI4, "z": z12})
    chk("T2 momentum reader on weight at |p| = 1.2 (the directional halves, v1b)", e2["shares_above"]["pi/4"] == 1.0 and e2["shares_above"]["z"] == 1.0 and abs(e2["shares_outward_above"]["pi/4"] - 0.5) < 1e-3 and abs(e2["shares_inward_above"]["pi/4"] - 0.5) < 1e-3,  # v1b
        "shares above pi/4 and z*: %.3f, %.3f; outward p > pi/4 %.4f, inward p < -pi/4 %.4f (the symmetric weight halves, up to the fftfreq grid's endpoint)" % (e2["shares_above"]["pi/4"], e2["shares_above"]["z"], e2["shares_outward_above"]["pi/4"], e2["shares_inward_above"]["pi/4"]))  # v1b
    # T4: sinusoids of known period on the 20 r_s window (the periodogram's resolution, v1b) and on a 50-cycle record (the arithmetic, v1b)
    Tg = np.arange(18.375, 40.0 + 1e-9, 0.125)
    rng = np.random.RandomState(7)
    for P in (20.125 / 3.0, 8.0):
        y = 38.0 + 1.5 * np.sin(2.0 * math.pi * Tg / P) + 0.05 * (Tg - 20.0) + 0.02 * rng.standard_normal(Tg.size)
        d = dft_period(Tg, y, 20.0, 40.0, 0.125, 16, True)
        f0, df = 1.0 / P, d["bin_width_per_rs"]  # v1b
        hw = d["padded"]["half_width_period_rs"]  # v1b
        ok_plain = abs(d["plain"]["f_peak"] - f0) <= df + 1e-12  # v1b
        ok_pad = hw is not None and abs(d["padded"]["period_rs"] - P) <= hw  # v1b
        chk("T4 sinusoid of period %.4f r_s in the 20 r_s window (%.1f cycles): the periodogram's resolution" % (P, 20.0 / P), ok_plain and ok_pad,  # v1b
            "plain peak bin period %.4f (the bins are 20.125/k r_s; 1/P lies within one bin width %.4f per r_s of the peak bin's frequency %.4f), padded %.4f +- %.3f (P within the half-power half-width; the offset %.3f r_s is the %.1f-cycle record's leakage and line removal, not an error to tolerate; version 1 asked 0.25 r_s of a 2.5-cycle record)" % (  # v1b
                d["plain"]["period_rs"], df, d["plain"]["f_peak"], d["padded"]["period_rs"], hw, abs(d["padded"]["period_rs"] - P), 20.0 / P))  # v1b
    Tl = np.arange(0.0, 400.0 + 1e-9, 0.125)  # v1b
    yl = 38.0 + 1.5 * np.sin(2.0 * math.pi * Tl / 8.0) + 0.05 * (Tl - 20.0) + 0.02 * rng.standard_normal(Tl.size)  # v1b
    dl = dft_period(Tl, yl, 0.0, 400.0, 0.125, 16, True)  # v1b
    chk("T4 sinusoid of period 8 r_s on a 50-cycle record (0 to 400 r_s): the arithmetic", abs(dl["padded"]["period_rs"] - 8.0) < 0.02 and dl["padded"]["half_width_period_rs"] is not None and dl["padded"]["half_width_period_rs"] < 0.1,  # v1b
        "padded period %.4f +- %.4f r_s against 8 (the padded grid's step at 8 r_s is 0.010 r_s; the bias scales as the inverse square of the cycle count)" % (dl["padded"]["period_rs"], dl["padded"]["half_width_period_rs"]))  # v1b
    # T4: the PG fall and the literal integral
    t1_, t2_ = pg_ingoing_time(0.5, 64.0, 64.0 / (1 - 0.25), 4000), pg_ingoing_time(0.5, 64.0, 64.0 / (1 - 0.25), 16000)
    chk("T4 PG ingoing time quadrature", abs(t1_ - t2_) / t2_ < 1e-6 and t1_ > 0, "T_in(e = 0.5, r_s = 64) = %.6f a at 4000 and %.6f a at 16000 points" % (t1_, t2_))
    val, status = literal_null_roundtrip(37.5, 75.0, 64.0)
    chk("T4 literal integral diverges with the root inside r_s", val is None and status.startswith("divergent"), status)
    val2, status2 = literal_null_roundtrip(70.0, 75.0, 64.0)
    chk("T4 literal integral finite outside r_s", val2 is not None and val2 > 0, "2 int_70^75 dr/(1 - sqrt(64/r)) = %s a" % fx(val2, "%.4f"))
    # T6: constant integrand
    Rg = np.arange(1.0, 129.0)
    I6 = integrate_between(Rg, 1.0 / (0.5 * (1.0 + 1.0)) * np.ones(Rg.size), 7.3, 51.6)
    chk("T6 integral of a constant integrand", abs(I6 - (51.6 - 7.3)) < 1e-9, "int = %.9f against %.9f" % (I6, 51.6 - 7.3))
    # T7: an exact R^2 law
    Rq = np.arange(1.0, 9.0)
    yq = np.log(0.3 * np.exp(-0.004 * Rq ** 2)) - math.log(0.3 * math.exp(-0.004))
    s7, c7, r27 = fit_lin(Rq ** 2, yq)
    chk("T7 fit of an exact R^2 law", abs(s7 + 0.004) < 1e-12 and abs(1.0 - r27) < 1e-12, "slope %.12f (expected -0.004), r2 - 1 = %.1e" % (s7, r27 - 1.0))
    # T8: a pass and a counter-example on manufactured mass functions
    class _Fake:
        pass

    def fake(Mfun, E_, Ro):
        F = _Fake()
        F.R = np.arange(1.0, 129.0)
        F.M_w_d = Mfun(F.R)
        F.M_w_a = F.M_w_d
        F.Ro_d = F.Ro_a = Ro
        F.Ri_d = F.Ri_a = 5.0
        F.E = E_
        F.R_c = 25.0
        F.rooted = True
        F.history, F.T, F.key, F.c64 = "fake", 1.0, "fake_T0001.00", False
        F.short_label = lambda: "synthetic"
        return F
    o8 = t8_all([fake(lambda R: np.minimum(2.0 * R, 90.0), 100.0, 40.0)], a)
    chk("T8 passes on a monotone mass function below E", o8["all_pass_dense"] and o8["line"].startswith("the mass function peaks at the root below E on all 1 rooted"), o8["line"])
    o8b = t8_all([fake(lambda R: 2.0 * R + 40.0 * np.exp(-((R - 20.0) / 5.0) ** 2), 100.0, 40.0)], a)
    chk("T8 names the counter-example", not o8b["all_pass_dense"] and o8b["line"].startswith("not printed"), o8b["line"][:160])
    # the readers end to end on a tiny synthetic history (v1b: rooted, by the corrected flow; every pipeline check guarded and named once)
    syn = os.path.join(out_dir, "synthetic_well")
    info = write_synthetic_history(syn)
    cubic = np.roots([1.0, -2.0 * info["g"] * info["E"], 0.0, info["rc"] ** 3])  # v1b: the roots of N + v = 0 where M = R/(2 g): R^3 - 2 g E R^2 + r_c^3 = 0
    roots_c = sorted(float(x.real) for x in cubic if abs(x.imag) < 1e-9 and x.real > 0)  # v1b
    chk("the synthetic history is rooted: its 2 zeros of N + v against the roots of R^3 - 2 g E R^2 + r_c^3 = 0",  # v1b
        len(info["zeros"]) == 2 and len(roots_c) == 2 and all(abs(z_ - c_) < 0.01 for z_, c_ in zip(sorted(info["zeros"]), roots_c)),  # v1b
        "zeros %s a against %s a (version 1's w = -sqrt(M/(2 g R)) put the inner root at 9.6 a and the outer beyond the 96-site chain)" % ([fx(z_, "%.3f") for z_ in info["zeros"]], [fx(c_, "%.3f") for c_ in roots_c]))  # v1b
    O = None  # v1b
    try:
        O = run_tests(a, log, well=syn, rung=syn, out_dir=os.path.join(out_dir, "synthetic_out"), histories=["syn"], clocks={"syn": [10.0, 11.0]})
        ok_files = os.path.isfile(os.path.join(out_dir, "synthetic_out", TITLE + ".md")) and os.path.isfile(os.path.join(out_dir, "synthetic_out", TITLE + ".json"))
        chk("readers end to end on the synthetic history", ok_files, "md and json written; slices read %s; synthetic roots %s" % (O["inputs"]["slices_read"], [fx(z, "%.3f") for z in info["zeros"]]))
    except Exception as ex:  # noqa: BLE001
        chk("readers end to end on the synthetic history", False, "%s: %s" % (type(ex).__name__, ex))
    if O is not None:  # v1b
        def c_t1():  # v1b
            t1s = O["T1"]["syn_T0010.00"]  # v1b
            lam50 = [q["lambda_dense"] for q in t1s["radii"] if q["name"] == "R_50"]  # v1b
            R50, lam0, ell = info["R_50"], info["lam"], info["ell"]  # v1b
            bias = 0.5 * ell * ell * laplacian_ratio(R50, info["rc"])  # the three-dimensional Gaussian's leading bias (ell^2/2) nabla^2 rho/rho at R_50
            trunc = (1.0 / 6.0) * m3_over_m1(R50, info["rc"])  # the centred difference's relative error of M_w' at h = 1
            scale = ((1.0 - 1e-3) / (1.0 + 1e-3)) ** 2  # the synthetic's dense M_w = scale x M
            expected = lam0 * (1.0 + bias) / (scale * (1.0 + trunc))  # v1b
            ok = bool(lam50) and abs(lam50[0] - expected) < 0.02  # v1b
            return ok, "lambda at R_50 = %s against lambda_0 (1 + (ell^2/2) nabla^2 rho/rho)/(scale (1 + h^2 M'''/(6 M'))) = %.4f x (1 %+.4f)/(%.6f x (1 %+.5f)) = %.4f (derived; the O(ell^4) remainder below 0.01; version 1 read 0.4885 because its mass function was 4 M)" % (  # v1b
                fx(lam50[0] if lam50 else None, "%.4f"), lam0, bias, scale, trunc, expected)  # v1b
        guarded("T1 pipeline on the synthetic: lambda at R_50 against lambda_0 with the kernel's derived O(ell^2) bias", c_t1)  # v1b

        def c_halo():  # v1b
            h_ = O["T1"]["syn_T0010.00"]["halo"]  # v1b
            ok = bool(h_.get("available")) and abs(h_["mean_lambda_dense"] - info["lam"]) < 1e-4 and abs(h_["edge_lapse_law_over_measured"] - 1.0) < 0.05  # v1b
            return ok, "the halo R_c %s to the root %s a: mean lambda 1 + 2 secant %s against %.1f (the secant is kernel-free), the edge lapse law over measured %s (the kernel's O(ell^2) bias on the pointwise lambda)" % (  # v1b
                fx(h_.get("R_c"), "%.1f"), fx(h_.get("R_root"), "%.3f"), fx(h_.get("mean_lambda_dense"), "%.6f"), info["lam"], fx(h_.get("edge_lapse_law_over_measured"), "%.4f"))  # v1b
        guarded("T1 pipeline on the synthetic: the halo's mean lambda and the edge lapse (red team premise (b))", c_halo)  # v1b

        def c_win():  # v1b
            w_ = O["T1"]["syn_T0010.00"]["window"]  # v1b
            ok = bool(w_.get("available")) and abs(w_["secant_lnM_dense"] - 0.5 * (info["lam"] - 1.0)) < 1e-9  # v1b
            return ok, "the window 1 a to R_50: secant d ln N/d ln M_w %s against (lambda - 1)/2 = %.3f; weighted mean of (lambda - 1)/2 from rho_ell %s" % (fx(w_.get("secant_lnM_dense"), "%.9f"), 0.5 * (info["lam"] - 1.0), fx(w_.get("weighted_half_dense"), "%.4f"))  # v1b
        guarded("T1 pipeline on the synthetic: the R <= R_50 window's secant", c_win)  # v1b

        def c_t2():  # v1b
            t2s = O["T2"]  # v1b
            absent = bool(t2s) and all("status" in v_ for v_ in t2s.values())  # v1b
            Ssyn = Slice(syn, "syn", 10.0)  # v1b
            o2 = t2_slice(Ssyn, a)  # v1b
            ch = o2["channels"]  # v1b
            e0, e1 = ch[0]["edge"], ch[1]["edge"]  # v1b
            ok = (absent and len(ch) == 2 and all(abs(c_["share_check_npz_minus_json"]) < 1e-12 for c_ in ch) and ch[0]["four_a_n_rad"] == 20.0 and ch[1]["four_a_n_rad"] == 16.0  # v1b
                  and o2["regime"]["four_a_n_rad_max"] == 20.0 and abs(e1["shares_outward_above"]["pi/4"] - 0.5) < 1e-3 and abs(e1["shares_inward_above"]["pi/4"] - 0.5) < 1e-3  # v1b
                  and abs(e0["shares_outward_above"]["pi/4"] + e0["shares_inward_above"]["pi/4"] - e0["shares_above"]["pi/4"]) < 1e-12 and o2["z_star"]["inner_root_dense_w"] is not None and o2["z_out"]["inner_root_dense_w"] is not None)  # v1b
            return ok, "rung targets absent %s; share check npz - json %s; 4 a n_rad %s; n_rad,max 4a %s; the symmetric weight at |p| = 1 splits %.4f outward, %.4f inward; z* %s, z_out %s at the inner-root |w| %s" % (  # v1b
                absent, [fx(c_["share_check_npz_minus_json"], "%.1e") for c_ in ch], [c_["four_a_n_rad"] for c_ in ch], o2["regime"]["four_a_n_rad_max"], e1["shares_outward_above"]["pi/4"], e1["shares_inward_above"]["pi/4"],  # v1b
                fx(o2["z_star"]["inner_root_dense_w"], "%.4f"), fx(o2["z_out"]["inner_root_dense_w"], "%.4f"), fx(o2["abs_w_inner_root_dense"], "%.4f"))  # v1b
        guarded("T2 pipeline: the rung targets absent are reported; the shares, the regime radii and the directional halves on the synthetic", c_t2)  # v1b

        def c_t3():  # v1b
            wk = O["T3"]["syn"]["work"][0]["applied"]["total"]  # v1b
            expected = 0.01 * float(np.sum(info["eps"]))  # v1b
            return abs(wk["dN_rho"] - expected) < 1e-9, "sum Delta N eps = %.6f against %.6f (Delta N = 0.01 everywhere); Delta v J part %.6f" % (wk["dN_rho"], expected, wk["dv_J"])  # v1b
        guarded("T3 pipeline: the Delta N eps part of the work on the synthetic", c_t3)  # v1b

        def c_t3b():  # v1b
            r1 = O["T3"]["syn"]["rows"][1]  # v1b
            return abs(r1["mean_omega_over_m_mb"] - 1.0) < 1e-12 and O["T3"]["syn"]["rows"][0]["mean_omega_over_m_mb"] is None, "E_(m,b]/(m N_(m,b]) at the second clock %s (E 2, N 10, m 0.2: 1.0); None at the first (N 0)" % fx(r1["mean_omega_over_m_mb"], "%.6f")  # v1b
        guarded("T3 pipeline: the energy per shed particle column (v1b)", c_t3b)  # v1b

        def c_t4():  # v1b
            d4 = O["T4"]["syn"]["series"]["R_t"]["dft_detrended"]  # v1b
            hw = d4["padded"]["half_width_period_rs"]  # v1b
            ok = abs(d4["plain"]["f_peak"] - 1.0 / 8.0) <= d4["bin_width_per_rs"] + 1e-12 and hw is not None and abs(d4["padded"]["period_rs"] - 8.0) <= hw  # v1b
            return ok, "the synthetic root's plain bin %.4f r_s, padded %.4f +- %.3f r_s against 8 (the same 2.5-cycle resolution as the sinusoid check); 2 pi/P over T_H %s" % (d4["plain"]["period_rs"], d4["padded"]["period_rs"], hw, fx(d4["padded"]["omega_over_T_H"], "%.2f"))  # v1b
        guarded("T4 pipeline: the synthetic root's period 8 r_s within the periodogram's resolution", c_t4)  # v1b

        def c_t5():  # v1b
            r5 = O["T5"]["syn"]["rows"][0]  # v1b
            return abs(r5["pairs_share_of_matter_FM"] - 0.25) < 1e-12 and r5["signs"]["dev_FM"] == "-" and r5["T_ll_stress_split"].startswith("not available"), "pairs share %s (-0.01/-0.04), sign %s, stress split: %s" % (fx(r5["pairs_share_of_matter_FM"], "%.4f"), r5["signs"]["dev_FM"], r5["T_ll_stress_split"][:13])  # v1b
        guarded("T5 pipeline on the synthetic", c_t5)  # v1b

        def c_t6():  # v1b
            rows6 = O["T6"]["syn"]["rows"]  # v1b
            if not rows6:  # v1b
                return False, "no T6 row: the synthetic is not rooted"  # v1b
            t6s = rows6[0]  # v1b
            Rz = info["R"]  # v1b
            Nd = info["N"] * (1.0 + 1e-3)  # v1b
            wd = info["w"] * (1.0 - 1e-3) / (1.0 + 1e-3)  # v1b
            ref6 = integrate_between(Rz, 1.0 / (Nd * (1.0 + np.abs(wd))), min(info["zeros"]), max(info["zeros"]))  # v1b
            return abs(t6s["integral_dense_a"] - ref6) < 1e-9, "integral %.6f a against the direct %.6f a between the roots %s" % (t6s["integral_dense_a"], ref6, [fx(z_, "%.3f") for z_ in info["zeros"]])  # v1b
        guarded("T6 pipeline on the synthetic", c_t6)  # v1b

        def c_t8():  # v1b
            return bool(O["T8"]["all_pass_dense"]) and O["T8"]["rooted_slices_dense"] == 2, O["T8"]["line"]  # v1b
        guarded("T8 pipeline on the synthetic (2 rooted slices)", c_t8)  # v1b
    n_ok = sum(1 for c in checks if c["ok"])
    L = ["# %s self-test (%s)" % (TITLE, CODE), "", "%d of %d checks passed." % (n_ok, len(checks)), "", "| check | ok | detail |", "|---|---|---|"]
    L += ["| %s | %s | %s |" % (c["name"], "yes" if c["ok"] else "NO", c["detail"].replace("|", "/")) for c in checks]
    L += ["", "- manufactured: a constant-lambda profile M = E R^3/(R^3 + r_c^3), rho = lambda M'/(4 pi R^2), N = (M/E)^((lambda - 1)/2) (the identity's residual zero analytically; the discrete lambda off by the centred difference's own truncation error lambda (h^2/6)|M'''/M'|); shells at 1 and 40 a and a uniform density for the kernel (the volume integral from r = 0; the chain-end deficit at 236 a derived); sinusoids of period 20.125/3 and 8 r_s with a line and noise on the 20 r_s window and a 50-cycle record of period 8 r_s; an exact R^2 law; a monotone and a bumped mass function; a tiny synthetic history in the readers' schema with w = -sqrt(2 g M/R), rooted at 22.73 and 69.39 a (v1b).", "- %.0f s." % (time.time() - T0)]  # v1b
    with open(os.path.join(out_dir, "ASYMPTOTIC_selftest_v1b.md"), "w") as f:  # v1b
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "ASYMPTOTIC_selftest_v1b.json"), "w") as f:  # v1b
        json.dump({"checks": checks, "passed": n_ok, "total": len(checks), "seconds": time.time() - T0}, f, default=jdefault, indent=1)
    log("self-test: %d of %d checks passed; %s" % (n_ok, len(checks), out_dir))
    return n_ok == len(checks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--well", default=WELL_DIR, help="the directory of core_well_v1's per-slice outputs and records_<history>.json")
    ap.add_argument("--rung", default=RUNG_DIR, help="the directory of the rung stand-ins' per-slice outputs (T2 only; read where present)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--histories", default="prod,s8")
    ap.add_argument("--clocks", default="all", help="all or a comma list of T/r_s (applied to every history)")
    ap.add_argument("--tests", default="all", help="all or a comma list among T1..T8")
    ap.add_argument("--kernel", default="shell3d", choices=sorted(KERNELS), help="the Gaussian kernel of T1's rho_ell route (the engine's form is not among the stored outputs)")
    ap.add_argument("--threads", type=int, default=None, help="BLAS threads (8 where not given)")
    ap.add_argument("--max-load", dest="max_load", type=float, default=90.0, help="wait while the 1-minute load average exceeds this percent of the CPUs")
    ap.add_argument("--pad", type=int, default=16, help="the zero-padding factor of T4's refined peak")
    ap.add_argument("--t8-tol", dest="t8_tol", type=float, default=1e-3, help="T8's tolerance on M_w's monotonicity and peak, relative to M_w at the root")
    ap.add_argument("--self-test", dest="self_test", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "asymptotic_v1b.log"))  # v1b
    log("%s: threads %s; kernel %s; out %s" % (CODE, os.environ.get("OMP_NUM_THREADS"), a.kernel, a.out))
    if a.self_test:
        ok = selftest(a, log)
        sys.exit(0 if ok else 1)
    run_tests(a, log)


if __name__ == "__main__":
    main()
