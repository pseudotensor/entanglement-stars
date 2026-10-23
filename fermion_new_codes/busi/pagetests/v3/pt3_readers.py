#!/usr/bin/env python
"""pt3_readers.py (busi, PAGE design v3 sec. 10; A9, F4, reviewer A pc9): the pass / fail / inconclusive readers, with the predeclared budgets:
one percent of the emitted energy for charges, certified intervals for entropy signs. A failure to reach the error target is INCONCLUSIVE, not a
physical failure (reviewer B pc9 shared contract). A converged refutation is decisive; neither reference's prediction has to occur (A9).
Each reader returns {"claim", "verdict": PASS | FAIL | INCONCLUSIVE (or the claim-specific word), "why", "source"}."""
import math

PASS, FAIL, INC = "PASS", "FAIL", "INCONCLUSIVE"


def _v(claim, verdict, why, source, **kw):
    d = {"claim": claim, "verdict": verdict, "why": why, "source": source}
    d.update(kw)
    return d


def reference_charges_emission(E_packet, dM_retained, residual, E_emit, budget=0.01, resolved=True):
    """A9 line 1: the physical packet energy and the retained-charge change agree after all ledger terms, residual below budget."""
    c = "a reference charges internal emission"
    if not resolved or E_emit is None or E_emit <= 0:
        return _v(c, INC, "output not resolved or no emission", "A9")
    if abs(residual) > budget * E_emit:
        return _v(c, INC, "ledger residual %.3g above %.0f%% of E_emit %.3g" % (residual, 100 * budget, E_emit), "A9")
    ok = abs(E_packet + dM_retained) <= budget * E_emit
    return _v(c, PASS if ok else FAIL, "E_packet %.4g vs -dM_ret %.4g (budget %.3g)" % (E_packet, -dM_retained, budget * E_emit), "A9")


def ran_artefact_F4(lam, dE_ext_Ran, E_emit, eps_n, collar_jump, eps_c, dE_ext_Rloc, dE_ext_Rstar, t_first_cross, t_sweep_start,
                    window=40.0, frac=0.1):
    """F4: at lambda << 1, R-an's exterior Delta-energy = E_emit - eps_n with a collar-charge jump by the cavity level's charge within ~40 r_s
    of the sweep start (the first same-sector crossing); R-loc and R-star give E_emit; pass/fail at frac E_emit."""
    c = "R-an's artefact (F4)"
    if lam is None or lam >= 1.0:
        return _v(c, INC, "not in the lambda << 1 regime (lambda %s)" % lam, "F4")
    tol = frac * abs(E_emit)
    sig_ext = abs(dE_ext_Ran - (E_emit - eps_n)) <= tol
    sig_jump = collar_jump is not None and abs(collar_jump - eps_c) <= frac * abs(eps_c) and t_first_cross is not None and \
        (t_first_cross - t_sweep_start) <= window
    others = abs(dE_ext_Rloc - E_emit) <= tol and abs(dE_ext_Rstar - E_emit) <= tol
    if sig_ext and sig_jump and others:
        return _v(c, "SHOWN", "R-an exterior = E_emit - eps_n and the collar jump eps_c; R-loc, R-star = E_emit", "F4")
    if not sig_ext and not sig_jump:
        return _v(c, "REFUTED", "R-an shows neither signature (reviewer A pc9 Test I: reviewer A's reservation refuted in the tested regime)", "reviewer A pc9 Test I")
    return _v(c, INC, "partial signatures (ext %s, jump %s, R-loc/R-star at E_emit %s)" % (sig_ext, sig_jump, others), "F4")


def ran_artefact_A9(charge_removed, output_resolved, ledger_closed, small_C_minus_P_only=False):
    c = "R-an's artefact (reviewer B's condition)"
    if small_C_minus_P_only:
        return _v(c, INC, "small C - P or diagonal exterior energy alone is insufficient", "A9")
    if not (output_resolved and ledger_closed):
        return _v(c, INC, "output not resolved or ledger not closed", "A9")
    return _v(c, "SHOWN" if charge_removed else "NOT SHOWN", "charge removed despite resolved output and closed ledger: %s" % charge_removed, "A9")


def rloc_declaration(assignments_by_RQ, jump_sum, d_omega, Gamma_res, Delta_box, tol=0.25):
    """reviewer A pc9: assignments unchanged under the R_Q scan; the jump sum within 2 d_omega (Gamma_res/(2 pi Delta_box))^(1/2) (tol relative)."""
    c = "R-loc as a declaration"
    sets = [tuple(a) for a in assignments_by_RQ]
    same = all(s == sets[0] for s in sets)
    bound = 2.0 * abs(d_omega) * math.sqrt(Gamma_res / (2 * math.pi * Delta_box))
    within = abs(abs(jump_sum) - bound) <= tol * bound
    if not same:
        return _v(c, FAIL, "assignments change across the R_Q scan: not a declaration", "reviewer A pc9", bound=bound)
    return _v(c, PASS if within else FAIL, "jump sum %.4g against 2 d_omega sqrt(Gamma/2 pi Delta) = %.4g" % (jump_sum, bound), "reviewer A pc9", bound=bound)


def slide_fed_transfer(identified_supply, traversal_reached, escape_time_ok, renewed_flux_over_held, chirp_correlation, threshold=3.0, corr_min=0.5):
    c = "slide-fed transfer"
    if not (traversal_reached and escape_time_ok):
        return _v(c, INC, "a null needs adequate traversal and escape time", "A9; reviewer A pc9")
    ok = identified_supply and renewed_flux_over_held >= threshold and chirp_correlation >= corr_min
    return _v(c, PASS if ok else FAIL, "supply %s, renewed/held %.3g, chirp corr %.2f" % (identified_supply, renewed_flux_over_held, chirp_correlation),
              "A9; reviewer A pc9")


def kinematics(rate_rising, rate_pocket, aq=None):
    """reviewer A pc9: filled rising levels descend (rate < 0 at e > 0) and empty pockets ascend (rate > 0 at e < 0); <aq> in [1e-2, 0.3], below 1e-3 kills."""
    c = "kinematics"
    ok = rate_rising < 0 and rate_pocket > 0
    why = "rising rate %.3g, pocket rate %.3g" % (rate_rising, rate_pocket)
    if aq is not None:
        if abs(aq) < 1e-3:
            return _v(c, FAIL, why + "; |<aq>| < 1e-3 kills the refill picture", "reviewer A pc9")
        ok = ok and 1e-2 <= abs(aq) <= 0.3
    return _v(c, PASS if ok else FAIL, why, "reviewer A pc9")


def qft_agreement(discrepancy, tolerance, converged, interval_valid, late_interval=False, c6="both", comparator_reach=None, junction_shown=False):
    """A9 and C6: converged covariance and flux agree with the valid same-background comparator over the claimed interval; the late interval
    beyond the continuum's certified reach is printed per the C6 option."""
    c = "QFT agreement"
    if late_interval:
        out = []
        if c6 in ("both", "unresolved"):
            out.append(_v(c, "UNRESOLVED" if not junction_shown else INC, "beyond the continuum's certified reach %s (A7)" % comparator_reach, "A7"))
        if c6 in ("both", "dispersion"):
            out.append(_v(c, "DISPERSION EFFECT" if discrepancy > tolerance else "AGREES WITH SPLICE",
                          "against the spliced Unruh value (F11), junction shown %s" % junction_shown, "F11"))
        return out
    if not (converged and interval_valid):
        return _v(c, INC, "not converged or outside the comparator's validity", "A9")
    return _v(c, PASS if discrepancy <= tolerance else FAIL, "discrepancy %.3g vs tolerance %.3g" % (discrepancy, tolerance), "A9")


def purification(rows, floor, min_bins=2):
    """A8, F9, reviewer B pc9 Test A: PASS if some later bin's certified upper bound S(b|R)_hi < -floor, with all intervals valid; INCONCLUSIVE if
    intervals invalid or no bin beyond the first; FAIL otherwise. A Page-like curve additionally needs the resolved rise and fall (reported)."""
    c = "purification"
    late = [r for r in rows if r["n"] >= min_bins]
    if not late or not all(r["valid"] for r in rows):
        return _v(c, INC, "too few bins or an invalid entropy interval", "A8, F9")
    neg = [r["n"] for r in late if r["S_b_given_R_hi"] < -abs(floor)]
    SR = [r["S_R"] for r in rows]
    peak = max(range(len(SR)), key=lambda i: SR[i])
    rise_fall = 0 < peak < len(SR) - 1
    return _v(c, PASS if neg else FAIL, "bins with certified S(b|R) < -floor: %s; S(R) rises and falls: %s" % (neg, rise_fall), "A8, A9, F9",
              page_like_rise_fall=rise_fall)


def complete_drain(peak, n_drain, late_excess, inventory, floor, tol=0.01):
    c = "complete drain"
    target = n_drain * math.log(2.0)
    ok = abs(peak - target) <= tol * target and late_excess <= inventory + floor
    return _v(c, PASS if ok else FAIL, "peak %.5g vs %d ln 2 = %.5g; late excess %.3g vs inventory + floor %.3g" % (peak, n_drain, target, late_excess, inventory + floor),
              "reviewer A pc9 Test II")


def no_turnover(interval):
    return _v("no turnover", "BOUND ONLY", "a bound over the measured interval %s and accessible modes only" % (interval,), "A9")


def charge_calibration(dE, target, omega, gauss_residual):
    c = "packet-charge calibration (ledger calibration label: does not separate arms)"
    tol = max(1e-3 * abs(omega), gauss_residual)
    return _v(c, PASS if abs(dE - target) <= tol else FAIL, "dE %.6g vs <psi|h|psi> %.6g (tol %.2g)" % (dE, target, tol), "F6, A3")
