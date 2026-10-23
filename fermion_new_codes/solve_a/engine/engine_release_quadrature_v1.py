#!/usr/bin/env python
"""engine_release_quadrature_v1.py (builder analyst; engine spec 37.3 Ruled (3): "the outer layer of the released ball from the
production's stored release radius and charge (read from its json and printed), the flat-slice time t(R) = integral of gamma/z with
z = sqrt(r_g/x - r_g/R_0) and the proper time, each in lattice units, in the lane's nominal T/rs and in units of 2gE0, beside the recorded
formation 11.906 to 12.125 and the fold clock 12.055; the lead reads in turn 46 and tells the icloud lane (its I2) the unit"). Record-only,
CPU, a quadrature.

The control (WEB40F P20 sec. 2 and 2.1, re-derived as stated there; a pressureless outermost shell released at coordinate rest from R_0
with the charge E inside it, N = 1 on it): r_g = 2 g E, e = sqrt(1 - r_g/R_0), b = sqrt(r_g/x), z = sqrt(r_g/x - r_g/R_0),
gamma = (e^2 + b^2)/(e + b z) (P20 eq. 6), the flat-slice time t(R) = int_R^R_0 gamma(x)/z(x) dx and the proper time
tau(R) = int_R^R_0 dx/z(x) (P20 eq. 10). The endpoint singularity at x = R_0 is removed by x = R_0 - u^2 (dx/z = 2 sqrt(x R_0/r_g) du),
the integral by scipy.integrate.quad at relative tolerance 1e-13. The independent check P20 states: the canonical equations
dR/dt = p/gamma - b, dp/dt = -b p/(2R), dtau/dt = 1/gamma (b = sqrt(r_g/R), N = 1) integrated from coordinate rest at R_0
(p = b/sqrt(1 - b^2), so that dR/dt = 0; Reading A) to R = r_g (DOP853, rtol 1e-12), whose t and tau must equal the quadrature's.
Printed per case: R_0 and E (the production's head values, printed as read; the T = 0 record's 99-percent dust radius R_sh as a second
release radius; P20's two controls r_g 59.05 and 64 at R_0 256 as the quadrature's validation against its table: 13.2248621796 and
12.4186017985 flat-slice and proper at 59.05, 12.6937264016 and 11.8416820247 at 64, in units of 64 a), t(r_g) and tau(r_g) in lattice
units, over the nominal r_s 64 a (the lane's T/rs) and over 2 g E0; beside them the recorded formation clocks of the production's tau 8,
4, 2 histories (their first records holding a zero) and the fold clock (engine_fold_birth_v1.crossing_events, linear) of each, read from
the jsons. Nothing is read or ruled. Args: --json (the tau 8 production) --tau4 --tau2 --md."""
import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.integrate import quad, solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fold_birth_v1 as FB  # noqa: E402


def quadrature(rg, R0, Rend):
    """t(Rend) and tau(Rend) for the outer shell released from rest at R0 with charge radius rg (lattice units)."""
    e = math.sqrt(1.0 - rg / R0)

    def x_of(u):
        return R0 - u * u

    def dtau_du(u):
        x = x_of(u)
        return 2.0 * math.sqrt(x * R0 / rg)

    def dt_du(u):
        x = x_of(u)
        b = math.sqrt(rg / x); z = math.sqrt(max(rg / x - rg / R0, 0.0))
        gam = (e * e + b * b) / (e + b * z)
        return gam * 2.0 * math.sqrt(x * R0 / rg)
    U = math.sqrt(R0 - Rend)
    tau, etau = quad(dtau_du, 0.0, U, epsabs=0.0, epsrel=1e-13, limit=400)
    t, et = quad(dt_du, 0.0, U, epsabs=0.0, epsrel=1e-13, limit=400)
    return t, tau, et, etau


def canonical(rg, R0, Rend):
    """P20's independent check: the canonical equations from rest at R0 to R = Rend (N = 1)."""
    def rhs(t, y):
        R, p, tau = y
        gam = math.sqrt(1.0 + p * p); b = math.sqrt(rg / R)
        return [p / gam - b, -b * p / (2.0 * R), 1.0 / gam]
    ev = lambda t, y: y[0] - Rend
    ev.terminal = True; ev.direction = -1
    b0 = math.sqrt(rg / R0)
    p0 = b0 / math.sqrt(1.0 - b0 * b0)      # coordinate rest: dR/dt = p/gamma - b = 0 (Reading A, P20 eq. 2)
    s = solve_ivp(rhs, (0.0, 1e5), [R0, p0, 0.0], method="DOP853", rtol=1e-12, atol=1e-12, events=ev)
    if not s.t_events[0].size:
        return None, None
    return float(s.t_events[0][0]), float(s.y_events[0][0][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--tau4", default=None); ap.add_argument("--tau2", default=None)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    J, R = FB.load_records(a.json)
    g = float(J["args"]["g"]); rs = float(J["args"]["rs"]); E0 = float(J["head"]["E0"]); R0 = float(J["head"]["R0"])
    Rsh0 = R[0].get("R_sh")
    rg0 = 2.0 * g * E0
    cases = [("the production's release (head R0, head E0)", R0, E0, rg0),
             ("the production's T = 0 dust radius R_sh (99 percent of the dust energy inside), head E0", float(Rsh0), E0, rg0) if Rsh0 else None,
             ("P20's control r_g 59.05 at R_0 256 (its table: 13.2248621796 flat-slice, 12.4186017985 proper, in units of 64 a)", 256.0, 59.05 / (2.0 * g), 59.05),
             ("P20's control r_g 64 at R_0 256 (its table: 12.6937264016 flat-slice, 11.8416820247 proper, in units of 64 a)", 256.0, 64.0 / (2.0 * g), 64.0)]
    L = ["## The released ball's outer layer: the flat-slice and proper times to r_g (engine_release_quadrature_v1.py; engine spec 37.3 (3), WEB40F P20 secs. 2, 2.1; record-only)", "",
         "- read from %s: g %g, the nominal r_s %g a, the head's release radius R0 %.6f a and charge E0 %.11f (2 g E0 = %.6f a), the T = 0 record's R_sh %s a." % (
             os.path.basename(a.json), g, rs, R0, E0, rg0, ("%.3f" % Rsh0) if Rsh0 else "-"),
         "- t = int_R^R0 gamma/z dx, tau = int_R^R0 dx/z, gamma = (e^2 + b^2)/(e + b z), e = sqrt(1 - r_g/R0), b = sqrt(r_g/x), z = sqrt(r_g/x - r_g/R0) (P20 eqs. 6, 10), to R = r_g; the canonical-equation check from rest beside (P20's own check: t 846.3911794961956 and tau 794.7905151009893 lattice units at r_g 59.05).", "",
         "| case | R_0 (a) | r_g = 2gE (a) | t (a) | tau (a) | t/64 a (nominal T/rs) | tau/64 a | t/(2 g E0) | tau/(2 g E0) | t/r_g | tau/r_g | canonical t (a) | canonical tau (a) | quad error estimates |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    t_rel = None
    for c in [c for c in cases if c]:
        lab, R0c, Ec, rg = c
        t, tau, et, etau = quadrature(rg, R0c, rg)
        if t_rel is None:
            t_rel = t                      # the first case: the production's release (head R0, head E0)
        tc, tauc = canonical(rg, R0c, rg)
        L.append("| %s | %.6f | %.6f | %.10f | %.10f | %.10f | %.10f | %.10f | %.10f | %.10f | %.10f | %s | %s | %.1e, %.1e |" % (
            lab, R0c, rg, t, tau, t / rs, tau / rs, t / rg0, tau / rg0, t / rg, tau / rg, ("%.10f" % tc) if tc is not None else "-", ("%.10f" % tauc) if tauc is not None else "-", et, etau))
    L += ["", "### Beside: the production's recorded formation and fold clocks (T/rs, the nominal r_s 64 a)", "",
          "| history | tau | first record with a zero (T/rs) | the record before it | m = min(N - abs v) at the two | fold clock (linear) | (quadratic) | in units of 2 g E0: first record, fold clock | the release case's t/64 a minus the first record, minus the fold clock (T/rs) |", "|---|---|---|---|---|---|---|---|---|"]
    for lab, p in (("tau 8 (the production)", a.json), ("tau 4", a.tau4), ("tau 2", a.tau2)):
        if not p:
            continue
        Jx, Rx = FB.load_records(p)
        b = FB.first_birth(Rx)
        f1 = next((r for r in Rx if r.get("trapping_zeros")), None)
        if b is None or f1 is None:
            L.append("| %s | %g | none | - | - | - | - | - | - |" % (lab, Jx["args"]["tau"])); continue
        L.append("| %s | %g | %.5f | %.5f | %+.4f, %+.4f | %.5f | %s | %.6f, %.6f | %+.5f, %+.5f |" % (
            lab, Jx["args"]["tau"], f1["T_over_rs"], b["records"][0], b["m"][0], b["m"][1], b["T_lin"], FB.fmt(b["T_quad"], "%.5f"),
            f1["T"] / rg0, b["T_lin"] * rs / rg0, t_rel / rs - f1["T_over_rs"], t_rel / rs - b["T_lin"]))
    L += ["", "- every number above is printed by the script (the quadrature, the canonical check, and the jsons' records); the unit statements are the columns' definitions, nothing is read or ruled."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
