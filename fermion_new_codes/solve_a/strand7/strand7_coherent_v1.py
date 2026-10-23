#!/usr/bin/env python
"""Strand 7 S7a part 2 (c): the coherent thin-shell integration (SPEC_strand7_collapse.md sec. 10.2(c); builder analyst).  One shell of total
marker weight M = E_0 (m = 1, j = 0) moving under H = N W with the thin-shell fields at its radius s: psi_s = 1 + x, chi_s = 1 - b x, x = A/s,
and the midpoint derivatives psi'_s = -A/(2 s^2), chi'_s = B/(2 s^2) (B = b A); the closure b = N_s S, N_s = 1/(1 + x (1 + S)),
S = 1 + 2 psi_s^-4 p^2/W^2, and the clock charge from the shell's energy, A = M W/(4 pi k psi_s) with k = 1/(2 pi), solved at every instant
(a scalar root in A; W = sqrt(1 + psi_s^-4 p^2) depends on A through psi_s).  Hamilton: s_dot = N psi^-4 p/W, p_dot = -[N' W + N (psi^-4)' p^2/(2W)],
N = chi/psi, N' = (chi' psi - chi psi')/psi^2, (psi^-4)' = -4 psi^-5 psi'.  From rest at the areal radius R_0 = s (1 + A/s)^2 = 4 r_s0 (r_s0 = 2 E_0).
DOP853 (rtol 1e-12, atol 1e-12 in units of r_s0), dense output to T = 60 r_s0 or until x exceeds 50.  Reported: A(T)/A(0) - 1 (the theorem's
constancy: the integration's control), the throat crossing s = A (T, c_+ = N psi^-2, N, B/A) against the markers' (the S7a JSONs given), the
values at T = 20, 30, 40, 50, 60 r_s0 (s/A, B/A, N_s, R/4A, ds/dT r_s0/s) beside the markers' r_sh/A, B/A, N(r_sh), R_sh/4A and their rate
d ln r_sh/dT from consecutive outputs, and the late rate against -s^3/(4 A^3) per r_s0 (the lead's law, as d ln s/dT r_s0 = -(s/A)^2 r_s0/(4A))."""
import argparse
import json
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

K4PI = 2.0  # 4 pi k with k = 1/(2 pi)


def clock_charge(s, p, M):
    """A with A = M W/(4 pi k psi_s), psi_s = 1 + A/s, W = sqrt(1 + psi_s^-4 p^2): the root of f(A) = A - M W/(K4PI psi_s)."""
    def f(A):
        ps = 1 + A / s
        W = math.sqrt(1 + ps ** -4 * p * p)
        return A - M * W / (K4PI * ps)
    hi = M * math.sqrt(1 + p * p) / K4PI + 1.0
    return brentq(f, 1e-14, hi, xtol=1e-15, rtol=1e-15, maxiter=200)


def state(s, p, M):
    A = clock_charge(s, p, M)
    x = A / s
    ps = 1 + x
    W = math.sqrt(1 + ps ** -4 * p * p)
    u2 = ps ** -4 * p * p / (W * W)
    S = 1 + 2 * u2
    Ns = 1 / (1 + x * (1 + S))
    b = Ns * S
    B = b * A
    ch = 1 - B / s
    return {"A": A, "x": x, "psi": ps, "chi": ch, "N": ch / ps, "Ns_closure": Ns, "W": W, "u2": u2, "S": S, "B": B}


def rhs(T, y, M):
    s, p = y
    st = state(s, p, M)
    A, B, ps, ch, W = st["A"], st["B"], st["psi"], st["chi"], st["W"]
    dps = -A / (2 * s * s)
    dch = B / (2 * s * s)
    N = ch / ps
    dN = (dch * ps - ch * dps) / ps ** 2
    q = ps ** -4
    dq = -4 * ps ** -5 * dps
    return [N * q * p / W, -(dN * W + N * dq * p * p / (2 * W))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E0", type=float, default=16.0)
    ap.add_argument("--R0-over-rs", type=float, default=4.0)
    ap.add_argument("--T-over-rs", type=float, default=60.0)
    ap.add_argument("--markers", default="", help="comma list of S7a JSONs of the same E_0 (the self-consistent runs) to compare")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    M, rs0 = a.E0, 2 * a.E0
    R0 = a.R0_over_rs * rs0

    def g(s):
        A = clock_charge(s, 0.0, M)
        return s * (1 + A / s) ** 2 - R0
    s0 = brentq(g, 0.5 * R0 / 4, R0, xtol=1e-14, rtol=1e-15)
    st0 = state(s0, 0.0, M)
    A0 = st0["A"]

    def ev_throat(T, y, M):
        return y[0] - clock_charge(y[0], y[1], M)
    ev_throat.terminal = False
    ev_throat.direction = -1

    def ev_deep(T, y, M):
        return clock_charge(y[0], y[1], M) / y[0] - 50.0
    ev_deep.terminal = True
    sol = solve_ivp(rhs, (0.0, a.T_over_rs * rs0), [s0, 0.0], method="DOP853", rtol=1e-12, atol=1e-12 * rs0, dense_output=True,
                    events=[ev_throat, ev_deep], args=(M,))
    Tend = float(sol.t[-1])
    Tg = np.arange(0.0, Tend + 1e-9, 0.1 * rs0)
    rows = []
    for T in Tg:
        s, p = sol.sol(T)
        st = state(s, p, M)
        rows.append({"T_over_rs0": T / rs0, "s": s, "p": p, "A": st["A"], "A_rel": st["A"] / A0 - 1, "x": st["x"], "B_over_A": st["B"] / st["A"],
                     "N": st["N"], "Ns_closure_minus_N": st["Ns_closure"] - st["N"], "c_plus": st["N"] / st["psi"] ** 2, "u2": st["u2"],
                     "R_over_4A": s * st["psi"] ** 2 / (4 * st["A"])})
    for i in range(1, len(rows) - 1):
        rows[i]["dlns_dT_rs0"] = (math.log(rows[i + 1]["s"]) - math.log(rows[i - 1]["s"])) / (rows[i + 1]["T_over_rs0"] - rows[i - 1]["T_over_rs0"])
        rows[i]["lead_law_dlns_dT_rs0"] = -(rows[i]["s"] / rows[i]["A"]) ** 2 * rs0 / (4 * rows[i]["A"])
    cross = None
    if len(sol.t_events[0]):
        Tc = float(sol.t_events[0][0])
        s, p = sol.sol(Tc)
        st = state(s, p, M)
        cross = {"T_over_rs0": Tc / rs0, "c_plus": st["N"] / st["psi"] ** 2, "N": st["N"], "B_over_A": st["B"] / st["A"], "A_rel": st["A"] / A0 - 1,
                 "R_over_rs0": s * st["psi"] ** 2 / rs0, "u2": st["u2"]}
    mk = {}
    for pth in [x for x in a.markers.split(",") if x.strip()]:
        R = json.load(open(pth))
        recs = R["run"]["records"]
        tag = R["meta"]["tag"]
        Tm = np.array([q["T_over_rs"] for q in recs])
        rsh = np.array([q["r_sh"] for q in recs])
        mk[tag] = {"throat_crossing": R["run"].get("throat_crossing"), "at": {}}
        for Tq in (20, 30, 40, 50, 60):
            k = np.nonzero(np.abs(Tm - Tq) < 1e-6)[0]
            if not len(k):
                continue
            k = int(k[0])
            rate = None
            if 0 < k < len(recs) - 1:
                rate = (math.log(rsh[k + 1]) - math.log(rsh[k - 1])) / (Tm[k + 1] - Tm[k - 1])
            q = recs[k]
            mk[tag]["at"][Tq] = {"r_sh_over_A": q["r_sh_over_A"], "B_over_A": q["B_over_A"], "N_sh": q["N_sh"], "R_sh_over_4A": q["R_sh"] / (4 * q["A"]),
                                 "A_rel": q["A_rel"], "dln_rsh_dT_rs0": rate}
    out = {"meta": {"E0": M, "r_s0": rs0, "R0": R0, "s0": s0, "A0": A0, "A0_over_rs0": A0 / rs0, "T_end_over_rs0": Tend / rs0, "status": sol.message,
                    "nfev": int(sol.nfev), "script": "strand7_coherent_v1.py", "builder": "analyst"},
           "throat_crossing": cross, "max_abs_A_rel": float(max(abs(r["A_rel"]) for r in rows)), "rows": rows, "markers": mk}
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)

    def f(x, d=5):
        return "-" if x is None else ("%." + str(d) + "g") % x
    L = ["# Strand 7 S7a part 2 (c): the coherent thin-shell integration at E_0 = %g (strand7_coherent_v1.py)" % M, "",
         "- Start: s0 = %s (s0/A0 = %s), A0 = %s = %s r_s0; integration to T = %s r_s0 (%s); the control: max |A(T)/A(0) - 1| = %s." % (
             f(s0, 10), f(s0 / A0, 8), f(A0, 10), f(A0 / rs0, 8), f(Tend / rs0, 6), sol.message, f(out["max_abs_A_rel"], 3)), ""]
    if cross:
        L.append("- The throat crossing s = A: T = %s r_s0, c_+ = %s, N = %s, B/A = %s, A/A(0) - 1 = %s, R/r_s0 = %s, u^2 = %s." % (
            f(cross["T_over_rs0"], 6), f(cross["c_plus"]), f(cross["N"]), f(cross["B_over_A"]), f(cross["A_rel"], 3), f(cross["R_over_rs0"]), f(cross["u2"])))
    for tag, m in mk.items():
        tc = m["throat_crossing"]
        if tc:
            L.append("- The markers (%s), table TC: T = %s r_s0, c_+ = %s, N = %s, B/A = %s." % (tag, f(tc["T_over_rs"], 6), f(tc["c_plus_r"]), f(tc["N_sh"]),
                                                                                       f(tc["B_over_A"])))
    L += ["", "| T/r_s0 | s/A | B/A | N_s | R/4A | d ln s/dT r_s0 | lead's law -(s/A)^2 r_s0/(4A) | markers: r_sh/A, B/A, N(r_sh), R_sh/4A, d ln r_sh/dT r_s0 |",
          "|---|---|---|---|---|---|---|---|"]
    for Tq in (20, 30, 40, 50, 60):
        k = [r for r in rows if abs(r["T_over_rs0"] - Tq) < 1e-6]
        if not k:
            continue
        r = k[0]
        ms = "; ".join("%s: %s, %s, %s, %s, %s" % (tag, f(m["at"][Tq]["r_sh_over_A"]), f(m["at"][Tq]["B_over_A"]), f(m["at"][Tq]["N_sh"]),
                                                     f(m["at"][Tq]["R_sh_over_4A"]), f(m["at"][Tq]["dln_rsh_dT_rs0"]))
                       for tag, m in mk.items() if Tq in m["at"])
        L.append("| %g | %s | %s | %s | %s | %s | %s | %s |" % (Tq, f(1 / r["x"]), f(r["B_over_A"]), f(r["N"]), f(r["R_over_4A"]), f(r.get("dlns_dT_rs0")),
                                                              f(r.get("lead_law_dlns_dT_rs0")), ms))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
