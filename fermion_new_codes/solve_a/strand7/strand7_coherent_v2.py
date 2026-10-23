#!/usr/bin/env python
"""Strand 7 S7a part 2 (c), version 2: the coherent thin-shell benchmark (SPEC_strand7_collapse.md secs. 10.2(c) and 11.4; builder analyst).
One shell of total marker weight M = E_0 (m = 1, j = 0) moving under H = N W with the thin-shell fields at its radius s: psi_s = 1 + x,
chi_s = 1 - b x, x = A/s, and the midpoint derivatives psi'_s = -A/(2 s^2), chi'_s = B/(2 s^2) (B = b A); the closure b = N_s S,
N_s = 1/(1 + x (1 + S)), S = 1 + 2 psi_s^-4 p^2/W^2, and the clock charge from the clock row's jump, A = M W/(4 pi k psi_s) with k = 1/(2 pi),
solved at every instant (a scalar root in A; W = sqrt(1 + psi_s^-4 p^2) depends on A through psi_s), so A's constancy is the integration's
control, not an input.  Hamilton: s_dot = N psi^-4 p/W, p_dot = -[N' W + N (psi^-4)' p^2/(2W)], N = chi/psi, N' = (chi' psi - chi psi')/psi^2,
(psi^-4)' = -4 psi^-5 psi'.  DOP853 (rtol 1e-12, atol 1e-12 in units of r_s0), dense output to T = 60 r_s0 or until x exceeds 50.
Version 1: from rest at the areal radius R_0 = s (1 + A/s)^2 = 4 r_s0 (r_s0 = 2 E_0), E_0 = 16, the throat crossing s = A and the values at
T = 20, ..., 60 r_s0 against the markers.
Version 2 (sec. 11.4): (a) the dilation moment D_r = M s p and the virial check, (A - B)/G against -dD_r/dT along the integration (G = 1;
dD_r/dT = M (s_dot p + s p_dot) from Hamilton's equations; the residual (A - B + dD_r/dT)/A per row and its largest value over the
integration's 0.1 r_s0 grid); (b) the release-from-rest relation u^2 = 1 - ((1 + x_0)/(1 + x))^2 against the integration's u^2 per row;
(c) the finite-x table (x, u^2, N_s, B/A, B/s, R/4A, the rate d ln s/dT r_s0) beside the asymptotes N_s ~ 1/(4x), B/A ~ 3/(4x), B/s -> 3/4,
d ln s/dT r_s0 ~ -(s/A)^2 r_s0/(4A); (d) --E0 takes a comma list (8, 16, 32), each started at rest from its own run's x_0 at T = 0 (--x0-from:
the S7a JSONs, matched by the meta E0; x_0 = 1/(r_sh/A) of the first record, at T = 0) or, for an E_0 without a run, from R_0 = 4 r_s0 as in
version 1; the passage s = A and the table per E_0 against that E_0's markers (--markers, matched by E0)."""
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


def f(x, d=5):
    return "-" if x is None else ("%." + str(d) + "g") % x


def run_one(M, x0_inv, T_over_rs, markers):
    rs0 = 2 * M
    if x0_inv is None:
        R0 = 4.0 * rs0

        def g(s):
            A = clock_charge(s, 0.0, M)
            return s * (1 + A / s) ** 2 - R0
        s0 = brentq(g, 0.5 * R0 / 4, R0, xtol=1e-14, rtol=1e-15)
        start = "from rest at the areal radius R_0 = 4 r_s0 (version 1's start)"
    else:
        def g(s):
            return s / clock_charge(s, 0.0, M) - x0_inv
        s0 = brentq(g, 1e-3 * rs0, 100 * rs0, xtol=1e-14, rtol=1e-15)
        start = "from rest at its run's s/A = %s at T = 0 (the first record's r_sh/A)" % f(x0_inv, 8)
    st0 = state(s0, 0.0, M)
    A0 = st0["A"]
    x0 = A0 / s0

    def ev_throat(T, y, M):
        return y[0] - clock_charge(y[0], y[1], M)
    ev_throat.terminal = False
    ev_throat.direction = -1

    def ev_deep(T, y, M):
        return clock_charge(y[0], y[1], M) / y[0] - 50.0
    ev_deep.terminal = True
    sol = solve_ivp(rhs, (0.0, T_over_rs * rs0), [s0, 0.0], method="DOP853", rtol=1e-12, atol=1e-12 * rs0, dense_output=True,
                    events=[ev_throat, ev_deep], args=(M,))
    Tend = float(sol.t[-1])
    Tg = np.arange(0.0, Tend + 1e-9, 0.1 * rs0)
    rows = []
    vir_max = 0.0
    for T in Tg:
        s, p = sol.sol(T)
        st = state(s, p, M)
        sd, pd = rhs(T, [s, p], M)
        Dr = M * s * p
        dDr = M * (sd * p + s * pd)
        vir = (st["A"] - st["B"] + dDr) / st["A"]
        vir_max = max(vir_max, abs(vir))
        x = st["x"]
        rows.append({"T_over_rs0": T / rs0, "s": s, "p": p, "A": st["A"], "A_rel": st["A"] / A0 - 1, "x": x, "B_over_A": st["B"] / st["A"],
                     "B_over_s": st["B"] / s, "N": st["N"], "Ns_closure_minus_N": st["Ns_closure"] - st["N"], "c_plus": st["N"] / st["psi"] ** 2,
                     "u2": st["u2"], "u2_release_relation": 1 - ((1 + x0) / (1 + x)) ** 2, "R_over_4A": s * st["psi"] ** 2 / (4 * st["A"]),
                     "D_r": Dr, "minus_dDr_dT": -dDr, "A_minus_B": st["A"] - st["B"], "virial_residual_over_A": vir,
                     "asym_Ns": 1 / (4 * x), "asym_B_over_A": 3 / (4 * x), "asym_B_over_s": 0.75,
                     "asym_dlns_dT_rs0": -(s / st["A"]) ** 2 * rs0 / (4 * st["A"])})
    for i in range(1, len(rows) - 1):
        rows[i]["dlns_dT_rs0"] = (math.log(rows[i + 1]["s"]) - math.log(rows[i - 1]["s"])) / (rows[i + 1]["T_over_rs0"] - rows[i - 1]["T_over_rs0"])
    cross = None
    if len(sol.t_events[0]):
        Tc = float(sol.t_events[0][0])
        s, p = sol.sol(Tc)
        st = state(s, p, M)
        sd, pd = rhs(Tc, [s, p], M)
        cross = {"T_over_rs0": Tc / rs0, "c_plus": st["N"] / st["psi"] ** 2, "N": st["N"], "B_over_A": st["B"] / st["A"], "A_rel": st["A"] / A0 - 1,
                 "R_over_rs0": s * st["psi"] ** 2 / rs0, "u2": st["u2"], "u2_release_relation": 1 - ((1 + x0) / 2) ** 2,
                 "D_r": M * s * p, "virial_residual_over_A": (st["A"] - st["B"] + M * (sd * p + s * pd)) / st["A"]}
    mk = {}
    for R in markers:
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
                                 "B_over_rsh": q["B"] / q["r_sh"], "u2": q.get("u2"), "A_rel": q["A_rel"], "dln_rsh_dT_rs0": rate}
    return {"meta": {"E0": M, "r_s0": rs0, "start": start, "s0": s0, "A0": A0, "x0": x0, "A0_over_rs0": A0 / rs0, "T_end_over_rs0": Tend / rs0,
                     "status": sol.message, "nfev": int(sol.nfev)},
            "throat_crossing": cross, "max_abs_A_rel": float(max(abs(r["A_rel"]) for r in rows)), "max_abs_virial_residual_over_A": vir_max,
            "max_abs_u2_minus_release_relation": float(max(abs(r["u2"] - r["u2_release_relation"]) for r in rows)), "rows": rows, "markers": mk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E0", default="8,16,32", help="comma list of E_0")
    ap.add_argument("--T-over-rs", type=float, default=60.0)
    ap.add_argument("--x0-from", default="", help="comma list of S7a JSONs whose first record's r_sh/A starts the same E_0")
    ap.add_argument("--markers", default="", help="comma list of S7a JSONs (the self-consistent runs) to compare, matched by E_0")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    Es = [float(x) for x in a.E0.split(",") if x.strip()]
    x0src, mks = {}, {}
    for pth in [x for x in a.x0_from.split(",") if x.strip()]:
        R = json.load(open(pth))
        r0 = R["run"]["records"][0]
        if abs(r0["T_over_rs"]) > 1e-9:
            raise SystemExit("%s: the first record is not at T = 0 (T = %s)" % (pth, r0["T_over_rs"]))
        x0src[float(R["meta"]["E0"])] = (float(r0["r_sh_over_A"]), pth)
    for pth in [x for x in a.markers.split(",") if x.strip()]:
        R = json.load(open(pth))
        mks.setdefault(float(R["meta"]["E0"]), []).append(R)
    out = {"meta": {"script": "strand7_coherent_v2.py", "builder": "analyst", "x0_from": {str(k): v[1] for k, v in x0src.items()}}, "runs": {}}
    L = ["# Strand 7 S7a part 2 (c), version 2: the coherent thin-shell benchmark at E_0 = %s (strand7_coherent_v2.py)" % ", ".join("%g" % E for E in Es), ""]
    for E in Es:
        r = run_one(E, x0src[E][0] if E in x0src else None, a.T_over_rs, mks.get(E, []))
        out["runs"]["E%g" % E] = r
        m = r["meta"]
        L += ["## E_0 = %g" % E, "",
              "- Start: %s: s0 = %s, A0 = %s = %s r_s0, x_0 = %s; integration to T = %s r_s0 (%s)." % (
                  m["start"], f(m["s0"], 10), f(m["A0"], 10), f(m["A0_over_rs0"], 8), f(m["x0"], 8), f(m["T_end_over_rs0"], 6), m["status"]),
              "- Controls: max |A(T)/A(0) - 1| = %s (the clock jump solved at every instant); max |(A - B + dD_r/dT)/A| = %s (the virial identity "
              "(A - B)/G = -dD_r/dT, D_r = M s p, G = 1); max |u^2 - (1 - ((1 + x_0)/(1 + x))^2)| = %s (the release-from-rest relation)." % (
                  f(r["max_abs_A_rel"], 3), f(r["max_abs_virial_residual_over_A"], 3), f(r["max_abs_u2_minus_release_relation"], 3))]
        c = r["throat_crossing"]
        if c:
            L.append("- The passage s = A: T = %s r_s0, c_+ = %s, N = %s, B/A = %s, u^2 = %s (the relation: %s), A/A(0) - 1 = %s, R/r_s0 = %s, "
                     "D_r = %s, virial residual %s." % (f(c["T_over_rs0"], 6), f(c["c_plus"]), f(c["N"]), f(c["B_over_A"]), f(c["u2"]),
                                                       f(c["u2_release_relation"]), f(c["A_rel"], 3), f(c["R_over_rs0"]), f(c["D_r"]),
                                                       f(c["virial_residual_over_A"], 3)))
        for tag, mm in r["markers"].items():
            tc = mm["throat_crossing"]
            if tc:
                L.append("- The markers (%s), table TC: T = %s r_s0, c_+ = %s, N = %s, B/A = %s." % (
                    tag, f(tc["T_over_rs"], 6), f(tc["c_plus_r"]), f(tc["N_sh"]), f(tc["B_over_A"])))
        L += ["", "| T/r_s0 | x | u^2 (relation) | N_s (1/(4x)) | B/A (3/(4x)) | B/s (3/4) | R/4A | d ln s/dT r_s0 (-(s/A)^2 r_s0/(4A)) | -dD_r/dT against A - B |"
              " markers: r_sh/A, B/A, N(r_sh), B/r_sh, R_sh/4A, d ln r_sh/dT r_s0 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for Tq in (20, 27, 30, 40, 50, 60):
            k = [q for q in r["rows"] if abs(q["T_over_rs0"] - Tq) < 1e-6]
            if not k:
                continue
            q = k[0]
            ms = "; ".join("%s: %s, %s, %s, %s, %s, %s" % (tag, f(mm["at"][Tq]["r_sh_over_A"]), f(mm["at"][Tq]["B_over_A"]), f(mm["at"][Tq]["N_sh"]),
                                                         f(mm["at"][Tq]["B_over_rsh"]), f(mm["at"][Tq]["R_sh_over_4A"]), f(mm["at"][Tq]["dln_rsh_dT_rs0"]))
                           for tag, mm in r["markers"].items() if Tq in mm["at"])
            L.append("| %g | %s | %s (%s) | %s (%s) | %s (%s) | %s (0.75) | %s | %s (%s) | %s against %s | %s |" % (
                Tq, f(q["x"]), f(q["u2"]), f(q["u2_release_relation"]), f(q["N"]), f(q["asym_Ns"]), f(q["B_over_A"]), f(q["asym_B_over_A"]),
                f(q["B_over_s"]), f(q["R_over_4A"]), f(q.get("dlns_dT_rs0")), f(q["asym_dlns_dT_rs0"]), f(q["minus_dDr_dT"]), f(q["A_minus_B"]), ms))
        L.append("")
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
