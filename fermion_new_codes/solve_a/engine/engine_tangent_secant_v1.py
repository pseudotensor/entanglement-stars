#!/usr/bin/env python
"""engine_tangent_secant_v1.py (builder analyst; engine spec 36.4 (1): "P21 (1): two short members at sigma/a 3.6 and 4.4 from the stored
preparation on engine_collapse_v27 ..., tau 8, to T/rs 17.5 ...; then one table: the variance tangent (Q(4.4) - Q(3.6))/(4.4^2 - 3.6^2)
against the secant (Q(8) - Q(4))/48 for the birth event (T_*, R_*), R_t at matched post-formation clocks and E/E0 at matched clocks, with
P21's eq. 33 bounds printed as the return's, no criterion of this lane; it serves item 5's averaging control as a second route (is the scale
dependence a regular variance response) and is read in turn 45").

Reads the four members' run jsons (the tangent pair and the secant pair, each named by its sigma/a; the kernel's per-axis variance
x = sigma^2 = ell^2, no cell convolution in this back-map) and prints:
(1) the members: script (and runner), solver, records, first and last T/rs, the preparation's charge E0 against the sigma/a 4 member's, E/E0
    at T = 0;
(2) per quantity Q the four values, the variance tangent t = (Q(4.4) - Q(3.6))/(4.4^2 - 3.6^2), the secant s = (Q(8) - Q(4))/(8^2 - 4^2),
    s - t, s/t, the two one-sided slopes about variance 16 ((Q(4) - Q(3.6))/(16 - 12.96) and (Q(4.4) - Q(4))/(19.36 - 16)), and the
    quadratic model's curvature Q'' = 2 (s - t)/[(64 + 16) - (19.36 + 12.96)] (Q = c_0 + c_1 x + c_2 x^2 gives s - t = c_2 times that
    bracket; a model, not a bound);
    the quantities: the birth event's clock T_* (linear interpolation of m = min(N - |v|) between the records about its first sign change;
    the quadratic clock beside, engine_fold_birth_v1.crossing_events) and radius R_* (the minimum's radius at T_*), the first episode's
    closing clock where the records hold it, R_t (the outermost zero of the dense N + v) at the matched clocks every --every r_s from
    --rt-from to --rt-to at which all four members hold a zero, and E/E0 at the matched clocks every --every r_s from --e-from;
(3) WEB40F P21 eq. 33, the return's bound and not this lane's criterion: |(Q_8 - Q_4)/(48 a^2) - d| <= (eps_4 + eps_8)/(48 a^2) + eps_d
    + 24 a^2 M_2 with d = Q'(16 a^2); the return supplies no eps and no M_2; printed per quantity: |s - t| and the M_2 at which 24 a^2 M_2
    alone equals it, the finite-difference tangent's own Taylor coefficient C_d in |t - Q'(16)| <= C_d M_2 + (eps_3.6 + eps_4.4)/6.4
    (C_d = [(19.36 - 16)^2 + (16 - 12.96)^2]/(2 x 6.4), the builder's arithmetic printed by the script), and the records' proxies for the eps
    (the interpolation's linear-quadratic difference for T_*, the largest residual over the root slope for R_t, the largest last-iteration
    |dE/E| for E/E0);
(4) the collar's node: for a non-integer ell the last coarse node floor(2048/ell) ell (the collar's node once the sources reach the chain's
    end) lies below the last site 2048; the sigma/a 4 member's stored shell totals beyond it (--fields4) bound the charge it leaves out.
Every number is printed by the script. Args: --member sigma=json (repeated) --tangent 3.6,4.4 --secant 4,8 --every --rt-from --rt-to --e-from
--fields4 --md."""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fold_birth_v1 as FB  # noqa: E402


def rec_residual(r):
    ih = r.get("iteration_history") or []
    if not ih:
        return None
    if not r.get("converged"):
        return (r.get("unconverged") or {}).get("residual")
    h = ih[-1]
    return max(h[1], h[2])


def slope_at_Rt(r):
    d = (r.get("surface_jets") or {}).get("d_N_plus_v")
    return abs(d[0]) if d else None


def last_dE(r):
    ih = r.get("iteration_history") or []
    return ih[-1][0] if ih and ih[-1][0] is not None else None


def ts(Q, xs, tg, sc):
    """tangent, secant, the one-sided slopes and the quadratic model's curvature."""
    a, b = tg; c, d = sc
    x = {s: s * s for s in xs}
    t = (Q[b] - Q[a]) / (x[b] - x[a])
    s = (Q[d] - Q[c]) / (x[d] - x[c])
    left = (Q[c] - Q[a]) / (x[c] - x[a]) if c in Q else None
    right = (Q[b] - Q[c]) / (x[b] - x[c]) if c in Q else None
    br = (x[d] + x[c]) - (x[b] + x[a])
    return {"t": t, "s": s, "d": s - t, "ratio": (s / t) if t != 0 else None, "left": left, "right": right, "Qpp": 2.0 * (s - t) / br, "bracket": br}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="sigma=json")
    ap.add_argument("--tangent", default="3.6,4.4"); ap.add_argument("--secant", default="4,8")
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--rt-from", type=float, default=13.0); ap.add_argument("--rt-to", type=float, default=17.5)
    ap.add_argument("--e-from", type=float, default=0.5)
    ap.add_argument("--fields4", default=None, help="the sigma/a 4 member's fields npz (the collar's node check)")
    ap.add_argument("--nsites", type=int, default=2048)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    tg = tuple(float(x) for x in a.tangent.split(",")); sc = tuple(float(x) for x in a.secant.split(","))
    M = {}
    for spec in a.member:
        k, p = spec.split("=", 1)
        J, R = FB.load_records(p)
        M[float(k)] = {"path": p, "J": J, "R": R, "byT": {round(r["T_over_rs"], 6): r for r in R}}
    sig = sorted(M)
    xs = sig
    L = ["## The variance tangent against the secant on the fixed-preparation ladder (engine_tangent_secant_v1.py; engine spec 36.4 (1), WEB40F P21 eqs. 20, 23, 33)", "",
         "- the kernel's per-axis variance x = sigma^2 = ell^2 (no cell convolution in this back-map); the tangent t = (Q(%g) - Q(%g))/(%g^2 - %g^2) = (Q(%g) - Q(%g))/%g, the secant s = (Q(%g) - Q(%g))/(%g^2 - %g^2) = (Q(%g) - Q(%g))/%g; s/t = 1 is a response linear in the variance over [%g, %g]; the one-sided slopes about x = %g and the quadratic model's Q'' = 2 (s - t)/%g are printed beside (a model, not a bound); no criterion of this lane." % (
             tg[1], tg[0], tg[1], tg[0], tg[1], tg[0], tg[1] ** 2 - tg[0] ** 2, sc[1], sc[0], sc[1], sc[0], sc[1], sc[0], sc[1] ** 2 - sc[0] ** 2, tg[0] ** 2, sc[1] ** 2, sc[0] ** 2,
             (sc[1] ** 2 + sc[0] ** 2) - (tg[1] ** 2 + tg[0] ** 2)),
         "- definitions: m = min(N - |v|) of the dense fields per record; T_* its first sign change, interpolated linearly between the two records (the quadratic clock beside), R_* the minimum's radius at T_*; the closing clock likewise at the first episode's end; R_t the outermost zero of the dense N + v (a geometric root, not the applied-field root); E/E0 the record's charge over its own preparation's E0 (printed per member: one preparation).", ""]
    L += ["| sigma/a | run | script (runner) | solver | records | first T/rs | last T/rs | E0 | E0 - E0(sigma/a 4) | E/E0 at T = 0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    E04 = M[sc[0]]["J"]["head"]["E0"] if sc[0] in M else None
    for s in sig:
        J = M[s]["J"]; R = M[s]["R"]
        L.append("| %g | %s | %s%s | %s | %d | %.3f | %.3f | %.11f | %s | %.7f |" % (
            s, os.path.basename(M[s]["path"]), J.get("script"), (" (" + J["runner"].split(":")[0] + ")") if J.get("runner") else "", (J.get("args") or {}).get("mix", "plain"),
            len(R), R[0]["T_over_rs"] if R else float("nan"), R[-1]["T_over_rs"] if R else float("nan"), J["head"]["E0"],
            ("%+.3e" % (J["head"]["E0"] - E04)) if E04 is not None else "-", R[0]["E_over_E0"] if R else float("nan")))
    have = all(s in M for s in tg + sc)
    if not have:
        L.append(""); L.append("- a tangent or secant member is missing; nothing more is printed.")
        open(a.md, "w").write("\n".join(L) + "\n"); print("\n".join(L)); return
    # (2a) the birth and closing events
    ev = {s: FB.crossing_events(M[s]["R"]) for s in sig}
    birth = {s: next((e for e in ev[s] if e["kind"] == "birth"), None) for s in sig}
    close = {s: (next((e for e in ev[s] if e["kind"] == "closing" and birth[s] and e["T_lin"] > birth[s]["T_lin"]), None)) for s in sig}
    L += ["", "### The events (per member)", "", "| sigma/a | birth between T/rs | m there | T_* (linear) | T_* (quadratic) | R_* | first closing T/rs (linear) | (quadratic) | its radius |", "|---|---|---|---|---|---|---|---|---|"]
    for s in sig:
        b = birth[s]; c = close[s]
        L.append("| %g | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s, ("%.3f, %.3f" % b["records"]) if b else "none to %.3f" % M[s]["R"][-1]["T_over_rs"], ("%+.4f, %+.4f" % b["m"]) if b else "-",
            ("%.4f" % b["T_lin"]) if b else "-", FB.fmt(b["T_quad"], "%.4f") if b else "-", FB.fmt(b["R_lin"], "%.3f") if b else "-",
            ("%.4f" % c["T_lin"]) if c else ("none to %.3f" % M[s]["R"][-1]["T_over_rs"]), FB.fmt(c["T_quad"], "%.4f") if c else "-", FB.fmt(c["R_lin"], "%.3f") if c else "-"))
    Cd = ((tg[1] ** 2 - 16.0) ** 2 + (16.0 - tg[0] ** 2) ** 2) / (2.0 * (tg[1] ** 2 - tg[0] ** 2)) if tg[0] < 4 < tg[1] else None
    L += ["", "### The tangent against the secant", "",
          "- WEB40F P21 eq. 33, the return's (not this lane's criterion): |(Q_8 - Q_4)/(48 a^2) - d| <= (eps_4 + eps_8)/(48 a^2) + eps_d + 24 a^2 M_2, d = Q'(16 a^2), M_2 a bound on |Q''| over [16, 64] a^2; the return supplies no eps and no M_2. Printed per row: \"M_2 for 24 M_2 = |s - t|\" (the M_2 at which that term alone equals the difference) and the records' eps proxies; the finite-difference tangent's own Taylor coefficient C_d in |t - Q'(16)| <= C_d M_2 + (eps_%g + eps_%g)/%g is %s (the builder's arithmetic, printed by the script)." % (
              tg[0], tg[1], tg[1] ** 2 - tg[0] ** 2, ("%.4f" % Cd) if Cd is not None else "-"), "",
          "| Q | Q(%g) | Q(%g) | Q(%g) | Q(%g) | tangent t | secant s | s - t | s/t | slope left of 16 | slope right of 16 | Q'' (quadratic model) | M_2 for 24 M_2 = abs(s - t) | eps proxy (largest of the four) |" % tuple(sorted(tg + sc)),
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = sorted(tg + sc)

    def row(name, Q, eps):
        if any(Q.get(s) is None for s in order):
            L.append("| %s | %s | - | - | - | - | - | - | - | - | %s |" % (name, " | ".join(FB.fmt(Q.get(s), "%.6f") for s in order), "a member holds no value"))
            return None
        r = ts(Q, order, tg, sc)
        L.append("| %s | %s | %+.6e | %+.6e | %+.4e | %s | %s | %s | %+.4e | %.4e | %s |" % (
            name, " | ".join("%.6f" % Q[s] for s in order), r["t"], r["s"], r["d"], FB.fmt(r["ratio"], "%.4f"), FB.fmt(r["left"], "%+.4e"), FB.fmt(r["right"], "%+.4e"),
            r["Qpp"], abs(r["d"]) / 24.0, FB.fmt(eps, "%.2e")))
        return r
    out = {}
    Tq = {s: birth[s]["T_lin"] if birth[s] else None for s in sig}
    eT = max((abs(birth[s]["T_lin"] - birth[s]["T_quad"]) for s in sig if birth[s] and birth[s]["T_quad"] is not None), default=None)
    out["T_*"] = row("T_* (T/rs)", Tq, eT)
    out["R_*"] = row("R_* (a)", {s: birth[s]["R_lin"] if birth[s] else None for s in sig}, None)
    if all(close[s] for s in sig):
        eC = max((abs(close[s]["T_lin"] - close[s]["T_quad"]) for s in sig if close[s]["T_quad"] is not None), default=None)
        out["closing"] = row("first closing (T/rs)", {s: close[s]["T_lin"] for s in sig}, eC)
    else:
        L.append("| first closing (T/rs) | %s | - | - | - | - | - | - | - | - | %s |" % (" | ".join(("%.4f" % close[s]["T_lin"]) if close[s] else "-" for s in order), "not every member holds it in its records"))
    # (2b) R_t at matched clocks
    common = set(M[sig[0]]["byT"])
    for s in sig[1:]:
        common &= set(M[s]["byT"])
    clocks = sorted(t for t in common if abs(t / a.every - round(t / a.every)) < 1e-9)
    nrt = 0
    for t in clocks:
        if t < a.rt_from - 1e-9 or t > a.rt_to + 1e-9:
            continue
        rs_ = {s: M[s]["byT"][t] for s in sig}
        zs = {s: FB.zeros_of(rs_[s]) for s in sig}
        Q = {s: (zs[s][-1] if zs[s] else None) for s in sig}
        eps = [rec_residual(rs_[s]) / slope_at_Rt(rs_[s]) for s in sig if zs[s] and rec_residual(rs_[s]) is not None and slope_at_Rt(rs_[s])]
        nz = ",".join(str(len(zs[s])) for s in order)
        rr = row("R_t at T/rs %.3f (zeros %s)" % (t, nz), Q, max(eps) if eps else None)
        if rr:
            out["R_t %.3f" % t] = rr; nrt += 1
    ne = 0
    for t in clocks:
        if t < a.e_from - 1e-9:
            continue
        rs_ = {s: M[s]["byT"][t] for s in sig}
        eps = [abs(last_dE(rs_[s])) for s in sig if last_dE(rs_[s]) is not None]
        rr = row("E/E0 at T/rs %.3f" % t, {s: rs_[s]["E_over_E0"] for s in sig}, max(eps) if eps else None)
        if rr:
            out["E/E0 %.3f" % t] = rr; ne += 1
    L += ["", "- rows: %d R_t clocks (every %g r_s from %g to %g with a zero in all four members), %d E/E0 clocks (every %g r_s from %g); the clocks are those present in all four members' records (the shortest member's last record %.3f)." % (
        nrt, a.every, a.rt_from, a.rt_to, ne, a.every, a.e_from, min(M[s]["R"][-1]["T_over_rs"] for s in sig))]
    rows_rt = [v for k_, v in out.items() if k_.startswith("R_t") and v]
    rows_e = [v for k_, v in out.items() if k_.startswith("E/E0") and v]
    if rows_rt:
        rr = [v["ratio"] for v in rows_rt if v["ratio"] is not None]
        L.append("- s/t over the R_t clocks: %.4f to %.4f (median %.4f); over the E/E0 clocks: %s." % (min(rr), max(rr), float(np.median(rr)),
                 ("%.4f to %.4f (median %.4f)" % (min(v["ratio"] for v in rows_e if v["ratio"] is not None), max(v["ratio"] for v in rows_e if v["ratio"] is not None),
                                                   float(np.median([v["ratio"] for v in rows_e if v["ratio"] is not None])))) if rows_e else "-"))
    for k_ in ("T_*", "R_*", "closing"):
        v = out.get(k_)
        if v:
            L.append("- %s: t %+.6e, s %+.6e, s/t %s." % (k_, v["t"], v["s"], FB.fmt(v["ratio"], "%.4f")))
    # (4) the collar's node
    if a.fields4:
        Z = np.load(a.fields4, allow_pickle=True)
        Rs = np.asarray(Z["R"], float); TT = np.asarray(Z["T"], float); e = np.asarray(Z["eps_tot"], float)
        rs = float(M[sc[0]]["J"]["args"]["rs"])
        sel = TT / rs <= a.rt_to + 1e-9
        for s in tg:
            ln = math.floor(a.nsites / s) * s
            beyond = e[sel][:, Rs > ln].sum(axis=1); tot = e[sel].sum(axis=1)
            q = np.abs(beyond) / np.maximum(np.abs(tot), 1e-300)
            i = int(np.argmax(q))
            L.append("- the collar's node at sigma/a %g: the last coarse node %.6g against the last site %d; the sigma/a 4 member's stored shell totals beyond it are at most %.3e of its shell total over its records to T/rs %.3f (at T/rs %.3f): the charge the node leaves out, measured on the sigma/a 4 history." % (
                s, ln, a.nsites, float(q[i]), float(TT[sel][-1] / rs), float(TT[sel][i] / rs)))
    L.append("- runs read: " + "; ".join("sigma/a %g %s" % (s, M[s]["path"]) for s in sig))
    L.append("- every number above is printed by the script from the members' json records (and the sigma/a 4 member's fields npz); the members are read as they stand (a member still running is read to its last record).")
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
