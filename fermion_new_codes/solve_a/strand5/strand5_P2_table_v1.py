#!/usr/bin/env python
"""Strand 5 production, job P2 (SPEC_strand5_covariant_row.md sec. 3; builder analyst): the family table on the covariant two-derivative row
from the existing records, with the dilute-limit control.  No solve.

Inputs: J1's record (strand5_star_v4.py: star_v4.json; the reference family T1a, 60 members, and the grid family T1b at n = 8000; G = m = 1,
g = 16) and the (0.01, 0.1) local_ka0 families of J4run4 at couplings (0, 0) and (0.2, 1) (strand5_lattice_star_v3.py; lattice units, the
family's own G = 1/(2 pi k_op), m = m_light a = 0.01).
Per member: z_c = k_F(0)/m, M, N_B, E_b = M - m N_B, E_b/M, 2GM/R_s (areal R_s), GM/R_s, Q_b = E_b/(M GM/R_s), mu_inf/m, N(0), the first law's
residual, A/B - 1, D/E_bdy = (A - B)/(2A) (D = (A + C)/G with C = -B, E_bdy = 2A/G; P33's convention), r_s/B and R_s/(2GM).
  J1: the first law's residual is the record's centred-step value (dM - mu_inf dN_B)/dM at each member (the solver re-run at z_c +- step).
  Lattice families: the record has no per-member first-law steps; the residual is estimated from cubic splines of M and N_B in ln z_c over the
  20 members, (M' - mu_inf N_B')/M' (labelled "spline"); the masses are also given in J1's units by the invariant of the TF scaling,
  M~ = G^(3/2) M m^2 and N_B~ = G^(3/2) m^3 N_B (with the same g = 16 these equal J1's M and N_B at the same z_c up to the remainders).
Controls (P2e): (a) the dilute limit Q_b -> -3/7 (the n = 3/2 polytrope): Q_b + 3/7 against z_c over J1's first ten reference members, the
least-squares slope of ln|Q_b + 3/7| against ln z_c (expected 2) and (Q_b + 3/7)/z_c^2; (b) per family the member brackets of the maxima of M
and N_B and of the minimum of E_b, and their spline-refined locations in z_c; (c) at J1's Brent maximum (the record's reference_maximum): E_b/M
and 2GM/R_s; (d) the two lattice families against J1's reference at the same z_c (J1 interpolated by cubic splines in ln z_c): the relative
differences of M~ and of 2GM/R_s, and the differences of E_b/M and mu_inf/m.  Numbers emitted, not retyped."""
import argparse
import json
import math

import numpy as np
from scipy.interpolate import CubicSpline


def f(v, sig=4):
    if v is None:
        return "-"
    v = float(v)
    if v == 0:
        return "0"
    return ("%.*e" % (sig - 1, v)) if (abs(v) < 1e-3 or abs(v) >= 1e5) else ("%.*g" % (sig, v))


def table(head, rows):
    return "\n".join(["| " + " | ".join(str(c) for c in head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


HEAD = ["z_c", "M", "N_B", "E_b", "E_b/M", "2GM/R_s", "GM/R_s", "Q_b", "mu_inf/m", "N(0)", "first law residual", "A/B - 1", "D/E_bdy", "r_s/B",
        "R_s/(2GM)"]


def rows_of(ms):
    rows = []
    for x in ms:
        rows.append(["%.6g" % x["z"], "%.10e" % x["M"], "%.10e" % x["NB"], "%.6e" % x["Eb"], "%.6e" % (x["Eb"] / x["M"]), "%.8f" % x["C2"], "%.8f" % (x["C2"] / 2),
                     "%.6f" % x["Qb"], "%.8f" % x["mu"], "%.8f" % x["N0"], f(x["fl"], 3), f(x["AB1"], 3), f((x["A"] - x["B"]) / (2 * x["A"]), 3),
                     "%.5f" % (x["rs"] / x["B"]), "%.6f" % (1 / x["C2"])])
    return rows


def j1_members(lst):
    out = []
    for m in lst:
        M, NB, C2 = m["M"], m["N_B"], m["compactness_2GM_over_Rs"]
        Eb = M - NB
        fl = m.get("first_law") or {}
        out.append(dict(z=m["z_c"], M=M, NB=NB, Eb=Eb, C2=C2, Qb=Eb / (M * C2 / 2), mu=m["mu_inf"], N0=m["N0"], fl=fl.get("dM_minus_mu_dNB_rel"),
                        AB1=m["A_over_B_minus_1"], A=m["A"], B=m["B"], rs=m["r_s"]))
    return out


def lat_members(F):
    k = F["meta"]["k_op"]
    G = 1.0 / (2 * math.pi * k)
    m = F["meta"]["mlight"]
    ms = sorted(F["members"], key=lambda x: x["z_c"])
    z = np.array([x["z_c"] for x in ms])
    Mv = np.array([x["M"] for x in ms])
    Nv = np.array([x["N_B"] for x in ms])
    sM, sN = CubicSpline(np.log(z), Mv), CubicSpline(np.log(z), Nv)
    out = []
    for i, x in enumerate(ms):
        M, NB = x["M"], x["N_B"]
        Eb = M - m * NB
        C2 = x["compactness"]
        dM, dN = float(sM(math.log(x["z_c"]), 1)), float(sN(math.log(x["z_c"]), 1))
        fl = (dM - x["mu_inf"] * dN) / dM
        out.append(dict(z=x["z_c"], M=M, NB=NB, Eb=Eb, C2=C2, Qb=Eb / (M * C2 / 2), mu=x["mu_inf"] / m, N0=x["N0"], fl=fl, AB1=x["A_over_B_minus_1"],
                        A=x["A"], B=x["B"], rs=x["r_s"], Mt=G ** 1.5 * M * m * m, NBt=G ** 1.5 * m ** 3 * NB))
    return out, dict(G=G, m=m, k_op=k, couplings=F["meta"]["couplings"], operator=F["meta"]["operator"])


def brackets(ms):
    z = np.array([x["z"] for x in ms])
    res = {}
    for name, v, sgn in (("max M", [x["M"] for x in ms], 1), ("max N_B", [x["NB"] for x in ms], 1), ("min E_b", [x["Eb"] for x in ms], -1)):
        v = sgn * np.array(v)
        i = int(np.argmax(v))
        if 0 < i < len(z) - 1:
            sp = CubicSpline(np.log(z), v)
            xs = np.linspace(math.log(z[i - 1]), math.log(z[i + 1]), 20001)
            zr = float(math.exp(xs[int(np.argmax(sp(xs)))]))
        else:
            zr = None
        res[name] = (i, z[max(i - 1, 0)], z[min(i + 1, len(z) - 1)], zr)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--j1", required=True)
    ap.add_argument("--lat", required=True, help="comma-separated lattice family records (label=path)")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    S = json.load(open(a.j1))
    ref = j1_members(S["reference"])
    grid = j1_members(S["grid"]["8000"]["members"])
    lats = []
    for item in a.lat.split(","):
        lab, p = item.split("=", 1)
        ms, meta = lat_members(json.load(open(p)))
        lats.append((lab, p, ms, meta))
    out = ["# Strand 5 production P2: the family table on the covariant row (generated by solve/code/strand5/strand5_P2_table_v1.py)", ""]
    summ = []
    # (a) the dilute limit
    zf = np.array([x["z"] for x in ref[:10]])
    qf = np.array([x["Qb"] + 3.0 / 7.0 for x in ref[:10]])
    ok = qf != 0
    slope = float(np.polyfit(np.log(zf[ok]), np.log(np.abs(qf[ok])), 1)[0]) if ok.sum() >= 2 else None
    erow = [["%.6g" % z_, "%.8f" % (q_ - 3.0 / 7.0), "%.6e" % q_, "%.6f" % (q_ / z_ ** 2)] for z_, q_ in zip(zf, qf)]
    summ.append("P2e(a): Q_b = E_b/(M GM/R_s) at J1's first member (z_c = %s) is %s against -3/7 = %s; over the first ten members ln|Q_b + 3/7| against "
                "ln z_c has slope %s, and (Q_b + 3/7)/z_c^2 runs from %s to %s (sign %s)." % (
                    "%.4g" % zf[0], "%.6f" % (qf[0] - 3.0 / 7.0), "%.6f" % (-3.0 / 7.0), f(slope, 4), "%.5f" % (qf[0] / zf[0] ** 2), "%.5f" % (qf[-1] / zf[-1] ** 2),
                    "+" if qf[0] > 0 else "-"))
    # (b) brackets
    brow = []
    fams = [("J1 reference", ref), ("J1 grid n = 8000", grid)] + [(lab, ms) for lab, _, ms, _ in lats]
    for lab, ms in fams:
        b = brackets(ms)
        brow.append([lab] + ["%d (%.5g to %.5g); %s" % (b[k][0], b[k][1], b[k][2], ("%.6f" % b[k][3]) if b[k][3] else "-") for k in ("max M", "max N_B", "min E_b")])
        same = len({b[k][0] for k in b}) == 1
        summ.append("P2e(b) %s: the member index of the maximum of M, of N_B and of the minimum of E_b: %s (%s); spline-refined z_c %s." % (
            lab, ", ".join(str(b[k][0]) for k in ("max M", "max N_B", "min E_b")), "the same member" if same else "NOT the same member",
            ", ".join(("%.6f" % b[k][3]) if b[k][3] else "-" for k in ("max M", "max N_B", "min E_b"))))
    # (c) the maximum
    rm = S["controls"]["reference_maximum"]
    EbM = (rm["M"] - rm["N_B"]) / rm["M"]
    summ.append("P2e(c): at J1's Brent maximum z_c = %s: E_b/M = %s, 2GM/R_s = %s, M = %s, N_B = %s (M/M_0 = %s, m N_B/M_0 = %s)." % (
        "%.10f" % rm["z_c"], "%.7f" % EbM, "%.9f" % rm["compactness_2GM_over_Rs"], "%.10e" % rm["M"], "%.10e" % rm["N_B"],
        "%.6f" % rm["in_P16_units"]["M_over_M0"], "%.6f" % rm["in_P16_units"]["m_NB_over_M0"]))
    # (d) lattice against J1 at the same z_c
    lz = np.log([x["z"] for x in ref])
    sp = {k: CubicSpline(lz, [x[k] for x in ref]) for k in ("M", "C2", "mu")}
    spEbM = CubicSpline(lz, [x["Eb"] / x["M"] for x in ref])
    drow = []
    for lab, _, ms, meta in lats:
        for x in ms:
            l_ = math.log(x["z"])
            drow.append([lab, "%.6g" % x["z"], "%.10e" % x["Mt"], "%.4e" % (x["Mt"] / float(sp["M"](l_)) - 1), "%.4e" % (x["Eb"] / x["M"] - float(spEbM(l_))),
                         "%.4e" % (x["C2"] / float(sp["C2"](l_)) - 1), "%.4e" % (x["mu"] - float(sp["mu"](l_)))])
        mx = max(ms, key=lambda x: x["M"])
        l_ = math.log(mx["z"])
        summ.append("P2e(d) %s (operator %s, couplings %s, G = %s, m = %s): at its largest-M member (z_c = %s) M~ = G^(3/2) M m^2 = %s, %s relative to J1 "
                    "there; 2GM/R_s %s relative; E_b/M %s against J1." % (
                        lab, meta["operator"], meta["couplings"], "%.6f" % meta["G"], meta["m"], "%.6g" % mx["z"], "%.8e" % mx["Mt"],
                        "%.4e" % (mx["Mt"] / float(sp["M"](l_)) - 1), "%.4e" % (mx["C2"] / float(sp["C2"](l_)) - 1), "%+.4e" % (mx["Eb"] / mx["M"] - float(spEbM(l_)))))
    out += ["## S Summary (emitted)", ""] + ["- " + x for x in summ] + [""]
    out += ["## P2a J1 reference family (G = m = 1, g = 16; reference ODE integration; first law from the record's centred steps)", "", table(HEAD, rows_of(ref)), ""]
    out += ["## P2b J1 grid family, n = 8000 (the grid solver; first law from the record's centred steps)", "", table(HEAD, rows_of(grid)), ""]
    for i, (lab, p, ms, meta) in enumerate(lats):
        out += ["## P2%s %s (%s; lattice units, G = 1/(2 pi k_op) = %s, m = %s; the first law's residual from cubic splines in ln z_c, an estimate)" % (
            "cd"[i] if i < 2 else str(i), lab, p, "%.6f" % meta["G"], meta["m"]), "", table(HEAD + ["M~ = G^(3/2) M m^2", "N_B~ = G^(3/2) m^3 N_B"],
                                                                                         [r + ["%.10e" % x["Mt"], "%.10e" % x["NBt"]] for r, x in zip(rows_of(ms), ms)]), ""]
    out += ["## P2e Controls", "", "(a) The dilute limit over J1's first ten reference members:", "", table(["z_c", "Q_b", "Q_b + 3/7", "(Q_b + 3/7)/z_c^2"], erow), "",
            "(b) The member brackets (index; the neighbours' z_c; the spline-refined z_c) of the maxima of M and N_B and the minimum of E_b:", "",
            table(["family", "max M", "max N_B", "min E_b"], brow), "",
            "(d) The lattice families against J1's reference at the same z_c (J1 by cubic splines in ln z_c):", "",
            table(["family", "z_c", "M~", "M<internal>/M_J1 - 1", "E_b/M - J1's", "2GM/R_s ratio - 1", "mu_inf/m - J1's"], drow), ""]
    open(a.md, "w").write("\n".join(out) + "\n")
    print("\n".join("- " + x for x in summ))
    print("wrote", a.md)


if __name__ == "__main__":
    main()
