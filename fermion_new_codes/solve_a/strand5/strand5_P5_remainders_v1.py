#!/usr/bin/env python
"""Strand 5 P5 (SPEC_strand5_covariant_row.md sec. 3 P5; builder analyst): the remainders table for the production star of the covariant row.
The builder emits the numbers; the lead assembles the reading.  The two members are J1's reference (strand5_star_v4.shoot, g = 16, G = m = 1):
the OV maximum (Brent on M(z_c)) and the member with GM/R_s = 2.1e-6 (solar compactness; Brent's root on z_c), R_s areal.  Physical inputs
(the spec's): m/m_P = 4.19e-23 (m_e) and 7.69e-20 (m_n); a/l_P = 0.0213 (the massless-light rectangle, mu = 0.0130903478) and 0.0055 (the
window-edge assignment); m a = (m/m_P)(a/l_P); J1's length unit is l_P (m_P/m)^2, so a/R_* = (a/l_P)(m/m_P)^2/R_s(J1) (the spec's R_0 = m_P/(sqrt(8)
m^2) mapping: J1's R_s = (R_s/R_0) R_0).
  r1 (P14 eq. 21 class): Q_1 := (2/5) Int N psi^10 I_6(k_F) r^2 dr/Int psi^5 I_rho(k_F) r^2 dr on the member's reference profile (composite
     Gauss-Legendre, 16 nodes per panel on the ODE's dense output, 400 panels, and the change from 200), k_F = sqrt(mu^2 - 1) the local proper
     Fermi momentum, I_rho(k) = int_0^k p^2 sqrt(1 + p^2) dp, I_6(k) = int_0^k p^6/sqrt(1 + p^2) dp (closed forms for k >= 0.1, the binomial
     series below: the closed form of I_6 cancels to rounding at the solar member's k_F ~ 2e-3); (A - B)_TF/A = -Q_1 (m a)^2.  Beside it the
     measured local values: the I2b table of RESULT_strand5_family.md (its last occurrence; the refined local_ka0 families' M_max against J1's
     maximum mapped, ratio - 1 recomputed from the table's M_max and J1 columns) at (0.01, 0.1) and (0.03, 0.3) and their ratio; and the family
     records' measured (A - B)/A and P14 prediction at the member nearest z_c = 0.8234 (local_ka0, (0, 0)).
  r2: delta M/M = c (g_4/k_a(0)) (a/R_*)^2 with c from the spec's range (28.7 to 30.5, JUDGMENT_strand5_family.md sec. 6, the J4 scaling table),
     the kernel's own (A - B)/A with the spec's 3.6 to 3.9; the physical g_4/k_a(0) is not evaluated (the physical rectangle (0, mu, mu,
     sqrt2 mu) has a massless packet, whose log m^2 in L = sum eta_s log m_s^2 diverges; the question is with the lead): the table gives
     c (a/R_*)^2 and the strain coefficient times (a/R_*)^2, to be multiplied by g_4/k_a(0); k_a(0; phys) of G1 (the last G1 table) is printed.
  r3: the last G3 table of RESULT_strand5_kernel_eos.md (revision 9), rows physical and window edge at the maximum and solar members, copied.
  r4: e_M/e_0 and e_B/e_0 from strand5_eos_v4.e1_parts (the resolved-cone forms, (g_M, g_B, Lambda, l) = (0.2, 1, 1, 1) lattice units, eight
     cones, n = 8 k_F^3/(3 pi^2), k_F = z_c m a) against the g = 16 continuum e_0 (strand5_eos_v4.e_cont_hat) at m a = 0.1, 0.03, 0.01 and
     z_c = 0.83; the least-squares power in m a and the pairwise powers; the value at the physical m a by the power law from m a = 0.01 and by
     direct evaluation; and the (0.2, 1) families' mass shift M(0.2, 1)/M(0, 0) - 1 at every common member (J4run2's four inventories, three
     operators each; J4run4's packet-built (0.01, 0.1) kernel and local_ka0): the member nearest z_c = 0.8234 and the range; [S: lane strand A,
     omits P36].
  r5 not computed (labelled; r4's order, O(g)); r6 "J5 pending"; r7 not quantified (stated).
Writes the markdown (--md) and the JSON (--out-json)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import math
import sys

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import binom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strand5_star_v4 as STAR  # noqa: E402
import strand5_eos_v4 as EOS  # noqa: E402

G16 = 16.0
C6 = [binom(-0.5, j) / (2 * j + 7) for j in range(14)]
CR = [binom(0.5, j) / (2 * j + 3) for j in range(14)]


def I_rho(k):
    k = np.asarray(k, float)
    s = np.sqrt(1 + k * k)
    out = (k * (2 * k * k + 1) * s - np.arcsinh(k)) / 8
    sm = k < 0.1
    if np.any(sm):
        ks = k[sm]
        out[sm] = ks ** 3 * sum(c * ks ** (2 * j) for j, c in enumerate(CR))
    return out


def I_6(k):
    k = np.asarray(k, float)
    s = np.sqrt(1 + k * k)
    out = (k * s * (8 * k ** 4 - 10 * k * k + 15) - 15 * np.arcsinh(k)) / 48
    sm = k < 0.1
    if np.any(sm):
        ks = k[sm]
        out[sm] = ks ** 7 * sum(c * ks ** (2 * j) for j, c in enumerate(C6))
    return out


def q1(zc, panels):
    ref = STAR.shoot(zc, G16, keep=True)
    sol = ref["_sol"]
    alpha, beta, R = ref["_scal"]
    x, w = np.polynomial.legendre.leggauss(16)
    e = np.linspace(sol.t[0], R, panels + 1)
    rr = ((e[1:] - e[:-1])[:, None] * (x[None, :] + 1) / 2 + e[:-1, None]).ravel()
    ww = ((e[1:] - e[:-1])[:, None] * w[None, :] / 2).ravel()
    Y = sol.sol(rr)
    psi, chi = Y[0] / alpha, Y[2] / beta
    N = chi / psi
    mu = math.sqrt(1 + zc * zc) * Y[0] / Y[2]
    kF = np.sqrt(np.clip(mu * mu - 1, 0, None))
    r = rr * alpha ** 2
    num = float(np.sum(ww * r * r * N * psi ** 10 * I_6(kF)))
    den = float(np.sum(ww * r * r * psi ** 5 * I_rho(kF)))
    return 0.4 * num / den, ref


def last_table(md_path, heading_prefix):
    """the rows (lists of cells) of the last markdown table under a heading starting with heading_prefix, and the heading's line number."""
    lines = open(md_path).read().split("\n")
    idx = [i for i, ln in enumerate(lines) if ln.startswith(heading_prefix)]
    if not idx:
        return None, None, None
    i = idx[-1] + 1
    while i < len(lines) and not lines[i].startswith("|"):
        i += 1
    head = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    rows = []
    i += 2
    while i < len(lines) and lines[i].startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    return head, rows, idx[-1] + 1


def fam_members(path):
    with open(path) as fh:
        return json.load(fh)["members"]


def nearest(mem, z):
    return min(mem, key=lambda q: abs(q["z_c"] - z))


def fmt(x, d=6):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    x = float(x)
    if x == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e5:
        return ("%." + str(d) + "g") % x
    return "%.4e" % x


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-md", required=True, help="RESULT_strand5_family.md")
    ap.add_argument("--kernel-eos-md", required=True, help="RESULT_strand5_kernel_eos.md")
    ap.add_argument("--local-families", required=True, help="label=path,... (local_ka0, (0, 0)) for r1's measured strain")
    ap.add_argument("--coupling-pairs", required=True, help="label=path_g00:path_g021,... for r4's mass shift")
    ap.add_argument("--solar", type=float, default=2.1e-6)
    ap.add_argument("--masses", default="m_e=4.19e-23,m_n=7.69e-20")
    ap.add_argument("--spacings", default="0.0213,0.0055")
    ap.add_argument("--r2-c", default="28.7,30.5")
    ap.add_argument("--r2-strain", default="3.6,3.9")
    ap.add_argument("--zc-r4", type=float, default=0.83)
    ap.add_argument("--mhats-r4", default="0.1,0.03,0.01")
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    masses = [(s.split("=")[0], float(s.split("=")[1])) for s in a.masses.split(",")]
    spac = [float(s) for s in a.spacings.split(",")]
    out = {"meta": vars(a)}
    # the members
    zmax = float(minimize_scalar(lambda z: -STAR.shoot(z, G16)["M"], bounds=(0.75, 0.95), method="bounded", options={"xatol": 1e-11}).x)
    zsol = float(brentq(lambda z: STAR.shoot(z, G16)["compactness_2GM_over_Rs"] / 2 - a.solar, 5e-4, 2e-2, xtol=1e-15, rtol=1e-13))
    members = {}
    for lab, z in (("maximum", zmax), ("solar", zsol)):
        Q400, ref = q1(z, 400)
        Q200, _ = q1(z, 200)
        members[lab] = {"z_c": z, "GM_over_Rs": ref["compactness_2GM_over_Rs"] / 2, "R_s_areal": ref["R_s_areal"], "A": ref["A"], "Q1": Q400,
                        "Q1_400_minus_200": Q400 - Q200}
    out["members"] = members
    L = ["# Strand 5 P5 tables (strand5_P5_remainders_v1.py)", ""]
    L += ["## Members (J1's reference, g = 16, G = m = 1)", "",
          table(["member", "z_c", "GM/R_s", "R_s (areal, J1 units)", "Q_1", "Q_1 400 - 200 panels"],
                [[k, fmt(v["z_c"], 11), fmt(v["GM_over_Rs"]), fmt(v["R_s_areal"], 9), fmt(v["Q1"], 9), fmt(v["Q1_400_minus_200"])] for k, v in members.items()]), ""]
    # r1 measured
    head, rows, ln = last_table(a.family_md, "### I2b")
    meas = []
    if rows:
        ci = {h: i for i, h in enumerate(head)}
        for r in rows:
            if r[ci["operator"]] == "local_ka0":
                Mx, J1m = float(r[ci["M_max"]]), float(r[ci["J1 M_max mapped"]])
                meas.append({"family": r[ci["(m a, aH)"]], "couplings": r[ci["(g_M, g_B)"]], "M_max": Mx, "J1_mapped": J1m, "ratio_minus_1": Mx / J1m - 1})
    rat = {}
    for c in ("0, 0", "0.2, 1"):
        v = {q["family"]: q["ratio_minus_1"] for q in meas if q["couplings"] == c}
        if "0.03, 0.3" in v and "0.01, 0.1" in v:
            rat[c] = v["0.03, 0.3"] / v["0.01, 0.1"]
    strain = []
    for item in a.local_families.split(","):
        lab, p = item.split("=")
        q = nearest(fam_members(p), 0.8234)
        strain.append({"family": lab, "record": p, "z_c": q["z_c"], "measured_AmB_over_A": q.get("measured_AmB_over_A"),
                       "P14_prediction": q.get("P14_TF_prediction_AmB_over_A")})
    out["r1"] = {"measured_I2b": meas, "I2b_line": ln, "I2b_ratio_0.03_over_0.01": rat, "strain_records": strain}
    L += ["## r1 The lattice EOS strain: -Q_1 (m a)^2; the measured local values", "",
          "I2b (RESULT_strand5_family.md, the table under line %s): M_max/J1 mapped - 1 recomputed from the table's columns:" % ln, "",
          table(["family (m a, aH)", "(g_M, g_B)", "M_max", "J1 mapped", "ratio - 1"],
                [[q["family"], q["couplings"], fmt(q["M_max"], 9), fmt(q["J1_mapped"], 9), fmt(q["ratio_minus_1"])] for q in meas]), "",
          "ratio (0.03, 0.3)/(0.01, 0.1): " + ", ".join("(%s) %s" % (k, fmt(v, 5)) for k, v in rat.items()), "",
          "The family records at the member nearest z_c = 0.8234:", "",
          table(["family", "record", "z_c", "measured (A - B)/A", "P14 prediction"],
                [[q["family"], q["record"], fmt(q["z_c"], 7), fmt(q["measured_AmB_over_A"]), fmt(q["P14_prediction"])] for q in strain]), ""]
    if len(strain) == 2 and strain[0]["measured_AmB_over_A"] and strain[1]["measured_AmB_over_A"]:
        L += ["ratio of the measured strains (%s)/(%s): %s" % (strain[1]["family"], strain[0]["family"],
                                                              fmt(strain[1]["measured_AmB_over_A"] / strain[0]["measured_AmB_over_A"], 5)), ""]
    # r2
    cs = [float(x) for x in a.r2_c.split(",")]
    ss = [float(x) for x in a.r2_strain.split(",")]
    g1h, g1rows, g1ln = last_table(a.kernel_eos_md, "### G1")
    ka0 = {}
    if g1rows:
        ci = {h: i for i, h in enumerate(g1h)}
        for r in g1rows:
            ka0[r[ci["assignment"]]] = r[ci["k_a(0; phys)"]]
    r2 = []
    for mlab, mm in masses:
        for al in spac:
            for lab, v in members.items():
                x2 = (al * mm * mm / v["R_s_areal"]) ** 2
                r2.append({"mass": mlab, "a_over_lP": al, "member": lab, "a_over_Rstar_sq": x2, "c_times": [c * x2 for c in cs],
                           "strain_times": [s * x2 for s in ss]})
    out["r2"] = {"rows": r2, "k_a0_phys_G1": ka0, "G1_line": g1ln, "g4_over_ka0_phys": "pending (the massless packet's log m^2)"}
    L += ["## r2 The kernel's mass shift: delta M/M = c (g_4/k_a(0)) (a/R_*)^2 (c = %s to %s); the kernel's own (A - B)/A: (%s to %s) (g_4/k_a(0)) (a/R_*)^2"
          % (cs[0], cs[-1], ss[0], ss[-1]), "",
          "g_4/k_a(0) at the physical masses: pending (the physical rectangle's massless packet: log m^2 diverges in L = sum eta_s log m_s^2). "
          "k_a(0; phys) from G1 (the table under line %s): %s." % (g1ln, ", ".join("%s %s" % kv for kv in ka0.items())), "",
          table(["mass", "a/l_P", "member", "(a/R_*)^2", "c (a/R_*)^2 (low, high)", "strain coefficient (a/R_*)^2 (low, high)"],
                [[q["mass"], fmt(q["a_over_lP"]), q["member"], fmt(q["a_over_Rstar_sq"]), ", ".join(fmt(x) for x in q["c_times"]),
                  ", ".join(fmt(x) for x in q["strain_times"])] for q in r2]), ""]
    # r3
    g3h, g3rows, g3ln = last_table(a.kernel_eos_md, "### G3")
    r3 = []
    if g3rows:
        ci = {h: i for i, h in enumerate(g3h)}
        for r in g3rows:
            if r[ci["member"]] in ("maximum", "solar"):
                r3.append({"assignment": r[ci["assignment"]], "member": r[ci["member"]], "z_c": r[ci["z_c"]], "gamma_minus_1": r[ci["gamma - 1"]],
                           "beta_minus_1_charge": r[ci["beta - 1 (charge)"]], "q0_gamma2_over_8": r[ci["q_0 gamma^2/8"]],
                           "beta_minus_1_with_it": r[ci["beta - 1 (with it)"]], "against": r[ci["against"]]})
    out["r3"] = {"rows": r3, "G3_line": g3ln}
    L += ["## r3 The running coefficient's ruler dependence (copied from RESULT_strand5_kernel_eos.md, the G3 table under line %s)" % g3ln, "",
          table(["assignment", "member", "z_c", "gamma - 1", "beta - 1 (charge)", "q_0 gamma^2/8", "beta - 1 (with it)", "against"],
                [[q["assignment"], q["member"], q["z_c"], q["gamma_minus_1"], q["beta_minus_1_charge"], q["q0_gamma2_over_8"], q["beta_minus_1_with_it"],
                  q["against"]] for q in r3]), ""]
    # r4
    mh = [float(x) for x in a.mhats_r4.split(",")]
    e1r = []
    for m_ in mh:
        k = a.zc_r4 * m_
        n = 8 * k ** 3 / (3 * np.pi ** 2)
        p = EOS.e1_parts(m_, n)
        e0 = EOS.e_cont_hat(m_, a.zc_r4)
        e1r.append({"m_a": m_, "n": n, "e_0": e0, "e_M": p["e_M"], "e_B": p["e_B"], "eM_over_e0": p["e_M"] / e0, "eB_over_e0": p["e_B"] / e0,
                    "e1_over_e0": p["e_1"] / e0})
    lm = np.log(mh)
    pw = {}
    for key in ("eM_over_e0", "eB_over_e0", "e1_over_e0"):
        y = np.log(np.abs([q[key] for q in e1r]))
        pw[key] = {"least_squares": float(np.polyfit(lm, y, 1)[0]), "pairwise": [float((y[i + 1] - y[i]) / (lm[i + 1] - lm[i])) for i in range(len(mh) - 1)]}
    phys = []
    i0 = int(np.argmin(mh))
    for mlab, mm in masses:
        for al in spac:
            ma = mm * al
            row = {"mass": mlab, "a_over_lP": al, "m_a": ma}
            for key in ("eM_over_e0", "eB_over_e0", "e1_over_e0"):
                row[key + "_power_law"] = e1r[i0][key] * (ma / mh[i0]) ** pw[key]["least_squares"]
            k = a.zc_r4 * ma
            n = 8 * k ** 3 / (3 * np.pi ** 2)
            p = EOS.e1_parts(ma, n)
            e0 = EOS.e_cont_hat(ma, a.zc_r4)
            row.update({"eM_over_e0_direct": p["e_M"] / e0, "eB_over_e0_direct": p["e_B"] / e0, "e1_over_e0_direct": p["e_1"] / e0})
            phys.append(row)
    shifts = []
    for item in a.coupling_pairs.split(","):
        lab, pp = item.split("=")
        p0, p1 = pp.split(":")
        m0 = {round(q["z_c"], 9): q for q in fam_members(p0)}
        m1 = {round(q["z_c"], 9): q for q in fam_members(p1)}
        common = sorted(z for z in m0 if z in m1)
        sh = [(z, m1[z]["M"] / m0[z]["M"] - 1) for z in common]
        if not sh:
            shifts.append({"family": lab, "n_common": 0})
            continue
        zn = min(common, key=lambda z: abs(z - 0.8234))
        vals = [s for _, s in sh]
        shifts.append({"family": lab, "n_common": len(common), "z_c_nearest": zn, "shift_nearest": m1[zn]["M"] / m0[zn]["M"] - 1, "min": min(vals),
                       "max": max(vals)})
    out["r4"] = {"e1": e1r, "powers": pw, "physical": phys, "coupling_shifts": shifts}
    L += ["## r4 The first coupling correction [S: lane strand A, omits P36]", "",
          "e_1 parts against the g = 16 continuum e_0 at z_c = %s ((g_M, g_B, Lambda, l) = (0.2, 1, 1, 1) lattice units):" % a.zc_r4, "",
          table(["m a", "n", "e_0", "e_M", "e_B", "e_M/e_0", "e_B/e_0", "e_1/e_0"],
                [[fmt(q["m_a"]), fmt(q["n"]), fmt(q["e_0"]), fmt(q["e_M"]), fmt(q["e_B"]), fmt(q["eM_over_e0"]), fmt(q["eB_over_e0"]), fmt(q["e1_over_e0"])]
                 for q in e1r]), "",
          "powers in m a: " + "; ".join("%s least squares %s, pairwise %s" % (k, fmt(v["least_squares"], 5), ", ".join(fmt(x, 5) for x in v["pairwise"]))
                                        for k, v in pw.items()), "",
          table(["mass", "a/l_P", "m a", "e_M/e_0 (power law)", "e_M/e_0 (direct)", "e_B/e_0 (power law)", "e_B/e_0 (direct)", "e_1/e_0 (power law)",
                 "e_1/e_0 (direct)"],
                [[q["mass"], fmt(q["a_over_lP"]), fmt(q["m_a"]), fmt(q["eM_over_e0_power_law"]), fmt(q["eM_over_e0_direct"]), fmt(q["eB_over_e0_power_law"]),
                  fmt(q["eB_over_e0_direct"]), fmt(q["e1_over_e0_power_law"]), fmt(q["e1_over_e0_direct"])] for q in phys]), "",
          "The (0.2, 1) families' mass shift against (0, 0) at the same z_c:", "",
          table(["family", "common members", "z_c nearest 0.8234", "shift there", "smallest", "largest"],
                [[q["family"], fmt(q["n_common"]), fmt(q.get("z_c_nearest"), 7), fmt(q.get("shift_nearest")), fmt(q.get("min")), fmt(q.get("max"))]
                 for q in shifts]), ""]
    # the P5 table
    cols = [(mlab, mm, al) for mlab, mm in masses for al in spac]
    head = ["remainder", "class", "lattice-unit measurement (record)", "scaling law"] + ["%s, a/l_P = %s: maximum; solar" % (ml, fmt(al)) for ml, _, al in cols]
    rows = []
    rows.append(["r1", "lattice EOS strain (P14 eq. 21)", "Q_1 = %s (maximum), %s (solar); I2b ratio %s" % (
        fmt(members["maximum"]["Q1"], 7), fmt(members["solar"]["Q1"], 7), ", ".join("(%s) %s" % (k, fmt(v, 4)) for k, v in rat.items())), "(m a)^2"] +
        ["%s; %s" % (fmt(-members["maximum"]["Q1"] * (mm * al) ** 2), fmt(-members["solar"]["Q1"] * (mm * al) ** 2)) for _, mm, al in cols])

    def r2cell(mlab, al):
        q = {x["member"]: x for x in r2 if x["mass"] == mlab and x["a_over_lP"] == al}
        return "%s; %s (x g_4/k_a(0))" % (fmt(q["maximum"]["c_times"][0]) + " to " + fmt(q["maximum"]["c_times"][-1]),
                                          fmt(q["solar"]["c_times"][0]) + " to " + fmt(q["solar"]["c_times"][-1]))
    rows.append(["r2", "nonlocal kernel's mass shift (dilation anomaly)", "c = %s to %s (J4 scaling table, spec)" % (cs[0], cs[-1]),
                 "(m/m_P)^4 (a/l_P)^2 x g_4/k_a(0) (log)"] + [r2cell(ml, al) for ml, _, al in cols])

    def r3cell(al):
        asg = "physical" if abs(al - spac[0]) < 1e-12 else "window edge"
        q = {x["member"]: x for x in r3 if x["assignment"] == asg}
        if "maximum" not in q or "solar" not in q:
            return "-"
        return "gamma - 1 %s, beta - 1 %s; %s, %s" % (q["maximum"]["gamma_minus_1"], q["maximum"]["beta_minus_1_with_it"], q["solar"]["gamma_minus_1"],
                                                     q["solar"]["beta_minus_1_with_it"])
    rows.append(["r3", "running coefficient's ruler dependence (P14 eq. 7, J2g)", "G3 (revision 9)", "profile; coefficient channel profile independent"] +
                [r3cell(al) for _, _, al in cols])

    def r4cell(mlab, al):
        q = [x for x in phys if x["mass"] == mlab and x["a_over_lP"] == al][0]
        return "%s; -" % fmt(q["e1_over_e0_direct"])
    rows.append(["r4", "first coupling correction [S: lane strand A, omits P36]", "e_1/e_0 power %s; (0.2, 1) shifts in the families" % fmt(
        pw["e1_over_e0"]["least_squares"], 4), "(m a)^p at fixed z_c"] + [r4cell(ml, al) for ml, _, al in cols])
    rows.append(["r5", "P36 ruler-row term", "not computed (labelled)", "r4's order, O(g)"] + ["-"] * len(cols))
    rows.append(["r6", "four-derivative nonlinearity of the vacuum (J5)", "J5 pending", "(lambda_C/R_*)^2"] + ["J5 pending"] * len(cols))
    rows.append(["r7", "local-density approximation, isotropic [100] symbol, radial reduction", "not quantified (stated)", "-"] + ["-"] * len(cols))
    out["P5_table"] = {"head": head, "rows": rows}
    L = L[:2] + ["## P5 The remainders (emitted; no inference by the builder)", "", table(head, rows), ""] + L[2:]
    with open(a.md, "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
