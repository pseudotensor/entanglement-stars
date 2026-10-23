#!/usr/bin/env python
"""Strand 5 P5 revision 6, r2 in the finite-light-mass form (JUDGMENT_strand5_production.md sec. 7, A1, the lead's BUILDER item; builder analyst).
From P5's record (P5run1/P5.json: the r2 rows' (a/R_*)^2 per mass, spacing and member, the members' z_c and areal R_s, k_a(0; phys) of G1 per
assignment, the masses m/m_P) and version 1's massless rows (P5run2/P5_r2.json, kept beside as the labelled proxy):
  L = log[m^2 (m^2 + 2 mu^2)/(m^2 + mu^2)^2] (the light packet's finite-mass q^4 coefficient on the rectangle (m^2, m^2 + mu^2, m^2 + mu^2,
      m^2 + 2 mu^2)), with mu a = 0.0130903478 at a/l_P = 0.0213 and 0.0033801 at 0.0055, and m a = (m/m_P)(a/l_P); evaluated as
      2 log(m/mu) + log 2 + log1p(x/2) - 2 log1p(x), x = (m/mu)^2;  g_4 = L/(2 pi^2);
  the coefficients referred to the isotropic radius r_*: (a/r_*)^2 = (a/R_*)^2 (R_*/r_*)^2 with r_* and R_* of the member from J1's shooting
      (strand5_star_v4.shoot at the member's z_c, g = 16); beside it psi_s^4 = (2/(1 + mu_inf/m))^4, the lead's form;
  delta M/M = c (a/r_*)^2 g_4/k_a(0) with c = 28.7 and 30.5, the kernel's own (A - B)/A = (3.6 to 3.9) (a/r_*)^2 g_4/k_a(0) (the same convention,
      stated).  Writes markdown and JSON."""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_star_v4 as STAR  # noqa: E402

MU = {0.0213: 0.0130903478, 0.0055: 0.0033801}
ASG = {0.0213: "physical", 0.0055: "window edge"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5-json", required=True)
    ap.add_argument("--v1-json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    P = json.load(open(a.p5_json))
    V1 = json.load(open(a.v1_json))
    masses = dict((k, float(v)) for k, v in (x.split("=") for x in P["meta"]["masses"].split(",")))
    cs = [float(x) for x in P["meta"]["r2_c"].split(",")]
    ss = [float(x) for x in P["meta"]["r2_strain"].split(",")]
    ka0 = {k: float(v) for k, v in P["r2"]["k_a0_phys_G1"].items()}
    geo = {}
    for lab, mem in P["members"].items():
        ref = STAR.shoot(mem["z_c"], 16.0)
        r_iso, R_ar, mu_inf = float(ref["r_s"]), float(ref["R_s_areal"]), float(ref["mu_inf"])
        geo[lab] = {"z_c": mem["z_c"], "r_iso": r_iso, "R_areal": R_ar, "R_areal_record": mem["R_s_areal"], "ratio_sq": (R_ar / r_iso) ** 2,
                    "psi_s4_lead": (2 / (1 + mu_inf)) ** 4, "mu_inf_over_m": mu_inf}
    v1 = {(q["mass"], round(q["a_over_lP"], 6), q["member"]): q for q in V1}
    out, rows = [], []
    for q in P["r2"]["rows"]:
        al = q["a_over_lP"]
        key = min(MU, key=lambda x: abs(x - al))
        mu, asg = MU[key], ASG[key]
        ma = masses[q["mass"]] * al
        x = (ma / mu) ** 2
        L = 2 * math.log(ma / mu) + math.log(2) + math.log1p(x / 2) - 2 * math.log1p(x)
        g4 = L / (2 * math.pi ** 2)
        ratio = g4 / ka0[asg]
        g = geo[q["member"]]
        x2r = q["a_over_Rstar_sq"] * g["ratio_sq"]
        dM = [c * x2r * ratio for c in cs]
        kAB = [s * x2r * ratio for s in ss]
        old = v1.get((q["mass"], round(al, 6), q["member"]))
        rec = {"mass": q["mass"], "a_over_lP": al, "member": q["member"], "assignment": asg, "m_a": ma, "mu_a": mu, "L_finite": L, "g4": g4,
               "g4_over_ka0": ratio, "a_over_rstar_sq": x2r, "R_over_r_sq": g["ratio_sq"], "psi_s4_lead": g["psi_s4_lead"], "deltaM_over_M": dM,
               "kernel_AmB_over_A": kAB, "massless_proxy": {"L_eff": old["L_eff"], "deltaM_over_M_areal": old["deltaM_over_M"]} if old else None}
        out.append(rec)
        rows.append("| %s | %s | %s | %s | %.4e | %.6g | %.6e | %.6e | %.5f | %.4e to %.4e | %.4e to %.4e | %s; %s |" % (
            q["mass"], al, q["member"], asg, ma, L, g4, ratio, g["ratio_sq"], dM[0], dM[1], kAB[0], kAB[1],
            ("%.5g" % old["L_eff"]) if old else "-", ("%.4e to %.4e" % tuple(old["deltaM_over_M"])) if old else "-"))
    mx = [r for r in out if r["member"] == "maximum"]
    summ = ["At the maximum, delta M/M in the finite-light-mass form referred to r_*: " + "; ".join(
        "%s, a/l_P = %s: %.3e to %.3e (L = %.5g)" % (r["mass"], r["a_over_lP"], r["deltaM_over_M"][0], r["deltaM_over_M"][1], r["L_finite"]) for r in mx) + ".",
            "The members' (R_*/r_*)^2 from J1's shooting: " + "; ".join("%s %.6f (the lead's psi_s^4 form %.6f)" % (k, v["ratio_sq"], v["psi_s4_lead"]) for k, v in geo.items()) + ".",
            "Orders at the maximum: m_e %s, m_n %s (the lead's stop rule: 1e-87 and 1e-74)." % (
                sorted(set("1e%d" % math.floor(math.log10(abs(v))) for r in mx if r["mass"] == "m_e" for v in r["deltaM_over_M"])),
                sorted(set("1e%d" % math.floor(math.log10(abs(v))) for r in mx if r["mass"] == "m_n" for v in r["deltaM_over_M"])))]
    md = ["# Strand 5 P5 revision 6: r2 in the finite-light-mass form, referred to r_* (strand5_P5_r2_v2.py)", "", "## Emitted summary", ""] + [
        "- " + s for s in summ] + ["",
        "## r2 rows (the massless rows of revision 5 beside them, the labelled proxy: its L and its delta M/M with the areal R_*)", "",
        "| mass | a/l_P | member | assignment | m a | L (finite mass) | g_4 | g_4/k_a(0) | (R_*/r_*)^2 | delta M/M (c = 28.7, 30.5) | kernel (A - B)/A | proxy: L; delta M/M |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    open(a.md, "w").write("\n".join(md) + "\n")
    json.dump({"rows": out, "members": geo, "summary": summ}, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(md[:8]))


if __name__ == "__main__":
    main()
