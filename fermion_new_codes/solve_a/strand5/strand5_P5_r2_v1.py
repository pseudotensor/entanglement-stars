#!/usr/bin/env python
"""Strand 5 P5 revision 5, r2's physical factor (SPEC_strand5_covariant_row.md sec. 9(b); builder analyst).  From P5's record (P5run1/P5.json:
the r2 rows with (a/R_*)^2 per mass, spacing and member, and k_a(0; phys) of G1 per assignment), the lead's rule:
    g_4(1/R_*) = [log 2 - 2 ln(mu R_*/a) + c']/(2 pi^2),   c' = 0 (stated: an O(1) unmeasured constant of the massless packet's q^4 log q^2 term),
mu = 0.0130903478 with k_a(0; phys) "physical" for a/l_P = 0.0213 (the diagnostic assignment) and mu = 0.0033801 with "window edge" for
a/l_P = 0.0055; R_*/a = ((a/R_*)^2)^(-1/2) per row.  Per row: L_eff = log 2 - 2 ln(mu R_*/a), g_4, g_4/k_a(0), and the r2 entries
c (a/R_*)^2 g_4/k_a(0) (c low and high) and the kernel's own (A - B)/A, (3.6 to 3.9) (a/R_*)^2 g_4/k_a(0).  Writes markdown and JSON."""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p5-json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cprime", type=float, default=0.0)
    a = ap.parse_args()
    P = json.load(open(a.p5_json))
    r2 = P["r2"]
    ka0 = {k: float(v) for k, v in r2["k_a0_phys_G1"].items()}
    mu = {0.0213: (0.0130903478, "physical"), 0.0055: (0.0033801, "window edge")}
    rows, out = [], []
    for q in r2["rows"]:
        al = q["a_over_lP"]
        m_, asg = mu[min(mu, key=lambda x: abs(x - al))]
        Rs_a = q["a_over_Rstar_sq"] ** -0.5
        L = math.log(2) - 2 * math.log(m_ * Rs_a) + a.cprime
        g4 = L / (2 * math.pi ** 2)
        ratio = g4 / ka0[asg]
        c_lo, c_hi = [x * ratio for x in q["c_times"]]
        s_lo, s_hi = [x * ratio for x in q["strain_times"]]
        rec = {"mass": q["mass"], "a_over_lP": al, "member": q["member"], "assignment": asg, "mu": m_, "R_star_over_a": Rs_a, "L_eff": L, "g4": g4,
               "g4_over_ka0": ratio, "deltaM_over_M": [c_lo, c_hi], "kernel_AmB_over_A": [s_lo, s_hi]}
        out.append(rec)
        rows.append("| %s | %s | %s | %s | %.6g | %.4e | %.6g | %.6e | %.6e | %.4e to %.4e | %.4e to %.4e |" % (
            q["mass"], al, q["member"], asg, m_, Rs_a, L, g4, ratio, c_lo, c_hi, s_lo, s_hi))
    md = ["# Strand 5 P5 revision 5: r2 at the physical inventory (strand5_P5_r2_v1.py)", "",
          "g_4(1/R_*) = [log 2 - 2 ln(mu R_*/a) + c']/(2 pi^2) with c' = %g (the lead's rule, SPEC_strand5_covariant_row.md sec. 9(b)); k_a(0; phys) "
          "from G1: %s." % (a.cprime, ", ".join("%s %s" % kv for kv in ka0.items())), "",
          "| mass | a/l_P | member | assignment | mu | R_*/a | L_eff | g_4 | g_4/k_a(0) | delta M/M (c low to high) | kernel's (A - B)/A |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    open(a.md, "w").write("\n".join(md) + "\n")
    json.dump(out, open(a.out_json, "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
