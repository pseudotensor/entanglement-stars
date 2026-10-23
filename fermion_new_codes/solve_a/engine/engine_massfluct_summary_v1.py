#!/usr/bin/env python
"""engine_massfluct_summary_v1.py (builder analyst): the table of engine_massfluct_v3's records (MF/massfluct3_<tag>.json): per member N_B,
E, the binding energy E - N_B m, the Wick exchange E_x^W, the same-kernel Hartree pair energy E_H (and from the record's eps_tot), y^W = E_x^W/E_H,
E_x^W/(E - N_B m), version 2's form E_x^(0) (matter, sea), the one-particle control's relative sum, and at R_surf the variances kept apart
(sqrt V(C)/E, sqrt V(C_0)/E, the signed sqrt dV/E, sqrt V(P)/E, the Wick-ordered source product over E^2, version 2's sqrt(Var_mat)/E), with
1/sqrt(N_B) beside. Args: records... --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = sorted((json.load(open(f)) for f in a.records), key=lambda d: d["N_B"])
    L = ["## engine_massfluct_v3 on the (0.1, 1/4) family's closed-count members (engine_massfluct_summary_v1.py over %s)" % ", ".join(os.path.basename(f) for f in a.records), "",
         "### The exchange in the Wick ordering against the star's own sea, and the denominators", "",
         "| member | N_B | E | E - N_B m (binding) | E_x^W | E_H (same kernel) | E_H from eps_tot | y^W = E_x^W/E_H | E_x^W/(E - N_B m) | E_x^(0) v2 form: matter, sea | one-particle control (E_x^W + E_H)/|E_H| |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in rows:
        c = d.get("one_particle_control") or {}
        L.append("| %d | %.2f | %.6f | %.6e | %.6e | %.6e | %.6e | %.6e | %.6e | %.4e, %.4e | %s |" % (
            d["member"], d["N_B"], d["E"], d["binding_E_minus_NBm"], d["E_x_W"], d["E_H_same_kernel"], d["E_H_from_record_eps_tot"], d["y_W"],
            d["E_x_W_over_binding"], d["Ex0_matter_v2_form"], d["Ex0_sea_v2_form"], ("%.1e" % c["relative"]) if c.get("relative") is not None else "-"))
    L += ["", "### The variances at R_surf, kept apart (each over E; the Wick product over E^2)", "",
          "| member | N_B | 1/sqrt(N_B) | sqrt V(C)/E (physical) | sqrt V(C_0)/E (reference) | signed sqrt dV/E (vacuum-subtracted) | sqrt V(P)/E (matter alone) | Wick product/E^2 | v2's sqrt(Var_mat)/E |",
          "|---|---|---|---|---|---|---|---|---|"]
    for d in rows:
        r = d["reads"].get("R_surf", {})
        f = lambda k, fmt="%.6e": (fmt % r[k]) if k in r and r[k] is not None else "-"
        L.append("| %d | %.2f | %.5f | %s | %s | %s | %s | %s | %s |" % (d["member"], d["N_B"], d["one_over_sqrt_NB"], f("sqrt_V_C_over_E"), f("sqrt_V_C0_over_E"),
                 f("signed_sqrt_dV_over_E"), f("sqrt_V_P_over_E"), f("Wick_product_over_E2"), f("v2_fluct_mat_over_E")))
    L += ["", "- every number above is printed by the script from the records; V(C) = Tr[C A (1 - C) A], V(C_0) = Tr[C_0 A (1 - C_0) A], dV = V(C) - V(C_0),",
          "  V(P) = Tr[P A (1 - P) A], the Wick product = -Tr(P A P A), A = m(R_surf); E_x^W = -(1/2) sum V Tr(P hcal P hcal), V = -g/max(R, R')."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
