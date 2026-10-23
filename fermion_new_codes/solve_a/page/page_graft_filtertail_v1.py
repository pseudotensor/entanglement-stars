#!/usr/bin/env python
"""page_graft_filtertail_v1.py (solve builder 12; engine spec 54.2 (3)): the discarded filter tail bounded beside every row of a graft rows
json (page_graft_purity_v1 or page_graft_pullback_v2), light parsing only. The polynomial p is exactly local (page_graft_posbound_v1 prints the
reach within the window) and |p - f| <= eps_sup on the spectrum (the measured sup-norm error there, at sigma and at 2 sigma), so the row's
readout through D_sigma differs from the one through f(H_walled) - f(H_connected) by at most
  null readouts:   2 eps_sup ||O|| ||X||_F^2           (X the window's detector columns; ||X||_F^2 the row's X_window_norm2)
  purity readouts: 4 eps_sup ||O|| ||X||_F ||Z||_F      (Z the window rows of Delta0 W; the row's Z_window_norm)
with ||O|| = ||J_SS|| for L and the 32-site split weight for E_flux_out(32) (from the posbound json of the same hold and channel).
Usage: --rows <tag>_rows.json --posbound <hold>_posbound.json --out <prefix>. Writes <prefix>_filtertail.json and .md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True); ap.add_argument("--posbound", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.load(open(a.rows)); P = json.load(open(a.posbound))
    digit = P["digit"]
    out = {"rows_file": a.rows, "posbound_file": a.posbound, "digit": digit, "channels": {}}
    L_ = ["# The discarded filter tail beside every row (engine spec 54.2 (3))", "",
          "rows: %s; norms and sup-norm errors: %s; digit %.4e. Bound per row: null 2 eps ||O|| ||X||_F^2, purity 4 eps ||O|| ||X||_F ||Z||_F; "
          "eps the measured sup-norm error of the polynomial at sigma and at 2 sigma." % (a.rows, a.posbound, digit), ""]
    for k, ch in R["channels"].items():
        pc = P["channels"][k]; f = pc["filter"]
        e1, e2 = f["sup_error"], f["sup_error_2sigma_truncated_at_M"]
        oL, oE = pc["J_SS_norm"], pc["split_weight_32"]
        rows = []; mx = {}
        for r in ch["rows"]:
            n = r.get("null") or {}
            x2 = n.get("X_window_norm2")
            if x2 is None:
                continue
            e = {"t_over_rs": r.get("t_over_rs"), "X_window_norm2": x2}
            for lab, eps in (("sigma", e1), ("2sigma", e2)):
                e["null_L_" + lab] = 2 * eps * oL * x2; e["null_Eflux32_" + lab] = 2 * eps * oE * x2
                pu = r.get("purity")
                if pu and "Z_window_norm" in pu:
                    z = pu["Z_window_norm"]; x = x2 ** 0.5
                    e["pur_L_" + lab] = 4 * eps * oL * x * z; e["pur_Eflux32_" + lab] = 4 * eps * oE * x * z
            for kk, vv in e.items():
                if kk not in ("t_over_rs", "X_window_norm2"):
                    mx[kk] = max(mx.get(kk, 0.0), vv)
            rows.append(e)
        out["channels"][k] = {"eps_sup": e1, "eps_sup_2sigma": e2, "J_SS_norm": oL, "split_weight_32": oE, "rows": rows, "max": mx,
                              "all_under_digit": all(v < digit for v in mx.values())}
        L_.append("- kappa %s: %d rows from T/r_s %s to %s; eps_sup %.2e (2 sigma %.2e); largest bound per readout: %s; all under the digit: %s" % (
            k, len(rows), rows[0]["t_over_rs"] if rows else None, rows[-1]["t_over_rs"] if rows else None, e1, e2,
            ", ".join("%s %.2e" % (kk, vv) for kk, vv in sorted(mx.items())), out["channels"][k]["all_under_digit"]))
    json.dump(out, open(a.out + "_filtertail.json", "w"), indent=1)
    open(a.out + "_filtertail.md", "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
