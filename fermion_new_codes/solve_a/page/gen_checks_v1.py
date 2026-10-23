#!/usr/bin/env python
"""gen_checks_v1.py (solve builder 6, analyst; engine spec 47.1 (2) (ii) to (iv)): light parsing of a page_history summary json (v6j/v6k/v6l),
no computation beyond arithmetic on stored numbers. Per checkpoint and channel: (ii) the archive's Gram deviation ||F^dag F - I||_F
(archive_gram_dev_F, criterion 1e-10 per block) and the all-history Gram (gram_all_history_F, printed beside the production's); (iii) the
energy identity residual (criterion 1e-6) and the collar identity residual, which are algebraically one number (web40g item 2), the two columns
that "differ in 39 of 40 entries" in read_v6j_gen_acceptance_v1 named (archive_gram_dev_F and energy_identity_residual: one is ||F^dag F - I||_F,
so nothing is to be found there); (iv) the archive size per channel, S_col per channel against the bound size x ln 2, S_vac(archive) beside, the
incoming fraction. Usage: gen_checks_v1.py SUMMARY.json --label NAME --md OUT.md --json OUT.json [--prod PROD_SUMMARY.json]"""
import argparse
import json
import math


def rows_of(p):
    d = json.load(open(p))
    return d, [r for r in d["rows"] if r.get("collect")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary"); ap.add_argument("--label", required=True); ap.add_argument("--md", required=True); ap.add_argument("--json", required=True)
    ap.add_argument("--prod")
    a = ap.parse_args()
    d, rows = rows_of(a.summary)
    prod = {}
    if a.prod:
        _, pr = rows_of(a.prod)
        prod = {round(r["t_over_rs"], 6): r for r in pr}
    st = d.get("settings", {})
    out = {"summary": a.summary, "label": a.label, "settings": {k: st.get(k) for k in ("collector_select", "collector_admit", "collector_mode", "eps_loc", "R_cut_rs", "channels", "cubic_window")},
           "rows": [], "fail": []}
    L = ["# %s: the corrected selector's checks (ii) to (iv) (engine spec 47.1 (2))" % a.label, "",
         "From %s (settings: %s); script code/page/gen_checks_v1.py; every number is the summary's or arithmetic on it." % (a.summary, json.dumps(out["settings"])), "",
         "| T/r_s | ch | archive | collected | S_col | size x ln 2 | S_col/bound | S_vac(archive) | ||F^dag F - I||_F | all-history Gram | prod. all-history Gram | energy id. | collar id. | incoming fraction | joint outside weight |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        T = r["t_over_rs"]
        for ch, x in sorted(r["per_channel"].items()):
            A = x.get("archive") or 0; S = x.get("S_esc") or 0.0; bnd = A * math.log(2.0)
            gd = x.get("archive_gram_dev_F"); gh = x.get("gram_all_history_F"); ei = x.get("energy_identity_residual"); ci = x.get("collar_identity_residual")
            pg = prod.get(round(T, 6), {}).get("per_channel", {}).get(ch, {}).get("gram_all_history_F") if prod else None
            row = {"T": T, "ch": ch, "archive": A, "collected": x.get("collected"), "S_col": S, "bound": bnd, "S_vac": x.get("S_vac_archive"), "gram_dev": gd,
                   "gram_hist": gh, "gram_hist_prod": pg, "energy_id": ei, "collar_id": ci, "incoming_fraction": x.get("incoming_fraction"),
                   "joint_outside_weight": x.get("joint_outside_weight")}
            out["rows"].append(row)
            if gd is not None and gd > 1e-10:
                out["fail"].append("(ii) %g ch %s gram_dev %.3e" % (T, ch, gd))
            if ei is not None and abs(ei) > 1e-6:
                out["fail"].append("(iii) %g ch %s energy identity %.3e" % (T, ch, ei))
            if S > bnd + 1e-12:
                out["fail"].append("(iv) %g ch %s S_col %.6g above size x ln 2 %.6g" % (T, ch, S, bnd))
            f = lambda v, fm: (fm % v) if isinstance(v, (int, float)) else "-"
            L.append("| %.3f | %s | %d | %s | %s | %.4g | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                T, ch, A, x.get("collected"), f(S, "%.6g"), bnd, f(S / bnd if bnd else None, "%.3g"), f(x.get("S_vac_archive"), "%.5g"), f(gd, "%.2e"), f(gh, "%.2e"),
                f(pg, "%.2e"), f(ei, "%.2e"), f(ci, "%.2e"), f(x.get("incoming_fraction"), "%.4g"), f(x.get("joint_outside_weight"), "%.6g")))
    same = sum(1 for r in out["rows"] if r["energy_id"] is not None and r["collar_id"] is not None and abs(r["energy_id"] - r["collar_id"]) <= 1e-14 + 1e-3 * abs(r["energy_id"]))
    gmax = max([r["gram_dev"] for r in out["rows"] if r["gram_dev"] is not None] or [0.0])
    emax = max([abs(r["energy_id"]) for r in out["rows"] if r["energy_id"] is not None] or [0.0])
    rmax = max([r["S_col"] / r["bound"] for r in out["rows"] if r["bound"] > 0] or [0.0])
    out["verdict"] = {"ii_gram_dev_max": gmax, "iii_energy_identity_max_abs": emax, "iii_energy_equals_collar_rows": same, "rows": len(out["rows"]),
                      "iv_S_over_bound_max": rmax, "pass": not out["fail"]}
    L += ["", "Verdict: (ii) ||F^dag F - I||_F at most %.2e (criterion 1e-10 per block); (iii) the energy identity at most %.2e in absolute value (criterion 1e-6); the collar identity equals it to 1e-3 relative (floating rounding apart) in %d of %d channel rows (one number, web40g item 2); the 2 columns read_v6j_gen_acceptance_v1 compared are archive_gram_dev_F (||F^dag F - I||_F) and the energy identity residual, so their differing is no finding; (iv) S_col per channel at most %.3g of its archive size x ln 2. %s." % (
        gmax, emax, same, len(out["rows"]), rmax, "PASS" if not out["fail"] else "FAIL: " + "; ".join(out["fail"][:6])),
          "", "Not here: the Z identity (the pilot code's direct route; page_history prints none): read on the regression under gen, 47.1 (2) (vii)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.json, "w"), indent=1)
    print(L[-3])


if __name__ == "__main__":
    main()
