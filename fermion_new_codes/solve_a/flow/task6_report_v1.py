#!/usr/bin/env python
"""Task 6 report (SPEC_task6_page_curve.md sec. 2; builder analyst): the tables of flow_page_v1.py's and flow_page_v2.py's per-channel records
(the formation state on the PG radial channels).  Per channel: the checks (Phi^H Phi - 1, the energy balance against the drive's work, the
P0 identities where run), the cut entropies S_A at R_c = r_s and 2 r_s (flat, formed, the excess) at every hold time, the growth fits; the
shells' mutual information and cross blocks at the shell times; per outgoing packet the occupation n at every hold time, the raw temperature
(greybody factor 1) and its plateau, the partner's norm identity, location and Killing content; the ray map's back-trace fits.
Args: --jsons (comma list) --json --md."""
import argparse
import json
import math


def f(v, d=4):
    if v is None:
        return "-"
    v = float(v)
    if not math.isfinite(v):
        return "-"
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    chans = []
    for p in a.jsons.split(","):
        d = json.load(open(p))
        for c in d["channels"]:
            c["_script"] = d["script"]
            c["_tag"] = p.split("/")[-1]
            chans.append(c)
    chans.sort(key=lambda c: (c["_script"], c["mass"], c["j"], c["kappa_j"], c["rs"], c["N"], c["dt"], c["t_r"]))
    L = ["# Task 6: the formation state on the flow chart, tables (task6_report_v1.py; prescribed flow (simpler version); PG radial; v declared; "
         "formation history declared)", ""]
    out = {"channels": []}
    for c in chans:
        kap = c["kappa"]
        holds = sorted(float(h) for h in c["S_A"]["formed"].keys())
        ck = c["checks"]
        L += ["## %s: m %s, j %s, kappa_j %s (r_s %s, N %d, t_r %s, dt %s)" % (c["_tag"], f(c["mass"]), f(c["j"]), f(c["kappa_j"]), f(c["rs"]), c["N"], f(c["t_r"]), f(c["dt"])), "",
              "- Checks: Phi^H Phi - 1 %s; E(t_r) - E(0) - W %s (W %s, relative %s); timing %s." % (
                  f(ck["PhiH_Phi_minus_1_max"], 2), f(ck["energy_balance"], 2), f(ck["work"], 5), f(ck["energy_balance_rel"], 2),
                  ", ".join("%s %.0f s" % (k, v) for k, v in c["timing_s"].items()))]
        if "p0" in c:
            p0 = c["p0"]
            L.append("- P0 at %s: S_A %s; from the singular values of G_{AA'} %s (difference %s); S_{A'} - S_A %s." % (
                p0["t"], f(p0["S_A_direct"], 10), f(p0["S_A_from_G_singular_values"], 10), f(p0["difference"], 2), f(-p0["S_A_minus_S_complement"], 2)))
        L += ["", "| t kappa | " + " | ".join("S_A(R_c = %s)" % k for k in c["S_A"]["flat"]) + " | " + " | ".join("excess (R_c = %s)" % k for k in c["S_A"]["flat"]) + " |",
              "|---|" + "---|" * (2 * len(c["S_A"]["flat"]))]
        L.append("| flat | " + " | ".join(f(v, 7) for v in c["S_A"]["flat"].values()) + " | " + " | ".join("0" for _ in c["S_A"]["flat"]) + " |")
        for h in holds:
            k = ("%g" % h) if ("%g" % h) in c["S_A"]["formed"] else "%.1f" % h
            k = k if k in c["S_A"]["formed"] else [x for x in c["S_A"]["formed"] if float(x) == h][0]
            L.append("| %s | " % f(h) + " | ".join(f(v, 7) for v in c["S_A"]["formed"][k].values()) + " | " + " | ".join(
                f(c["excess"][Rc][k], 5) for Rc in c["S_A"]["flat"]) + " |")
        L.append("")
        L.append("- Growth fits: " + "; ".join("R_c %s over %s/kappa: rate/kappa %s (rms %s, n %d)" % (
            k.replace("_late", ""), v.get("window_over_kappa", [12, 48]), f(v["rate_over_kappa"], 4), f(v["rms_residual"], 3), v["n"]) for k, v in c["growth_fit"].items()) + ".")
        for th, sh in c["shells"].items():
            L.append("- Shells at %s/kappa (d: I formed, I flat, excess I, ||Delta G||_HS^2): " % th + "; ".join(
                "%s: %s, %s, %s, %s" % (d_, f(v["formed"]["I"], 5), f(v["flat"]["I"], 5), f(v["excess"]["I"], 3), f(v["excess"]["dG_HS2"], 3)) for d_, v in sh.items()) + ".")
        L.append("")
        if c["packets"]:
            L += ["| omega/kappa | x_0 | sigma_x | packet Killing mean, spread | F_w(1) | n at the holds (t kappa: n) | theta_raw plateau (spread) |", "|---|---|---|---|---|---|---|"]
            for r in c["packets"]:
                if "skipped" in r:
                    L.append("| %s | skipped: %s | | | | | |" % (f(r["omega_over_kappa"]), r["skipped"]))
                    continue
                L.append("| %s | %s | %s | %s, %s | %s | %s | %s (%s) |" % (
                    f(r["omega_over_kappa"]), f(r["x0"], 5), f(r["sigma_x"], 4), f(r["killing_w"]["omega_mean"], 4), f(r["killing_w"]["omega_spread"], 3),
                    f(r["F_w(1)"], 4), "; ".join("%s: %s" % (f(float(k)), f(v["n"], 4)) for k, v in r["hold"].items()), f(r.get("theta_raw_plateau"), 4),
                    f(r.get("theta_raw_spread"), 3)))
            L.append("")
            for r in c["packets"]:
                for k, v in r.get("hold", {}).items():
                    pt = v.get("partner")
                    if pt:
                        kl = pt.get("killing", {})
                        L.append("- partner omega/kappa %s at %s/kappa: ||w_p||^2 - n(1 - n) %s; weights core %s, trapped %s, outside %s (fraction of n(1-n) "
                                 "trapped %s); Killing mean %s, weight at omega < 0 %s." % (
                                     f(r["omega_over_kappa"]), f(float(k)), f(pt["identity_error"], 2), f(pt["weight_core"], 3), f(pt["weight_trapped"], 3),
                                     f(pt["weight_outside"], 3), f(pt["fraction_trapped"], 3), f(kl.get("omega_mean"), 4), f(kl.get("weight_omega_neg"), 3)))
                rm = r.get("raymap")
                if rm:
                    tr = rm["trace"]
                    L.append("- ray map omega/kappa %s: back-trace to tau kappa = %s, x from %s to %s; window points %s; kappa_p/kappa from x %s, from k %s; "
                             "through the ramp: %s." % (f(r["omega_over_kappa"]), f(tr[-1]["tau"] * kap), f(tr[0]["x"], 5), f(tr[-1]["x"], 5), rm.get("n_points", 0),
                                                         f((rm.get("kappa_p_from_x") or float("nan")) / kap, 4), f((rm.get("kappa_p_from_k") or float("nan")) / kap, 4),
                                                         ("fit slope/kappa %s" % f(-rm["fit_through_ramp_ln_x_minus_rs"][0] / kap, 4)) if rm.get("fit_through_ramp_ln_x_minus_rs") else "no fit"))
            L.append("")
        out["channels"].append({"tag": c["_tag"], "script": c["_script"], "mass": c["mass"], "j": c["j"], "kappa_j": c["kappa_j"], "excess": c["excess"],
                                "growth_fit": c["growth_fit"]})
    L.append("- Every number above is read from the records by the script.")
    json.dump(out, open(a.json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L)[:5000])


if __name__ == "__main__":
    main()
