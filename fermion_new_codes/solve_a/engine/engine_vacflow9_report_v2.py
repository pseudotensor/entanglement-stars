#!/usr/bin/env python
"""engine_vacflow9_report_v2.py (builder analyst; version 2 = version 1 with: the cuts at which the record holds a smoothed-cut fluence
(version 1 found none at r_s: the --smooth-window about the r_s cut does not fit inside the chain), the verdict rule's share printed at every
cut that has one, and per channel the unweighted N_p and N_h at the last readout. Version 1's docstring: engine spec 19.4, J-VF's decisive rerun on engine_vacflow_v9): the record as tables, with
the channel sums rebuilt here. The record's own sums are empty: engine_vacflow_v9 takes the sums' time axis from the first channel and leaves
out every channel whose readout count differs, and its first channels (kappa 1 to 4) stopped part-way through the ramp (R2's continuation
ambiguous after 6 halvings, the degeneracy-aware block rule of engine_collapse_v6), so every channel was left out and the record's verdict
is null. Here: (1) the stopped channels apart (the reason, the last readout's s and t/r_s, the outflow through r_s up to there); (2) the sums
(weights 16 kappa) over the complete channels only (the full readout count, not rejected), with engine_vacflow_v9's own keys and rule: the
energy inside and outside R = r_s under NO, R2 and raw, N_p and N_h, every 2 r_s and at the ramp's end and the hold's end; per cut and phase
the outflow fluences (-I integrated by the record's trapezoid), the six window bands, the cross-band remainder, the work parts W_C and W_R0,
the Killing-energy changes, the balance's closure and the reference's own term, and the smoothed cut's total and bands; (3) per channel at
r_s the ramp's fluence, its middle-band share on the sharp and on the smoothed cut; (4) the verdict by engine_vacflow_v9's rule (the two
middle bands' share of the smoothed cut's outflow over the ramp at r_s; "rides" only above 0.9) on the complete channels' sums, labelled
with the channels it covers. Every number is read from the record. Args: record --md."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    d = json.load(open(a.record))
    args = d["args"]; rs = float(args["rs"]); bands = d["bands"]; chans = d["channels"]
    nmax = max(len(c["reads"]) for c in chans)
    full = [c for c in chans if not c.get("rejected") and len(c["reads"]) == nmax]
    stopped = [c for c in chans if c not in full]
    L = ["## J-VF on engine_vacflow_v9 at r_s %g (t_r %g r_s, hold %g r_s, kappa 1 to %d, %s current, the %s profile, m %g; engine_vacflow9_report_v1.py over %s)" % (
        rs, args["tr"], args["hold"], d.get("kappa_done", args["kmax"]), args["chain_current"], args["profile"], args["m"], a.record.split("/")[-1]), "",
         "The record's own channel sums are empty (its time axis came from kappa 1, which stopped early, so every channel was left out) and its verdict is null; the sums below are rebuilt over the %d complete channels (kappa %s) with the record's keys and rule." % (
             len(full), ",".join(str(c["kappa"]) for c in full)), "",
         "### (1) The stopped channels", "", "| kappa | weight | readouts | last s | last t/r_s | outflow through r_s to there | why |", "|---|---|---|---|---|---|---|"]
    key = "%g" % rs
    for c in stopped:
        last = c["reads"][-1]
        fr = (c.get("fluences") or {}).get(key, {}).get("ramp", {})
        L.append("| %d | %g | %d | %.6f | %.3f | %s | %s |" % (c["kappa"], c.get("weight", 16 * c["kappa"]), len(c["reads"]), last["s"], last["t_over_rs"],
                                                       ("%.4e" % fr["total"]) if "total" in fr else "-", (c.get("rejected") or "readout count %d" % len(c["reads"])).replace("|", "/")))
    t = np.array([q["t_over_rs"] for q in full[0]["reads"]])
    ph = [q["phase"] for q in full[0]["reads"]]
    S = {k: np.zeros(len(t)) for k in ("E_in_NO", "E_out_NO", "E_in_R2", "E_out_R2", "E_in_raw", "E_out_raw", "N_p", "N_h")}
    F = {}
    for c in full:
        w = 16.0 * c["kappa"]
        for k in S:
            S[k] += w * np.array([q.get(k, np.nan) for q in c["reads"]])
        for ck, f in c["fluences"].items():
            T_ = F.setdefault(ck, {})
            for phase, res in f.items():
                P_ = T_.setdefault(phase, {"total": 0.0, "bands": [0.0] * len(bands), "cross": 0.0, "window_total": 0.0, "work": 0.0, "smooth_bands": [0.0] * len(bands)})
                P_["total"] += w * res["total"]; P_["work"] += w * res["work"]
                if "bands" in res:
                    P_["bands"] = [x + w * y for x, y in zip(P_["bands"], res["bands"])]
                    P_["cross"] += w * res["cross"]; P_["window_total"] += w * res["window_total"]
                for k in ("total_C", "total_R2", "work_C", "work_R0", "dE_K_C", "dE_K_R0", "balance_C", "reference_own", "smooth_total_NO", "smooth_total_C", "smooth_total_R2", "smooth_cross"):
                    if k in res:
                        P_[k] = P_.get(k, 0.0) + w * res[k]
                if "smooth_bands" in res:
                    P_["smooth_bands"] = [x + w * y for x, y in zip(P_["smooth_bands"], res["smooth_bands"])]
    L += ["", "### (2) The sums over the complete channels: the energy inside and outside R = r_s, N_p and N_h", "",
          "| t/r_s | phase | E_in NO | E_out NO | E_in R2 | E_out R2 | E_in raw | E_out raw | N_p (weighted) | N_h (weighted) |", "|---|---|---|---|---|---|---|---|---|---|"]
    sel = [i for i in range(len(t)) if abs(t[i] / 2.0 - round(t[i] / 2.0)) < 1e-9 or i == len(t) - 1 or (ph[i] == "ramp" and i + 1 < len(ph) and ph[i + 1] == "hold")]
    for i in sorted(set(sel)):
        L.append("| %.2f | %s | %.6e | %.6e | %.6e | %.6e | %.6e | %.6e | %.4e | %.4e |" % (t[i], ph[i], S["E_in_NO"][i], S["E_out_NO"][i], S["E_in_R2"][i], S["E_out_R2"][i],
                                                                                     S["E_in_raw"][i], S["E_out_raw"][i], S["N_p"][i], S["N_h"][i]))
    L += ["", "### (2b) The outflow fluences through the cuts (sums over the complete channels; positive = energy leaving the inside)", ""]
    for ck in sorted(F, key=float):
        for phase, r in F[ck].items():
            b = r["bands"]; rev = b[1] + b[4]
            line = "- cut %s (%s): total %.6e; window %.6e = band-diagonal %.6e + cross %.6e; bands %s; cos 2p < 0 %.6e (share of the window %s)" % (
                ck, phase, r["total"], r["window_total"], sum(b), r["cross"], "; ".join("%s %.4e" % (n, x) for n, x in zip(bands, b)), rev,
                ("%.4f" % (rev / r["window_total"])) if r["window_total"] else "-")
            if "smooth_total_NO" in r:
                bs = r["smooth_bands"]; rs_ = bs[1] + bs[4]
                line += "; smoothed cut: total %.6e (C %.6e, R2 %.6e), bands %s, cross %.4e, middle bands %.6e (share %s)" % (
                    r["smooth_total_NO"], r.get("smooth_total_C", float("nan")), r.get("smooth_total_R2", float("nan")), "; ".join("%.4e" % x for x in bs), r.get("smooth_cross", float("nan")),
                    rs_, ("%.4f" % (rs_ / r["smooth_total_NO"])) if r["smooth_total_NO"] else "-")
            if "work_C" in r:
                line += "; work inside %.6e (W_C %.6e, W_R0 %.6e); Delta E_K: C %.6e, R0 %.6e; the balance's closure (C) %.3e; the reference's own term %.4e" % (
                    r["work"], r["work_C"], r["work_R0"], r["dE_K_C"], r["dE_K_R0"], r["balance_C"], r["reference_own"])
            L.append(line + ".")
    L += ["", "### (3) Per channel at R = r_s (the ramp)", "", "| kappa | complete | outflow fluence | sharp middle-band share | smoothed total | smoothed middle-band share | work inside | N_p, N_h at the last readout (unweighted) |", "|---|---|---|---|---|---|---|---|"]
    for c in chans:
        fr = (c.get("fluences") or {}).get(key, {}).get("ramp", {})
        b = fr.get("bands"); bs = fr.get("smooth_bands")
        L.append("| %d | %s | %s | %s | %s | %s | %s | %.4f, %.4f |" % (
            c["kappa"], "yes" if c in full else "no (stopped at t/r_s %.3f)" % c["reads"][-1]["t_over_rs"], ("%.4e" % fr["total"]) if "total" in fr else "-",
            ("%.4f" % ((b[1] + b[4]) / fr["window_total"])) if (b and fr.get("window_total")) else "-", ("%.4e" % fr["smooth_total_NO"]) if "smooth_total_NO" in fr else "-",
            ("%.4f" % ((bs[1] + bs[4]) / fr["smooth_total_NO"])) if (bs and fr.get("smooth_total_NO")) else "-", ("%.4e" % fr["work"]) if "work" in fr else "-",
            c["reads"][-1]["N_p"], c["reads"][-1]["N_h"]))
    fr = F.get(key, {}).get("ramp")
    L += ["", "### (4) The verdict by engine_vacflow_v9's rule on the complete channels' sums", ""]
    if fr and "smooth_total_NO" in fr:
        bs = fr["smooth_bands"]; share = (bs[1] + bs[4]) / fr["smooth_total_NO"] if fr["smooth_total_NO"] else None
        L.append("- kappa %s (the complete channels; kappa %s stopped and are not in it): the smoothed cut's outflow through r_s over the ramp %.6e, its two middle bands %.6e, share %s: %s (the rule: 'rides' only above 0.9)." % (
            ",".join(str(c["kappa"]) for c in full), ",".join(str(c["kappa"]) for c in stopped) or "none", fr["smooth_total_NO"], bs[1] + bs[4],
            ("%.4f" % share) if share is not None else "-", "rides the cos 2p < 0 bands" if (share is not None and share > 0.9) else "does not ride them"))
    else:
        L.append("- no smoothed-cut fluence at r_s in any channel of the record (the --smooth-window of %s sites about the r_s cut does not fit inside the chain), so the rule cannot be applied at r_s." % args.get("smooth_window"))
    have = sorted({ck for c in chans for ck, f in (c.get("fluences") or {}).items() if "smooth_total_NO" in f.get("ramp", {})}, key=float)
    L.append("- the cuts with a smoothed-cut fluence in the record: %s." % (", ".join("R = %s (%.2f r_s)" % (ck, float(ck) / rs) for ck in have) or "none"))
    for ck in have:
        r = F.get(ck, {}).get("ramp")
        if r and r.get("smooth_total_NO"):
            bs = r["smooth_bands"]
            L.append("- the rule's share at R = %s over the ramp (the complete channels): middle bands %.6e of the smoothed outflow %.6e, share %.4f: %s." % (
                ck, bs[1] + bs[4], r["smooth_total_NO"], (bs[1] + bs[4]) / r["smooth_total_NO"], "above 0.9" if (bs[1] + bs[4]) / r["smooth_total_NO"] > 0.9 else "not above 0.9"))
    L += ["", "- every number above is read from the record; the fluences are the record's per-channel trapezoids over the readouts every %g r_s, summed here with the weights 16 kappa." % args["every"]]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
