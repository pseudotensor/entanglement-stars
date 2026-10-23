#!/usr/bin/env python3
"""engine_vacflow_report_v3.py (builder analyst; J-VF's post-processing per SPEC_final4b_engine.md sec. 15.2; version 3 of
engine_vacflow_report_v1.py: version 2 read the record's summary under wrong key names (c, E_at_read; the records carry log_slope_c and
E_of_Rout), so its item (6) printed dashes; nothing else changes). Per record (engine_vacflow_v2 or v3) and ramp t_r: (1) the running sums E_raw, E_NO (and E_R2 when present)
at the probe radii {1.5, 2, 4, 8, 16} r_s at channel checkpoints; (2) the per-channel weighted increments dE(kappa) = E(kappa) - E(kappa - 1)
at every probe: E_NO's largest |dE| beyond kappa = 8 against 1e-4 |E_NO| (the lead's expectation), E_raw's last increments; (3) E_raw's tail:
a power law |dE_raw| = A kappa^p fitted over the last 64 channels at each probe, the sum of the fitted increments from the last channel done
to kappa_cap = 2 R_read extrapolated and labelled as such (never added to the measured sum); (4) the E_cut flag (engine_vacflow_v4's
measurement at the probe radii, --flags): the first flagged channel and, from it on, the bound sum_{kappa >= kappa_flag} |dE_raw(kappa)| at
each probe, with E_raw summed below kappa_flag beside it; (5) the R2 column at the last channel (E_raw, E_NO, E_R2 and E_NO - E_R2, R0's
refilling) and the continuation's halvings; (6) the record's own log-slope summary when present (version 3's). Args: records... --flags
<json> --md."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--flags", default=None)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    flag = None
    if a.flags:
        F = json.load(open(a.flags))
        flagged = sorted({c["kappa"] for c in F["channels"] for tr, kk in c["ramps"].items() for row in kk["raw_hold"]["probes"] if row["flag"]})
        tested = sorted({c["kappa"] for c in F["channels"]})
        flag = {"first": (flagged[0] if flagged else None), "tested": tested, "flagged": flagged}
    L = ["## J-VF post-processing (engine_vacflow_report_v3.py over %s)" % ", ".join(x.split("/")[-1] for x in a.records), ""]
    if flag:
        L += ["E_cut flag (engine_vacflow_v4 at the probe radii, %s): channels tested %s, flagged %s; the bound applies from kappa = %s on." % (
            a.flags.split("/")[-1], flag["tested"], flag["flagged"], flag["first"]), ""]
    for f in a.records:
        D = json.load(open(f))
        args = D["args"]; rs = args["rs"]
        R_read = 16 * rs
        kcap = int(2 * R_read)
        probes = [p for p in (1.5 * rs, 2 * rs, 4 * rs, 8 * rs, 16 * rs)]
        L += ["### %s: r_s %g, profile %s, m %g, E_cut %g, channels done %s (cap %d)" % (f.split("/")[-1], rs, args["profile"], args["m"], args["Ecut"], D.get("kappa_done"), kcap), ""]
        for tr, rows in D["running"].items():
            ks = [r["kappa"] for r in rows]
            has_r2 = "E_R2_at" in rows[0]
            L += ["#### t_r = %s r_s" % tr, "", "(1) running sums at R_p/r_s = %s" % ", ".join("%g" % (p / rs) for p in probes), "",
                  "| channels | E_raw | E_NO |%s" % (" E_R2 |" if has_r2 else ""), "|---|---|---|%s" % ("---|" if has_r2 else "")]
            for kq in sorted(set([k for k in (4, 8, 16, 32, 64, 96, 128, 160, 192, 256, 384, 512) if k <= ks[-1]] + [ks[-1]])):
                if kq in ks:
                    r = rows[ks.index(kq)]
                    L.append("| %d | %s | %s |%s" % (kq, ", ".join("%.5e" % x for x in r["E_raw_at"]), ", ".join("%.5e" % x for x in r["E_NO_at"]),
                             (" %s |" % ", ".join("%.5e" % x for x in r["E_R2_at"])) if has_r2 else ""))
            Er = np.array([r["E_raw_at"] for r in rows]); En = np.array([r["E_NO_at"] for r in rows])
            kk = np.array(ks, float)
            dEr = np.diff(np.vstack([np.zeros(Er.shape[1]), Er]), axis=0); dEn = np.diff(np.vstack([np.zeros(En.shape[1]), En]), axis=0)
            sel8 = kk > 8
            if sel8.any():
                mx = np.max(np.abs(dEn[sel8]), axis=0); rel = mx / np.abs(En[-1])
                L += ["", "(2) E_NO's largest |increment| beyond kappa = 8 per probe: %s; over |E_NO| at the last channel: %s (the lead's expectation below 1e-4)" % (
                    ", ".join("%.2e" % x for x in mx), ", ".join("%.2e" % x for x in rel))]
                L += ["    E_raw's increments at the last 3 channels: %s" % "; ".join("kappa %d: %s" % (int(kk[i]), ", ".join("%.3e" % x for x in dEr[i])) for i in range(max(0, len(kk) - 3), len(kk)))]
            if len(kk) >= 32:
                nfit = min(64, len(kk) - 8)
                ks_ = kk[-nfit:]
                tail = []
                for j in range(Er.shape[1]):
                    y = np.abs(dEr[-nfit:, j])
                    ok = y > 0
                    if ok.sum() < 8:
                        tail.append(None); continue
                    p_, lA = np.polyfit(np.log(ks_[ok]), np.log(y[ok]), 1)
                    kext = np.arange(int(kk[-1]) + 1, kcap + 1, dtype=float)
                    sgn = np.sign(np.sum(dEr[-nfit:, j]))
                    ext = float(sgn * np.sum(np.exp(lA) * kext ** p_)) if len(kext) else 0.0
                    tail.append((p_, ext))
                L += ["", "(3) E_raw's tail: |dE_raw| = A kappa^p over the last %d channels (%d to %d); the fitted increments summed to kappa_cap = %d (extrapolated, labelled):" % (
                    nfit, int(ks_[0]), int(ks_[-1]), kcap)]
                L += ["    " + "; ".join(("R_p/r_s %g: p = %.3f, extrapolated tail %+.4e (measured E_raw %.4e)" % (probes[j] / rs, t[0], t[1], Er[-1, j])) if t else "R_p/r_s %g: -" % (probes[j] / rs)
                                         for j, t in enumerate(tail))]
            if flag and flag["first"] is not None and flag["first"] <= kk[-1]:
                i0 = int(np.searchsorted(kk, flag["first"]))
                bnd = np.sum(np.abs(dEr[i0:]), axis=0)
                below = Er[i0 - 1] if i0 > 0 else np.zeros(Er.shape[1])
                L += ["", "(4) the E_cut bound from kappa = %d on (%d channels): E_raw summed below it %s; the bound sum |dE_raw| %s" % (
                    flag["first"], len(kk) - i0, ", ".join("%.4e" % x for x in below), ", ".join("%.3e" % x for x in bnd))]
            if has_r2:
                r = rows[-1]
                L += ["", "(5) the R2 column at kappa = %d: E_raw %s; E_NO %s; E_R2 %s; E_NO - E_R2 (R0's refilling) %s; halvings %s" % (
                    r["kappa"], ", ".join("%.4e" % x for x in r["E_raw_at"]), ", ".join("%.4e" % x for x in r["E_NO_at"]), ", ".join("%.4e" % x for x in r["E_R2_at"]),
                    ", ".join("%.4e" % (x - y) for x, y in zip(r["E_NO_at"], r["E_R2_at"])), sum((q.get("r2") or {}).get("halvings", 0) for q in rows))]
            sm = (D.get("summary") or {}).get(tr)
            if sm:
                L += ["", "(6) the record's summary (the log-slope fit of E(R_out) over the last factor 8 below R_read): " + "; ".join("%s: log slope c %s, E(R_read) %s" % (
                    k, ("%.4e" % v["log_slope_c"]) if isinstance(v, dict) and v.get("log_slope_c") is not None else "-",
                    ("%.4e" % v["E_of_Rout"][-1]) if isinstance(v, dict) and v.get("E_of_Rout") else "-") for k, v in sm.items())]
            L.append("")
        pk = D.get("pockets") or []
        L += ["Pockets (n_neg(R0) - n_sea) per channel, nonzero ones: %s" % (", ".join("%d: %d" % tuple(q) for q in pk if q[1]) or "none"), ""]
    L += ["- every number above is printed by the script; the extrapolated tails are fits, not measurements, and never enter a measured sum."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
