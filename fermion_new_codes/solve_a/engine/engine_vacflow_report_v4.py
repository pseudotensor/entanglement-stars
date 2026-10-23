#!/usr/bin/env python3
"""engine_vacflow_report_v4.py (builder analyst; J-VF's post-processing, version 4 of engine_vacflow_report_v1.py, after the J-VF red team
(reviewb redteam_ENG15b of 2026-10-15) and the lead's turn-23 plan (engine sec. 15.2), which version 3 did not follow in one point: the log fit of
E(R_out) over R_out in [1.5, 6] r_s; versions 1 to 3 printed only the record's own fit over [R_read/8, R_read] = [2, 16] r_s).

From each record's site totals (the weighted shell sums per site; engine_vacflow_v2 and v3 records), for every ramp and key (raw, NO, R2 when
present; the hold reads): (1) E(R_out) = the cumulative shell sum on the record's own R_out grid (24 points geometric from 1.5 r_s to
R_read) and the log-slope fits c over [1.5, 6] r_s (15.2's window) and over [2, 16] r_s (the record's), with the fit residual's rms and the
ratio |c_NO/c_raw| (VF2's criterion, below 1 percent) and |c_R2/c_raw|; (2) the annular energies E(R_2) - E(R_1) between the probes
{2, 4, 8, 16} r_s (beyond every channel's truncated edge kappa/E_cut for kappa <= 2 r_s E_cut: the edge offset the red team identified cancels in
them); (3) with --flag-kappa K (the first channel flagged by the lead's rule, engine_vacflow_v4's measurement): E_raw and E_NO at the probes
summed over the channels below K only (from the running sums) and the flagged channels' own total beside it, the rule applied. Args:
records... --flag-kappa --md."""
import argparse
import json

import numpy as np


def fit(R, E, lo, hi):
    R = np.asarray(R, float); E = np.asarray(E, float)
    sel = (R >= lo * (1 - 1e-9)) & (R <= hi * (1 + 1e-9))
    if sel.sum() < 3:
        return None, None, int(sel.sum())
    c = np.polyfit(np.log(R[sel]), E[sel], 1)
    res = E[sel] - np.polyval(c, np.log(R[sel]))
    return float(c[0]), float(np.sqrt(np.mean(res ** 2))), int(sel.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--flag-kappa", type=int, default=None)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## J-VF post-processing, version 4 (engine_vacflow_report_v4.py over %s)" % ", ".join(x.split("/")[-1] for x in a.records), ""]
    for f in a.records:
        D = json.load(open(f))
        args = D["args"]; rs = args["rs"]
        R_read = 16 * rs
        Rout = np.geomspace(1.5 * rs, R_read, 24)
        L += ["### %s: r_s %g, profile %s, m %g, E_cut %g, channels done %s" % (f.split("/")[-1], rs, args["profile"], args["m"], args["Ecut"], D.get("kappa_done")), "",
              "(1) the log-slope fits of E(R_out) (hold reads) over 15.2's window [1.5, 6] r_s and over the record's [2, 16] r_s", "",
              "| t_r/r_s | key | c over [1.5, 6] r_s (rms residual; points) | c over [2, 16] r_s (rms; points) | |c/c_raw| over [1.5, 6] | |c/c_raw| over [2, 16] | E(R_read) |",
              "|---|---|---|---|---|---|---|"]
        annular = []
        for tr, d in D["totals"].items():
            keys = [k for k in ("raw_hold", "NO_hold", "R2_hold") if k in d]
            fits = {}
            for key in keys:
                cum = np.cumsum(np.asarray(d[key], float))
                E = np.array([cum[int(r) - 1] for r in Rout])
                fits[key] = (fit(Rout, E, 1.5 * rs, 6 * rs), fit(Rout, E, 2 * rs, 16 * rs), float(E[-1]), cum)
            for key in keys:
                (c1, r1, n1), (c2, r2, n2), Er, cum = fits[key]
                cr1 = fits["raw_hold"][0][0]; cr2 = fits["raw_hold"][1][0]
                L.append("| %s | %s | %.4e (%.1e; %d) | %.4e (%.1e; %d) | %s | %s | %.4e |" % (tr, key.split("_")[0], c1, r1, n1, c2, r2, n2,
                         ("%.4f" % abs(c1 / cr1)) if (key != "raw_hold" and cr1) else "-", ("%.4f" % abs(c2 / cr2)) if (key != "raw_hold" and cr2) else "-", Er))
                pr = [2 * rs, 4 * rs, 8 * rs, 16 * rs]
                ann = [float(cum[int(pr[i + 1]) - 1] - cum[int(pr[i]) - 1]) for i in range(len(pr) - 1)]
                annular.append((tr, key, ann))
        L += ["", "(2) the annular energies E(R_2) - E(R_1) between the probes (the truncated edges' offset cancels beyond the edges)", "",
              "| t_r/r_s | key | [2, 4] r_s | [4, 8] r_s | [8, 16] r_s |", "|---|---|---|---|---|"]
        for tr, key, ann in annular:
            L.append("| %s | %s | %.4e | %.4e | %.4e |" % (tr, key.split("_")[0], *ann))
        if a.flag_kappa:
            L += ["", "(3) the lead's rule applied: the channels from kappa = %d on removed from the sums (the running sums at the probes 1.5, 2, 4, 8, 16 r_s)" % a.flag_kappa, "",
                  "| t_r/r_s | E_raw below the flag | the flagged channels' raw total | E_NO below the flag | the flagged channels' NO total |", "|---|---|---|---|---|"]
            for tr, rows in D["running"].items():
                ks = [r["kappa"] for r in rows]
                if a.flag_kappa not in ks:
                    L.append("| %s | the record ends at kappa = %d, below the flag | | | |" % (tr, ks[-1]))
                    continue
                i0 = ks.index(a.flag_kappa)
                below_r = rows[i0 - 1]["E_raw_at"] if i0 > 0 else [0.0] * 5
                below_n = rows[i0 - 1]["E_NO_at"] if i0 > 0 else [0.0] * 5
                fl_r = [x - y for x, y in zip(rows[-1]["E_raw_at"], below_r)]
                fl_n = [x - y for x, y in zip(rows[-1]["E_NO_at"], below_n)]
                L.append("| %s | %s | %s | %s | %s |" % (tr, ", ".join("%.4e" % x for x in below_r), ", ".join("%.4e" % x for x in fl_r),
                         ", ".join("%.4e" % x for x in below_n), ", ".join("%.3e" % x for x in fl_n)))
        L.append("")
    L += ["- every number above is printed by the script from the records' site totals and running sums; VF2's criterion (|c_NO/c_raw| below 1 percent)",
          "  is the lead's; the edge offset and the window are the red team's points (engine 15.2 red team, items 4 to 6)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
