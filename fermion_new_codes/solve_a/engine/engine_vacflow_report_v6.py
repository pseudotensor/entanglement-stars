#!/usr/bin/env python3
"""engine_vacflow_report_v6.py (builder analyst; version 6 of engine_vacflow_report_v1.py: version 5 plus --keys (the keys of tables (1)
and (2), default raw,NO,R2) and --skip-bl (table (3) left out), so a record whose raw and NO columns another report already prints can be
reported by its R2 column and its per-channel table alone; nothing else changes. Version 5's docstring follows.
engine_vacflow_report_v5.py (builder analyst; J-VF's post-processing per SPEC_final4b_engine.md sec. 16.6; version 5 of
engine_vacflow_report_v1.py). Version 4's tables (the log-slope windows, the annular energies, the flag applied) plus:
(1) the inside integrals E_in = sum over sites R < r_s of the stored site totals, per key (raw, NO, R2) and ramp, and their difference
    E_NO - E_R2 inside (R0's refilling restricted to the inside);
(2) the ell smoothing (ell = 4, 8, 16) of the stored site totals by engine_fields_v4's kernels (rho_ell(r) = sum_n K_0(r, R_n) e_n): the inside
    integral int_0^r_s 4 pi r^2 rho_ell dr and the exterior annuli [2, 4], [4, 8], [8, 16] r_s per ell (the goal's binding ell control of this
    quantity), beside the unsmoothed sums;
(3) both B_L normalizations against the measured raw annular energy per unit ln R: the literal fit rho = B_L v'^2 gives pi B_L r_s per unit
    ln R, the tensor form (B_L/2)[v'^2 + 2 v^2/R^2] gives (9 pi/2) B_L r_s (--BL, default the box's 0.019, labelled as a comparison of form);
(4) per channel, when the record carries them (engine_vacflow_v6): the weighted inside integrals, N_p = Tr[R0 (1 - R2)] and N_h = Tr[R2 (1 - R0)],
    the halvings, and the channel's annular energies from its running-sum increments (the restated flag: the annuli between probes beyond the
    edge carry no edge offset). Args: records... --BL --md."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_fields_v4 as FLD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--BL", type=float, default=0.019)
    ap.add_argument("--md", required=True)
    ap.add_argument("--keys", default="raw,NO,R2")
    ap.add_argument("--skip-bl", action="store_true")
    a = ap.parse_args()
    keys = [k + "_hold" for k in a.keys.split(",")]
    L = ["## J-VF post-processing, version 6 (engine_vacflow_report_v6.py over %s; keys %s%s)" % (", ".join(x.split("/")[-1] for x in a.records), a.keys, "; table (3) skipped" if a.skip_bl else ""), ""]
    for f in a.records:
        D = json.load(open(f))
        args = D["args"]; rs = args["rs"]
        L += ["### %s: r_s %g, profile %s, m %g, channels done %s%s" % (f.split("/")[-1], rs, args["profile"], args["m"], D.get("kappa_done"),
              ("; chain current %s" % args.get("chain_current")) if args.get("chain_current") else ""), ""]
        L += ["(1) inside integrals (sites R < r_s) of the stored totals at the hold, and (2) the ell smoothing: the inside integral and the annuli per ell", "",
              "| t_r/r_s | key | E_in (sites) | ell | E_in (smoothed) | annulus [2, 4] r_s | [4, 8] | [8, 16] |", "|---|---|---|---|---|---|---|---|"]
        for tr, d in D["totals"].items():
            for key in [k for k in ("raw_hold", "NO_hold", "R2_hold") if k in d and k in keys]:
                e = np.asarray(d[key], float)
                Rn = np.arange(1, len(e) + 1, dtype=float)
                Ein = float(e[Rn < rs].sum())
                first = True
                for ell in (4, 8, 16):
                    rr = np.arange(0.0, 16 * rs + 1e-9, ell / 8.0)
                    K0, _ = FLD.kernels(rr, Rn, ell)
                    rho = K0 @ e
                    dens = 4 * np.pi * rr ** 2 * rho
                    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(rr))])
                    at = lambda x: float(np.interp(x, rr, cum))
                    L.append("| %s | %s | %s | %d | %.4e | %.4e | %.4e | %.4e |" % (tr if first else "", key.split("_")[0] if first else "", ("%.4e" % Ein) if first else "",
                             ell, at(rs), at(4 * rs) - at(2 * rs), at(8 * rs) - at(4 * rs), at(16 * rs) - at(8 * rs)))
                    first = False
        if all(k in D["totals"].get("2", {}) for k in ("NO_hold", "R2_hold")):
            L.append("")
            for tr, d in D["totals"].items():
                eN = np.asarray(d["NO_hold"], float); e2 = np.asarray(d["R2_hold"], float)
                Rn = np.arange(1, len(eN) + 1, dtype=float)
                L.append("- t_r = %s r_s: E_NO - E_R2 inside (R < r_s) %.4e, outside %.4e" % (tr, float((eN - e2)[Rn < rs].sum()), float((eN - e2)[Rn >= rs].sum())))
        # (3) the B_L normalizations against the measured raw annuli per unit ln R
        if not a.skip_bl:
          L += ["", "(3) the raw energy per unit ln R in the annuli (sites) against the two normalizations at B_L = %g: literal pi B_L r_s = %.4f, tensor (9 pi/2) B_L r_s = %.4f (a comparison of form)" % (
            a.BL, np.pi * a.BL * rs, 4.5 * np.pi * a.BL * rs), "", "| t_r/r_s | [2, 4] r_s | [4, 8] | [8, 16] |", "|---|---|---|---|"]
        for tr, d in ([] if a.skip_bl else D["totals"].items()):
            cum = np.cumsum(np.asarray(d["raw_hold"], float))
            ann = [(cum[int(4 * rs) - 1] - cum[int(2 * rs) - 1]) / np.log(2), (cum[int(8 * rs) - 1] - cum[int(4 * rs) - 1]) / np.log(2),
                   (cum[int(16 * rs) - 1] - cum[int(8 * rs) - 1]) / np.log(2)]
            L.append("| %s | %.4f | %.4f | %.4f |" % (tr, *ann))
        # (4) per channel
        pc = D.get("per_channel", [])
        if pc and pc[0].get("inside_weighted") is not None:
            L += ["", "(4) per channel: the weighted inside integrals at the hold, the particle and hole counts, the halvings, and the annuli from the running-sum increments (t_r = first ramp)", "",
                  "| kappa | E_in raw | E_in NO | E_in R2 | N_p | N_h | halvings | annuli of raw ([2,4], [4,8], [8,16] r_s) |", "|---|---|---|---|---|---|---|---|"]
            tr0 = list(D["running"].keys())[0]
            run = D["running"][tr0]
            prev = None
            for i, c in enumerate(pc):
                ins = c["inside_weighted"].get(tr0, {})
                r2 = c.get("r2") or {}
                cur = run[i]["E_raw_at"] if i < len(run) else None
                inc = [x - y for x, y in zip(cur, prev)] if (cur and prev) else cur
                prev = cur
                ann = ("%.3e, %.3e, %.3e" % (inc[2] - inc[1], inc[3] - inc[2], inc[4] - inc[3])) if inc else "-"
                L.append("| %d | %.4e | %.4e | %s | %s | %s | %s | %s |" % (c["kappa"], ins.get("raw_hold", float("nan")), ins.get("NO_hold", float("nan")),
                         ("%.4e" % ins["R2_hold"]) if "R2_hold" in ins else "-", ("%.3e" % r2["N_p"]) if "N_p" in r2 else "-", ("%.3e" % r2["N_h"]) if "N_h" in r2 else "-",
                         r2.get("halvings", "-"), ann))
        L.append("")
    L += ["- every number above is printed by the script from the records; the probes are 1.5, 2, 4, 8, 16 r_s (the annuli use the last four)."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:50]))


if __name__ == "__main__":
    main()
