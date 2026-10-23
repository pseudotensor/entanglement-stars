#!/usr/bin/env python
"""item3_checks.py (busi readouts #235, item 3, v2): the second reader's two pending checks and the qualifications.
(a) Where the successive-difference ratios >= 1 sit in each record's iteration_history (index k of q_k = d_k/d_(k-1)); entry 0 is the
    lagged first iterate (engine_collapse_v27 lines 1345-1352: the solve at the previous record's applied fields with the previous
    record's reference, no eigendecomposition; its first field is None), entries k >= 1 are the plain map's iterates (lines 1355-1384).
    The ratios are split into q_1 (lagged entry in the denominator) and q_k, k >= 2 (both differences of the same map).
(b) Uninterrupted trails: per flagged record, history length = iterations, entry 0 the only lagged one, converged, not unconverged;
    the run's resume_from (one run from T = 0, no restart).
Also: n_pocket per record around the flagged ones, the per-record geometric-mean rate q_geo (d_last/d_first)^(1/(n-1)) and its maximum,
the d_last and last |dE/E| against the engine's stop rule (stop_E, stop_fields from the json args).
Usage: python item3_checks.py <json> [<json> ...]
"""
import json
import sys

import numpy as np


def run(fj):
    j = json.load(open(fj)); a = j["args"]; R = j["records"]
    print("== %s" % fj.split("/")[-1])
    print("script %s; mix %s; maxit %s; stop_E %s; stop_fields %s; resume_from %s; resume_T %s; records %d (T/rs %.3f..%.3f)" % (
        j.get("script"), a.get("mix"), a.get("maxit"), a.get("stop_E"), a.get("stop_fields"), a.get("resume_from"), a.get("resume_T_over_rs"),
        len(R), R[0]["T_over_rs"], R[-1]["T_over_rs"]))
    npk = {round(r["T_over_rs"], 6): sum(int(v) for v in (r.get("n_pocket") or {}).values()) for r in R}
    q1 = []; qk = []; flagged = []; qgeo = []; qgeo2 = []; dlast = []; dElast = []; bad_struct = []
    for r in R:
        h = r.get("iteration_history") or []
        if len(h) < 2:
            continue
        d = np.array([max(e[1], e[2]) for e in h], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = d[1:] / d[:-1]
        q1.append(q[0]); qk += list(q[1:])
        qgeo.append((d[-1] / d[0]) ** (1 / (len(d) - 1)) if d[0] > 0 else np.nan)
        if len(d) > 2:
            qgeo2.append((d[-1] / d[1]) ** (1 / (len(d) - 2)))
        dlast.append(d[-1]); dElast.append(h[-1][0])
        struct_ok = (len(h) == r.get("iterations")) and h[0][0] is None and all(e[0] is not None for e in h[1:])
        if not struct_ok:
            bad_struct.append(r["T_over_rs"])
        if (q >= 1).any():
            flagged.append((r, d, q, struct_ok))
    q1 = np.array(q1); qk = np.array(qk)
    print("ratios q_1 (lagged entry 0 in the denominator): %d, >= 1: %d, max %.3f" % (len(q1), int((q1 >= 1).sum()), np.nanmax(q1)))
    print("ratios q_k, k >= 2 (same plain map): %d, >= 1: %d, max %.3f" % (len(qk), int((qk >= 1).sum()), qk.max() if len(qk) else float("nan")))
    print("records whose history structure is not [lagged, plain x (iterations-1)]: %s" % (bad_struct or "none"))
    print("q_geo = (d_last/d_first)^(1/(n-1)) per record: median %.3f, max %.3f; from entry 1 (plain map only): median %.3f, max %.3f" % (
        np.nanmedian(qgeo), np.nanmax(qgeo), np.nanmedian(qgeo2), np.nanmax(qgeo2)))
    print("d_last (max(max|dN|, max|dv|) at the last iterate): max %.3e against stop_fields %s; last |dE/E|: max %.3e against stop_E %s; converged %d of %d" % (
        max(dlast), a.get("stop_fields"), max(dElast), a.get("stop_E"), sum(1 for r in R if r.get("converged")), len(R)))
    print("flagged records (some q >= 1):")
    for r, d, q, ok in flagged:
        T = r["T_over_rs"]
        ks = [k + 1 for k in range(len(q)) if q[k] >= 1]
        print("  T/rs %.3f: its %d, converged %s, unconverged %s, structure ok %s; q >= 1 at k = %s (values %s); n_pocket here %d (previous record %s, next %s)" % (
            T, r.get("iterations"), r.get("converged"), r.get("unconverged"), ok, ks, " ".join("%.3f" % q[k - 1] for k in ks),
            npk[round(T, 6)], npk.get(round(T - 0.125, 6)), npk.get(round(T + 0.125, 6))))
        print("      d_k: %s" % " ".join("%.2e" % x for x in d))
        print("      q_k: %s" % " ".join("%.3f" % x for x in q))
    print("n_pocket (summed over channels) per record 12.500..13.625: %s" % ", ".join(
        "%.3f:%d" % (T, n) for T, n in sorted(npk.items()) if 12.5 - 1e-9 <= T <= 13.625 + 1e-9))
    first = [T for T, n in sorted(npk.items()) if n > 0]
    print("first record with n_pocket > 0: %s; records with n_pocket > 0: %d" % (first[0] if first else None, len(first)))
    print()


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
