#!/usr/bin/env python
"""item3_iteration.py (busi readouts #235, item 3): the self-consistency iteration's convergence trail per record, from the run json's
per-record iteration histories.
Plain members (mix plain): the engine's iteration_history entry k is [|dE/E|, max|dN|, max|dv|] between fields_{k+1} and fields_k (entry 0
from the lagged first iterate); d_k = max(max|dN|, max|dv|) (the sup norm over both fields), q_k = d_k / d_{k-1} for k >= 1.
Newton-Krylov member (mix newton): the solve_v32 entry's line searches give per Newton step the fixed-point residual before the step and the
accepted trial's residual; the trail is residual_before of step 1, the accepted residuals after each step; the ratio of successive residuals
is printed; the kept and confirming residuals and the stop rule from the record.
Usage: python item3_iteration.py plain <json> <out_prefix> | newton <json> <out_prefix> [<other copies> ...]
"""
import json
import sys

import numpy as np


def plain(fj, outp):
    j = json.load(open(fj))
    rows = []
    for r in j["records"]:
        h = r.get("iteration_history") or []
        d = np.array([max(e[1], e[2]) for e in h], float)
        ok = d[:-1] > 0
        q = (d[1:][ok] / d[:-1][ok]) if len(d) > 1 else np.array([])  # ratios where the previous difference is nonzero
        rows.append({"T_over_rs": r["T_over_rs"], "iterations": r.get("iterations"), "n_hist": len(h), "converged": r.get("converged"),
                     "unconverged": r.get("unconverged"), "d": d.tolist(), "q": q.tolist(),
                     "q_median": float(np.median(q)) if len(q) else None, "q_max": float(q.max()) if len(q) else None,
                     "q_geo": float((d[-1] / d[0]) ** (1 / (len(d) - 1))) if len(d) > 1 and d[0] > 0 and d[-1] > 0 else None, "n_zero_d": int((d == 0).sum()),
                     "n_q_ge1": int((q >= 1).sum()), "d_last": float(d[-1]) if len(d) else None})
    json.dump(rows, open(outp + ".json", "w"), indent=0)
    L = ["T/rs     its conv  q_med   q_max   q_geo   n(q>=1)  d_first   d_last"]
    for x in rows:
        if x["q_median"] is None:
            L.append("%7.3f  %3s %s  (no ratio: %d history entries)" % (x["T_over_rs"], x["iterations"], x["converged"], x["n_hist"]))
            continue
        L.append("%7.3f  %3d %-5s %.3f  %7.3f  %.3f  %3d      %.2e  %.2e" % (x["T_over_rs"], x["iterations"], x["converged"], x["q_median"],
                                                                         x["q_max"], x["q_geo"], x["n_q_ge1"], x["d"][0], x["d_last"]))
    qs = [x for x in rows if x["q_median"] is not None]
    for x in qs:
        if x["q_geo"] is None: x["q_geo"] = float("nan")
    allq = np.concatenate([np.array(x["q"]) for x in qs])
    L.append("")
    L.append("SUMMARY %s: %d records (%d with >= 2 iterates); iterations median %d max %d; converged %d of %d; per-record median q: median %.3f, max %.3f; "
             "per-record max q: median %.3f, max %.3f; records with some q_k >= 1: %d; all q_k pooled: %d, median %.3f, share >= 1 %.3f; "
             "geometric-mean q per record: median %.3f max %.3f" % (
                 fj.split("/")[-1], len(rows), len(qs), int(np.median([x["iterations"] for x in rows])), max(x["iterations"] for x in rows),
                 sum(1 for x in rows if x["converged"]), len(rows), np.median([x["q_median"] for x in qs]), max(x["q_median"] for x in qs),
                 np.median([x["q_max"] for x in qs]), max(x["q_max"] for x in qs), sum(1 for x in qs if x["n_q_ge1"] > 0), len(allq),
                 np.median(allq), (allq >= 1).mean(), np.nanmedian([x["q_geo"] for x in qs]), np.nanmax([x["q_geo"] for x in qs])))
    L.append("records with an exactly zero successive difference (ratio undefined there): %d" % sum(1 for x in rows if x.get("n_zero_d")))
    for lo, hi in ((0, 12.75 - 1e-9), (12.75, 16.0), (16.0 + 1e-9, 1e9)):
        sub = [x for x in qs if lo <= x["T_over_rs"] <= hi]
        if sub:
            L.append("  window %.3g..%.3g: %d records; median q per record med %.3f max %.3f; max q per record med %.3f max %.3f; records with q>=1 %d; iterations med %d max %d; unconverged %d" % (
                sub[0]["T_over_rs"], sub[-1]["T_over_rs"], len(sub), np.median([x["q_median"] for x in sub]), max(x["q_median"] for x in sub),
                np.median([x["q_max"] for x in sub]), max(x["q_max"] for x in sub), sum(1 for x in sub if x["n_q_ge1"] > 0),
                int(np.median([x["iterations"] for x in sub])), max(x["iterations"] for x in sub), sum(1 for x in sub if not x["converged"])))
    open(outp + ".txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-4:]))


def newton(fj, outp, others):
    j = json.load(open(fj))
    rows = []
    for r in j["records"]:
        s = r.get("solve_v32") or {}
        sol = s.get("solves") or []
        trails = []
        for so in sol:
            ls = so.get("line_search") or []
            tr = ([ls[0]["residual_before"]] if ls else []) + [min(t[1] for t in st["trials"] if t[0] == st["accepted_step"]) if st.get("accepted_step") is not None else None for st in ls]
            trails.append({"name": so.get("name"), "status": so.get("status"), "steps": len(ls), "trail": tr,
                           "steps_taken": [st.get("accepted_step") for st in ls], "kept": so.get("kept_residual"),
                           "confirming": so.get("confirming_residual"), "meets": so.get("meets_stop_rule")})
        rows.append({"T_over_rs": r["T_over_rs"], "iterations": r.get("iterations"), "converged": r.get("converged"), "solves": trails,
                     "kept_from": s.get("kept_from"), "kept": s.get("kept_residual"), "confirming": s.get("confirming_residual"),
                     "meets": s.get("meets_stop_rule"), "continuation": s.get("continuation"), "E": r.get("E")})
    json.dump(rows, open(outp + ".json", "w"), indent=0)
    L = ["T/rs     its  solves  newton_steps  trail (fixed-point residual before step 1, after each accepted step)  kept  confirming  stop"]
    for x in rows:
        if not x["solves"]:
            L.append("%7.3f  %3s  (no Newton solve stored: %s)" % (x["T_over_rs"], x["iterations"], "plain record / T = 0" if x["iterations"] else ""))
            continue
        parts = []
        for so in x["solves"]:
            parts.append("%s[%s]: %s steps %s" % (so["name"], so["status"], " ".join("%.1e" % t for t in so["trail"] if t is not None),
                                                   ",".join("%g" % a for a in so["steps_taken"] if a is not None)))
        L.append("%7.3f  %3s  %d  %d  %s ; kept %.2e (%s) confirming %.2e meets %s" % (
            x["T_over_rs"], x["iterations"], len(x["solves"]), sum(so["steps"] for so in x["solves"]), " | ".join(parts),
            x["kept"] if x["kept"] is not None else float("nan"), x["kept_from"], x["confirming"] if x["confirming"] is not None else float("nan"), x["meets"]))
    rat = []
    for x in rows:
        for so in x["solves"]:
            t = [v for v in so["trail"] if v is not None]
            rat += [t[i + 1] / t[i] for i in range(len(t) - 1) if t[i] > 0]
    rat = np.array(rat)
    withs = [x for x in rows if x["solves"]]
    L.append("")
    L.append("SUMMARY %s: %d records, %d with a Newton solve; meets the stop rule %d; kept residual max %.2e; Newton steps per record median %d max %d; "
             "records needing more than the direct solve (continuation stages) %d; successive-residual ratios pooled %d: median %.3g, share >= 1 %.3f, max %.3g" % (
                 fj.split("/")[-1], len(rows), len(withs), sum(1 for x in withs if x["meets"]), max(x["kept"] for x in withs if x["kept"] is not None),
                 int(np.median([sum(so["steps"] for so in x["solves"]) for x in withs])), max(sum(so["steps"] for so in x["solves"]) for x in withs),
                 sum(1 for x in withs if len(x["solves"]) > 1), len(rat), np.median(rat), (rat >= 1).mean(), rat.max()))
    for p in others:
        o = json.load(open(p))["records"]
        mp = {round(r["T_over_rs"], 6): r["E"] for r in o}
        common = [x for x in rows if round(x["T_over_rs"], 6) in mp]
        dE = max(abs(mp[round(x["T_over_rs"], 6)] - x["E"]) for x in common)
        L.append("copy %s: %d records, %d common, max |E difference| %.3e" % (p, len(o), len(common), dE))
    open(outp + ".txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-1 - len(others):]))


if __name__ == "__main__":
    if sys.argv[1] == "plain":
        plain(sys.argv[2], sys.argv[3])
    else:
        newton(sys.argv[2], sys.argv[3], sys.argv[4:])
