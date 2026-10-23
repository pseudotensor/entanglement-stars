#!/usr/bin/env python
"""engine_clz_ladder_v2.py (builder analyst; engine spec 25.1: "the ladder is extended through the events it must cover: both members
continued from 14 to 21.5 (the drain, the closing, the re-formation, the second pocket episode at 20.875 to 21.0)"; its spread is "an
observed envelope, not a bound"): version 1 over --from to --to (default 11 to 21.5) with each member's trapping episodes (formation: the
first record with a trapping zero; closing: the first after it without) and its pocket phases (the first and last record of each run of
records with pockets) printed per run, and the summary worded as engine 25.1 rules. A member extended by engine_collapse_v25 is one run
(its json and fields hold the parent's records to the resume time). Version 1's docstring: engine spec 21.2's control: "the c_LZ ladder 1/2, 1, 2 on the resumed production over T/rs 11 to
14 at tau 8 ... the spread in E, R_t, r_in, the pairs' energy and the node fields is the reference-convention error"): the runs at the
ladder's c_LZ values read at their common record times: per time E/E0, R_t, r_in and the pairs' shell energy of each run, the pockets and
inversions, the spread (largest minus smallest over the runs) of each quantity, and the node fields' largest pairwise difference (the
record's N and v at the sites). Args: --runs label=dir/tag ... --from --to --md."""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--from", dest="t0", type=float, default=11.0)
    ap.add_argument("--to", dest="t1", type=float, default=21.5)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    R = []
    for spec in a.runs:
        lab, run = spec.split("=", 1)
        d, tag = os.path.split(run)
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
        rs = float(J["args"]["rs"])
        R.append((lab, tag, rs, float(J["head"]["E0"]), {round(r["T"] / rs, 6): r for r in J["records"] if r.get("E") is not None},
                  np.asarray(Z["T"], float) / rs, Z))
    common = sorted(set.intersection(*[set(k for k in r[4] if a.t0 + 1e-9 < k <= a.t1 + 1e-9) for r in R]))
    L = ["## The c_LZ ladder at matched times (engine_clz_ladder_v2.py)", "", "Runs: " + "; ".join("%s = %s" % (r[0], r[1]) for r in R) + ".", "",
         "| T/rs | " + " | ".join("E/E0 (%s)" % r[0] for r in R) + " | spread E/E0 | " + " | ".join("R_t (%s)" % r[0] for r in R) + " | spread R_t | " +
         " | ".join("r_in (%s)" % r[0] for r in R) + " | " + " | ".join("pairs (%s)" % r[0] for r in R) + " | spread pairs | max abs dN, dv | pockets | inversions |",
         "|---|" + "---|" * (4 * len(R) + 6)]
    worst = {"E": 0.0, "Rt": 0.0, "pairs": 0.0, "N": 0.0, "v": 0.0}
    for t in common:
        recs = [r[4][t] for r in R]
        E = [q["E"] / r[3] for q, r in zip(recs, R)]; Rt = [q.get("R_t") for q in recs]; ri = [q.get("r_in") for q in recs]; pa = [q["E_pairs_shells"] for q in recs]
        Ns = []; vs = []
        for r in R:
            k = int(np.argmin(np.abs(r[5] - t))); Ns.append(np.asarray(r[6]["N"][k], float)); vs.append(np.asarray(r[6]["v"][k], float))
        dN = max(float(np.max(np.abs(x - y))) for x in Ns for y in Ns); dv = max(float(np.max(np.abs(x - y))) for x in vs for y in vs)
        sE = max(E) - min(E); sR = (max(Rt) - min(Rt)) if all(x is not None for x in Rt) else None; sP = max(pa) - min(pa)
        worst["E"] = max(worst["E"], sE); worst["pairs"] = max(worst["pairs"], sP); worst["N"] = max(worst["N"], dN); worst["v"] = max(worst["v"], dv)
        if sR is not None:
            worst["Rt"] = max(worst["Rt"], sR)
        f = lambda x, fm: (fm % x) if x is not None else "-"
        L.append("| %.3f | %s | %.2e | %s | %s | %s | %s | %.2e | %.2e, %.2e | %s | %s |" % (
            t, " | ".join("%.7f" % x for x in E), sE, " | ".join(f(x, "%.3f") for x in Rt), f(sR, "%.2e"), " | ".join(f(x, "%.3f") for x in ri),
            " | ".join("%.4e" % x for x in pa), sP, dN, dv, ", ".join(str(sum((q.get("n_pocket") or {}).values())) for q in recs),
            ", ".join(str(q.get("inversions_total", "-")) for q in recs)))
    L += ["", "### The events per run", ""]
    for r in R:
        ts = sorted(r[4])
        eps_ = []; cur = None
        for t in ts:
            if r[4][t].get("R_t"):
                if cur is None:
                    cur = [t, None]
            elif cur is not None:
                cur[1] = t; eps_.append(cur); cur = None
        if cur is not None:
            eps_.append(cur)
        pk = []; pc = None
        for t in ts:
            npk = sum((r[4][t].get("n_pocket") or {}).values())
            if npk > 0:
                if pc is None:
                    pc = [t, t]
                pc[1] = t
            elif pc is not None:
                pk.append(pc); pc = None
        if pc is not None:
            pk.append(pc)
        L.append("- %s (records to T/rs %.3f): trapping episodes %s; pocket phases %s." % (
            r[0], ts[-1], "; ".join("formed at %.3f, %s" % (e[0], ("closed by %.3f" % e[1]) if e[1] is not None else "open at the last record") for e in eps_) or "none",
            "; ".join("%.3f to %.3f" % (p[0], p[1]) for p in pk) or "none"))
    L += ["", "- the largest spreads over T/rs %g to %g: E/E0 %.2e, R_t %.2e, the pairs' energy %.2e, N %.2e, v %.2e (the convention's measured sensitivity: an observed envelope, not a bound; engine 25.1)." % (
        a.t0, a.t1, worst["E"], worst["Rt"], worst["pairs"], worst["N"], worst["v"]), "- every number above is read from the runs' json and fields npz by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
