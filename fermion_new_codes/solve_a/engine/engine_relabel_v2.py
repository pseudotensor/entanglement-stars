#!/usr/bin/env python
"""engine_relabel_v2.py (builder analyst; engine spec 19.1 and 19.8: "engine_relabel_v2 (the estimate and q_ref on every stored history;
the ratio table against v15)"): per stored collapse record with an iteration history (engine_collapse_v15 and later: per iteration |dE/E|,
max |dN|, max |dv| of the solve against the applied fields), the last self-consistency change c = max(dN, dv) of the accepted iteration,
q_obs = c_last / c_previous (the ratio of the last two residuals; "unresolved" above 0.9) and the fixed-point estimate c q_obs/(1 - q_obs)
(19.1: the distance of the accepted fields from the fixed point under a geometric tail); for a reference run, q_ref = the largest ratio of
successive changes in each record's history and its estimate c q_ref/(1 - q_ref). With --against REF, the ratio table of 19.1's criterion:
for each run's records matched by T, the measured difference max |N - N_ref| of the stored fields over the reference's estimate (a ratio
above 3 reopens the comparison). Args: --runs dir/tag ... --against dir/tag --records --md."""
import argparse
import json
import os

import numpy as np


def load(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Zp = os.path.join(d, "collapse_%s_fields.npz" % tag)
    Z = np.load(Zp, allow_pickle=True) if os.path.exists(Zp) else None
    return tag, J, Z


def estimates(rec):
    h = rec.get("iteration_history") or []
    c = [max(e[1], e[2]) for e in h if e is not None]
    if len(c) < 2:
        return None
    last = c[-1]
    q_obs = c[-1] / c[-2] if c[-2] > 0 else None
    est = (last * q_obs / (1.0 - q_obs)) if (q_obs is not None and q_obs < 0.9) else None
    ratios = [c[i] / c[i - 1] for i in range(2, len(c)) if c[i - 1] > 0]       # from the second trial on (the lagged change is the step itself)
    q_ref = max(ratios) if ratios else None
    est_ref = (last * q_ref / (1.0 - q_ref)) if (q_ref is not None and q_ref < 0.9) else None
    return {"last_change": last, "q_obs": q_obs, "estimate": est, "q_ref": q_ref, "estimate_ref": est_ref, "iterations": len(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--against", default=None)
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## The fixed-point estimates of the stored records (engine_relabel_v2.py)", "",
         "c: the accepted iteration's last change max(dN, dv); q_obs: the ratio of the last two changes; the estimate c q_obs/(1 - q_obs) "
         "(unresolved above 0.9); q_ref: the largest ratio of successive changes from the second trial on; its estimate c q_ref/(1 - q_ref).", ""]
    ref = load(a.against) if a.against else None
    for run in a.runs:
        tag, J, Z = load(run)
        rs = float(J["args"]["rs"])
        recs = [r for r in J["records"] if r.get("E") is not None]
        if a.records:
            recs = recs[:a.records]
        L += ["### %s (%s)" % (tag, J.get("script")), "", "| T/rs | iterations | c | q_obs | estimate | q_ref | estimate (q_ref) |" + (" max abs N - N_ref | ref's estimate | ratio |" if ref else ""),
              "|---|---|---|---|---|---|---|" + ("---|---|---|" if ref else "")]
        worst = 0.0
        for r in recs:
            e = estimates(r)
            if e is None:
                L.append("| %.3f | - | no history |" % (r["T"] / rs))
                continue
            row = "| %.3f | %d | %.2e | %s | %s | %s | %s |" % (r["T"] / rs, e["iterations"], e["last_change"], ("%.3f" % e["q_obs"]) if e["q_obs"] is not None else "-",
                                                         ("%.2e" % e["estimate"]) if e["estimate"] is not None else "unresolved",
                                                         ("%.3f" % e["q_ref"]) if e["q_ref"] is not None else "-", ("%.2e" % e["estimate_ref"]) if e["estimate_ref"] is not None else "unresolved")
            if ref and Z is not None and ref[2] is not None:
                rtag, RJ, RZ = ref
                Tz = np.asarray(Z["T"], float); RT = np.asarray(RZ["T"], float)
                k = int(np.argmin(np.abs(Tz - r["T"]))); kr = int(np.argmin(np.abs(RT - r["T"])))
                rrec = [q for q in RJ["records"] if q.get("E") is not None and abs(q["T"] - r["T"]) < 1e-9]
                if abs(RT[kr] - r["T"]) < 1e-9 and rrec:
                    dN = float(np.max(np.abs(np.asarray(Z["N"][k]) - np.asarray(RZ["N"][kr]))))
                    er = estimates(rrec[0])
                    est_r = (er or {}).get("estimate_ref")
                    ratio = (dN / est_r) if est_r else None
                    if ratio is not None:
                        worst = max(worst, ratio)
                    row += " %.2e | %s | %s |" % (dN, ("%.2e" % est_r) if est_r else "-", ("%.2f" % ratio) if ratio is not None else "-")
                else:
                    row += " - | - | - |"
            L.append(row)
        if ref:
            L += ["", "- the largest ratio of the measured difference to the reference's estimate: %.2f (19.1: above 3 reopens the comparison)." % worst]
        L.append("")
    L += ["- every number above is printed by the script from the runs' json and fields npz."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
