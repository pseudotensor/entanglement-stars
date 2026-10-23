#!/usr/bin/env python
"""engine_tau_compare_v1.py (builder analyst; engine spec 24.3: "What the tau 4 control must show: at matched times the formation, closing
and re-formation within 0.5 r_s of tau 8's, the second surface's depth and the dust inside it at 20.625 within 10 percent, the pockets'
drain by 14.6 within the c_LZ spread"): two collapse runs of the same member at different tau (their jsons and fields npz) at matched times
T/rs on a grid of --every r_s where both hold a record: E/E0 of each and their difference, R_t, min(N - |v|), r_in, R_c, the dust's and the
pairs' shell energy inside R_t (eps_tot - eps_pairs and eps_pairs summed over the shells below R_t), the pairs' shell energy, the pockets;
then each run's trapping episodes (formation: the first record with a trapping zero; closing: the first record after it without) and the
pocket phase (the first and the last record with pockets), and 24.3's criteria where both runs reach the times (the episode times'
differences against 0.5 r_s; at --depth-at the relative differences of min(N - |v|) and of the dust inside R_t against 10 percent; the
pockets' last record). Args: --a dir/tag --b dir/tag --labels A,B --every 0.5 --depth-at 20.625 --md."""
import argparse
import json
import os

import numpy as np


def load(path):
    d, tag = os.path.split(path)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    rs = float(J["args"]["rs"])
    recs = {round(r["T"] / rs, 6): r for r in J["records"] if r.get("E") is not None}
    return {"tag": tag, "J": J, "Z": Z, "rs": rs, "E0": float(J["head"]["E0"]), "tau": float(J["args"]["tau"]), "recs": recs,
            "Tz": np.asarray(Z["T"], float), "R": np.asarray(Z["R"], float)}


def inside(run, r):
    Rt = r.get("R_t")
    if not Rt:
        return None, None
    k = int(np.argmin(np.abs(run["Tz"] - r["T"])))
    if abs(run["Tz"][k] - r["T"]) > 1e-9:
        return None, None
    ep = np.asarray(run["Z"]["eps_pairs"][k], float); et = np.asarray(run["Z"]["eps_tot"][k], float)
    sel = run["R"] < Rt
    return float((et - ep)[sel].sum()), float(ep[sel].sum())


def episodes(run):
    out = []; cur = None
    for t in sorted(run["recs"]):
        r = run["recs"][t]
        if r.get("R_t"):
            if cur is None:
                cur = {"formed": t, "last": t}
            cur["last"] = t
        elif cur is not None:
            cur["closed"] = t; out.append(cur); cur = None
    if cur is not None:
        cur["closed"] = None; out.append(cur)
    return out


def pockets(run):
    ts = [t for t in sorted(run["recs"]) if sum((run["recs"][t].get("n_pocket") or {}).values()) > 0]
    return (ts[0], ts[-1]) if ts else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--depth-at", type=float, default=20.625)
    ap.add_argument("--md", required=True)
    x = ap.parse_args()
    la, lb = x.labels.split(",")
    A = load(x.a); B = load(x.b)
    f = lambda v, fm: (fm % v) if v is not None else "-"
    common = sorted(set(A["recs"]) & set(B["recs"]))
    grid = [t for t in common if abs(t / x.every - round(t / x.every)) < 1e-6]
    L = ["## %s (%s, tau %g) against %s (%s, tau %g) at matched times (engine_tau_compare_v1.py)" % (A["tag"], la, A["tau"], B["tag"], lb, B["tau"]), "",
         "Common records from T/rs %s to %s; the table every %g r_s; each pair of numbers is (%s, %s)." % (f(common[0] if common else None, "%.3f"), f(common[-1] if common else None, "%.3f"), x.every, la, lb), "",
         "| T/rs | E/E0 | E difference (relative) | R_t | min(N - abs v) | r_in | R_c | dust inside R_t | pairs inside R_t | pairs' shell energy | pockets |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in grid:
        ra = A["recs"][t]; rb = B["recs"][t]
        da, pa = inside(A, ra); db, pb = inside(B, rb)
        L.append("| %.3f | %.7f, %.7f | %+.2e | %s, %s | %+.4f, %+.4f | %s, %s | %s, %s | %s, %s | %s, %s | %+.3e, %+.3e | %d, %d |" % (
            t, ra["E"] / A["E0"], rb["E"] / B["E0"], (rb["E"] - ra["E"]) / abs(ra["E"]), f(ra.get("R_t"), "%.3f"), f(rb.get("R_t"), "%.3f"),
            ra["min_N_minus_absv"], rb["min_N_minus_absv"], f(ra.get("r_in"), "%.3f"), f(rb.get("r_in"), "%.3f"), f(ra.get("R_c"), "%.2f"), f(rb.get("R_c"), "%.2f"),
            f(da, "%.3f"), f(db, "%.3f"), f(pa, "%+.3e"), f(pb, "%+.3e"), ra["E_pairs_shells"], rb["E_pairs_shells"],
            sum((ra.get("n_pocket") or {}).values()), sum((rb.get("n_pocket") or {}).values())))
    epA = episodes(A); epB = episodes(B)
    lastA = max(A["recs"]); lastB = max(B["recs"])
    L += ["", "### The trapping episodes and the pocket phase", ""]
    for lab, run, ep, last in ((la, A, epA, lastA), (lb, B, epB, lastB)):
        txt = "; ".join("episode %d formed at %.3f, %s" % (k + 1, e["formed"], ("closed by %.3f" % e["closed"]) if e.get("closed") else "open at the last record") for k, e in enumerate(ep)) or "no trapping zero"
        p0, p1 = pockets(run)
        L.append("- %s (records to T/rs %.3f): %s; pockets from %s to %s (the first and the last record with pockets)." % (lab, last, txt, f(p0, "%.3f"), f(p1, "%.3f")))
    L += ["", "### Engine 24.3's criteria where both runs reach the times", ""]
    names = []
    for k in range(max(len(epA), len(epB))):
        names.append(("formation %d" % (k + 1), epA[k]["formed"] if k < len(epA) else None, epB[k]["formed"] if k < len(epB) else None))
        names.append(("closing %d" % (k + 1), epA[k].get("closed") if k < len(epA) else None, epB[k].get("closed") if k < len(epB) else None))
    for nm, ta, tb in names:
        if ta is None and tb is None:
            continue
        if ta is not None and tb is not None:
            L.append("- %s: %.3f, %.3f; difference %.3f r_s (%s 0.5 r_s)." % (nm, ta, tb, tb - ta, "within" if abs(tb - ta) <= 0.5 + 1e-9 else "beyond"))
        else:
            reach = lastB if tb is None else lastA
            L.append("- %s: %s, %s; %s." % (nm, f(ta, "%.3f"), f(tb, "%.3f"), ("not reached by %s's records (to %.3f)" % (lb if tb is None else la, reach))))
    t = round(x.depth_at, 6)
    if t in A["recs"] and t in B["recs"]:
        ra = A["recs"][t]; rb = B["recs"][t]
        da, _ = inside(A, ra); db, _ = inside(B, rb)
        dd = (ra["min_N_minus_absv"], rb["min_N_minus_absv"])
        rel_depth = abs(dd[1] - dd[0]) / max(abs(dd[0]), 1e-300)
        rel_dust = (abs(db - da) / max(abs(da), 1e-300)) if (da is not None and db is not None) else None
        L.append("- at T/rs %.3f: min(N - |v|) %+.4f, %+.4f (relative difference %.3f, %s 10 percent); the dust inside R_t %s, %s (relative difference %s)." % (
            t, dd[0], dd[1], rel_depth, "within" if rel_depth <= 0.1 else "beyond", f(da, "%.3f"), f(db, "%.3f"), f(rel_dust, "%.3f")))
    else:
        L.append("- at T/rs %.3f: not reached by both runs (records to %.3f and %.3f)." % (t, lastA, lastB))
    L += ["- every number above is read from the two runs' jsons and fields npz by the script."]
    open(x.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
