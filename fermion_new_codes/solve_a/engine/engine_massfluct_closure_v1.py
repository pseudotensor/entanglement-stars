#!/usr/bin/env python
"""engine_massfluct_closure_v1.py (builder analyst; red team R25b defects 6 and 8): per member of a star family the zero-temperature filling
rebuilt exactly as engine_massfluct_v3 does it (its first pass: the channels' positive levels until two consecutive channels lie above E_top,
the pooled levels with the weights 16 kappa, the lowest N_B filled, a degenerate last multiplet filled fractionally), and printed: the last
multiplet's fraction and whether every occupation is 0 or 1 (a closed multiplet), the mean mass function <m(R_surf)> = the cumulative stored
source to R_surf against E, and R_surf. Args: --star --members --md --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v1 as OPS  # noqa: E402


def filling(M, half):
    m, nsites = M["m"], M["nsites"]
    N = np.array(M["N"]); v = np.array(M["v"])
    NBt = float(M["N_B"])
    ep_all = []
    above = 0
    for kappa in range(1, 2000):
        h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half)
        e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
        ep = e[e > 0]
        ep_all.append((kappa, ep))
        lv = np.concatenate([x[1] for x in ep_all]); wv = np.concatenate([np.full(len(x[1]), 16.0 * x[0]) for x in ep_all])
        o = np.argsort(lv); cw = np.cumsum(wv[o])
        E_top_est = float(lv[o][min(int(np.searchsorted(cw, NBt - 1e-9)), len(lv) - 1)]) if cw[-1] >= NBt else np.inf
        above = above + 1 if ep.min() > E_top_est else 0
        if above >= 2:
            break
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    left = NBt; i = 0; fracs = []
    while i < len(lev) and left > 1e-12:
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        wsum = sum(x[1] for x in lev[i:j])
        frac = min(1.0, left / wsum)
        fracs.append((lev[i][0], wsum, frac))
        left -= frac * wsum
        i = j
    return fracs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--members", type=int, nargs="+", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    D = json.load(open(a.star))
    half = bool(D["args"].get("half_offset", 0))
    rows = []
    for i in a.members:
        M = D["members"][i]
        fr = filling(M, half)
        last = fr[-1]
        closed = all(abs(f) < 1e-12 or abs(f - 1.0) < 1e-12 for _, _, f in fr)
        R = np.arange(M["nsites"], dtype=float) + (0.5 if half else 1.0)
        cum = np.cumsum(np.array(M["eps_tot"]))
        rs_ = M.get("R_surf")
        mean = float(cum[min(int(np.searchsorted(R, float(rs_))), len(R) - 1)]) if rs_ else None
        rows.append({"member": i, "N_B": M["N_B"], "E": M["E"], "last_multiplet": {"level": last[0], "weight": last[1], "fraction": last[2]}, "closed": closed,
                     "multiplets_filled": len(fr), "R_surf": rs_, "mean_m_R_surf": mean, "mean_over_E": (mean / M["E"]) if mean is not None else None})
        print("member %d: N_B %.2f, %d multiplets, the last at %.8f with weight %g filled to %.12f; closed %s; <m(R_surf)>/E %s" % (
            i, M["N_B"], len(fr), last[0], last[1], last[2], closed, ("%.6f" % rows[-1]["mean_over_E"]) if mean is not None else "-"), flush=True)
    json.dump({"script": "engine_massfluct_closure_v1.py", "star": os.path.basename(a.star), "rows": rows}, open(a.out_json, "w"), indent=1)
    L = ["## The members' fillings and surface means (engine_massfluct_closure_v1.py on %s)" % os.path.basename(a.star), "",
         "| member | N_B | E | multiplets filled | the last: level, weight, fraction | closed (every f in {0, 1}) | R_surf | <m(R_surf)>/E |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lm = r["last_multiplet"]
        L.append("| %d | %.2f | %.6f | %d | %.8f, %g, %.12f | %s | %s | %s |" % (r["member"], r["N_B"], r["E"], r["multiplets_filled"], lm["level"], lm["weight"], lm["fraction"],
                 "yes" if r["closed"] else "NO", ("%.3f" % r["R_surf"]) if r["R_surf"] else "-", ("%.6f" % r["mean_over_E"]) if r["mean_over_E"] is not None else "-"))
    L += ["", "- every number above is printed by the script; the filling is engine_massfluct_v3's first pass."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
