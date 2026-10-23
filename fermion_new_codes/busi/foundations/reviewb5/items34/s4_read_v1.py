#!/usr/bin/env python
"""s4_read_v1.py (busi, reviewb5 items 3/4): reads s2_items34_d<d>.npz/json and prints what the worst full-vector direction is made of
(clock vs flow weight; the flow's radial, tangential share about the centre and its rotation/divergence-like split per bump), and the
finite-amplitude table. Box, seconds. Output: <out>/s4_read_v1.json."""
import json
import os

import numpy as np

OUT = "<internal>/items34"


def comp(c, Y):
    c = c.reshape(-1, 4)
    wN = float((c[:, 0] ** 2).sum()); cv = c[:, 1:]
    r = np.sqrt((Y ** 2).sum(1)); rh = np.where(r[:, None] > 0, Y / np.where(r[:, None] > 0, r[:, None], 1.0), 0.0)
    rad = (cv * rh).sum(1)
    w_rad = float((rad ** 2).sum()); w_v = float((cv ** 2).sum())
    shells = {}
    for rr in np.unique(np.round(r, 6)):
        m = np.isclose(r, rr)
        shells["%.3f" % rr] = float((c[m] ** 2).sum())
    return {"weight_clock": wN, "weight_flow": w_v, "flow_radial_share": w_rad / w_v if w_v > 0 else None, "weight_by_centre_radius": shells}


def main():
    res = {}
    for d in (2, 3, 4):
        f = os.path.join(OUT, "s2_items34_d%d" % d)
        if not os.path.exists(f + ".json"):
            continue
        R = json.load(open(f + ".json")); Z = np.load(f + ".npz")
        Y = Z["Y"]
        out = {k: comp(Z[k], Y) for k in ("c_worst_all", "c_worst_N", "c_worst_v")}
        Ad = Z["Ad_b0"]
        for k in ("c_worst_all", "c_worst_N", "c_worst_v"):
            ds = Ad @ Z[k]
            out[k]["ds_energy_share"] = float((ds[0::4] ** 2).sum() / (ds ** 2).sum())
        rows = list(R.get("finite_amplitude", []))
        src = {"main": len(rows)}
        for extra in ([OUT + "/old/s2_items34_d3_run1.json", OUT + "/s2_items34_d3_clock.json"] if d == 3 else []):
            if os.path.exists(extra):
                rr = json.load(open(extra)).get("finite_amplitude", [])
                src[os.path.basename(extra)] = len(rr); rows += rr
        out["finite_sources"] = src
        out["finite"] = [{kk: row.get(kk) for kk in ("dir", "subspace", "amp_over_eta_max", "eta", "lambda_min", "one_minus_lambda_max",
                                                     "margin_eval", "norm_Y", "norm_R_ds", "norm_dP", "gap_hu", "n_neg_hu", "nonlinear_ds_rel")}
                         for row in rows]
        res["d%d" % d] = out
    print(json.dumps(res, indent=1))
    with open(os.path.join(OUT, "s4_read_v1.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
