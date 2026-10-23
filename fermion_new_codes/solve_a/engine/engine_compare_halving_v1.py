#!/usr/bin/env python
"""engine_compare_halving_v1.py (builder analyst; engine spec 21.3: "their differences from the resumed history over the same interval (E,
R_t, r_in, the pairs' energy, the node fields) are the halving convention's effect, the first measured size of the convention"): the
stopped production's records continued by halving (T/rs 11.125 to 12.625, engine_collapse_v17) against the resumed run's records at the
same times (engine_collapse_v19, the Landau-Zener maximum-overlap continuation): E/E0 and its relative difference, R_t, r_in, the pairs'
shell energy, min(N - |v|), the largest differences of the record's fields N, v and of the applied fields over the sites, and per record the
resumed run's continuation summary (mode, pockets, inversions, intermediate weights, masks equal to the negative set). Args: --old dir/tag
--new dir/tag --from 11.0 --md."""
import argparse
import json
import os

import numpy as np


def load(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    return tag, J, Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--from", dest="t0", type=float, default=11.0)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    to, Jo, Zo = load(a.old); tn, Jn, Zn = load(a.new)
    rs = float(Jo["args"]["rs"]); E0 = float(Jo["head"]["E0"])
    ro = {round(r["T"] / rs, 6): r for r in Jo["records"] if r.get("E") is not None}
    rn = {round(r["T"] / rs, 6): r for r in Jn["records"] if r.get("E") is not None}
    To = np.asarray(Zo["T"], float) / rs; Tn = np.asarray(Zn["T"], float) / rs
    L = ["## The halving convention against the Landau-Zener continuation (engine_compare_halving_v1.py)", "",
         "Old: %s (%s, R2 continued by halving from T/rs %.3f); new: %s (%s, resumed from the old run's record at T/rs %.3f)." % (to, Jo.get("script"), a.t0 + 0.125, tn, Jn.get("script"), a.t0), "",
         "| T/rs | E/E0 old | E/E0 new | relative difference | R_t old, new | r_in old, new | pairs' energy old, new | min(N - abs v) old, new | max abs dN, dv (record) | max abs dN, dv (applied) | new: continuation; pockets; inversions; intermediate; mask = neg |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted(k for k in ro if k > a.t0 + 1e-9 and k in rn)
    for k in keys:
        o, n = ro[k], rn[k]
        io = int(np.argmin(np.abs(To - k))); i_n = int(np.argmin(np.abs(Tn - k)))
        dN = float(np.max(np.abs(np.asarray(Zo["N"][io]) - np.asarray(Zn["N"][i_n])))); dv = float(np.max(np.abs(np.asarray(Zo["v"][io]) - np.asarray(Zn["v"][i_n]))))
        dNa = float(np.max(np.abs(np.asarray(Zo["N_applied"][io]) - np.asarray(Zn["N_applied"][i_n])))); dva = float(np.max(np.abs(np.asarray(Zo["v_applied"][io]) - np.asarray(Zn["v_applied"][i_n]))))
        f2 = lambda x: ("%.3f" % x) if x is not None else "-"
        L.append("| %.3f | %.7f | %.7f | %+.2e | %s, %s | %s, %s | %.4e, %.4e | %.4f, %.4f | %.2e, %.2e | %.2e, %.2e | %s; %s; %s; %s; %s |" % (
            k, o["E"] / E0, n["E"] / E0, (n["E"] - o["E"]) / o["E"], f2(o.get("R_t")), f2(n.get("R_t")), f2(o.get("r_in")), f2(n.get("r_in")),
            o["E_pairs_shells"], n["E_pairs_shells"], o["min_N_minus_absv"], n["min_N_minus_absv"], dN, dv, dNa, dva,
            (n.get("continuation_label") or "-")[:60], sum((n.get("n_pocket") or {}).values()), n.get("inversions_total", "-"), n.get("intermediate_weights_total", "-"),
            n.get("mask_equals_negative_set", "-")))
    L += ["", "- every number above is read from the two runs' json and fields npz by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
