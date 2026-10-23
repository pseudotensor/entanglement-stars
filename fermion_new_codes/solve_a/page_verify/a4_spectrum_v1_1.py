#!/usr/bin/env python3
"""a4_spectrum_v1_1.py (solve, page_verify A4 v1.1; the cold reading's P2 spectrum clause, reviewer A c4_claims): the post-burst outgoing spectrum
of kappa 1 and 2 on the exterior shells beyond 24 r_s from the stored 0.5 T_H^ref bins (d13.npz shellbin_*: per bin and sign the outgoing
subspace's mode count n_out, number content N_out_P and energy content E_out_P relative to the instantaneous vacuum P_-).
Per 2 T_H bin of |e|/T_H^ref (both signs together): the change over the shells' window [A, B] of the outgoing number content per outgoing mode,
the energy-weighted occupation sum_k w_k sum_bins dE_out_P/|e|_mid / sum_k w_k n_out (w = 16 kappa; particles and holes both counted positive; the signed number content cancels between the signs and is printed beside), and the energy shares above 10 and 25 T_H of the outgoing energy change.
Usage: a4_spectrum_v1_1.py OUT.json [A B]"""
import json, sys
import numpy as np

PV4 = "<internal>/page_v4_data_v1.json"


def main():
    out = sys.argv[1]; A = float(sys.argv[2]) if len(sys.argv) > 2 else 56.0; B = float(sys.argv[3]) if len(sys.argv) > 3 else 92.0
    d = json.load(open(PV4)); man = {m["label"]: m["realpath"] for m in d["manifest"]}
    THref = None; acc = {}
    for kap in (1, 2):
        pj = man["B2 production kappa %d d13.json" % kap]; z = np.load(pj[:-5] + ".npz", allow_pickle=True)
        J = json.load(open(pj))
        if THref is None:
            THref = float(J["clocks"]["%010.4f" % A]["per_channel"][str(kap)]["T_H_ref"])
        w = 16.0 * kap
        g = {}
        for t in (A, B):
            k = "T%010.4f_k%d_shellbin_" % (t, kap)
            g[t] = {q: np.asarray(z[k + q], float) for q in ("lo", "hi", "sign", "n_out", "N_out_P", "E_out_P")}
        assert np.array_equal(g[A]["lo"], g[B]["lo"]) and np.array_equal(g[A]["sign"], g[B]["sign"]), "bin layouts differ between the window's ends"
        lo = g[A]["lo"] / THref; hi = g[A]["hi"] / THref
        for i in range(len(lo)):
            if not np.isfinite(hi[i]) or lo[i] >= 40 - 1e-9:
                key = ("above40", None)
            else:
                key = ("bin2", int(lo[i] // 2) * 2)
            a = acc.setdefault(key, {"dN": 0.0, "n": 0.0, "dE": 0.0, "dNe": 0.0, "lo": None})
            emid = 0.5 * (g[A]["lo"][i] + (g[A]["hi"][i] if np.isfinite(g[A]["hi"][i]) else 2 * g[A]["lo"][i]))
            a["dN"] += w * (g[B]["N_out_P"][i] - g[A]["N_out_P"][i]); a["n"] += w * 0.5 * (g[A]["n_out"][i] + g[B]["n_out"][i]); a["dE"] += w * (g[B]["E_out_P"][i] - g[A]["E_out_P"][i])
            a["dNe"] += w * (g[B]["E_out_P"][i] - g[A]["E_out_P"][i]) / emid
    rows = []
    totE = sum(a["dE"] for a in acc.values())
    for key in sorted(acc, key=lambda k: (k[0] != "bin2", k[1] if k[1] is not None else 0)):
        a = acc[key]
        rows.append({"bin_TH": ("%d-%d" % (key[1], key[1] + 2)) if key[0] == "bin2" else ">=40 (coarse bins to the band top and above)",
                     "occupation_per_out_mode": a["dNe"] / a["n"] if a["n"] else None, "signed_number_per_out_mode": a["dN"] / a["n"] if a["n"] else None, "dE_out": a["dE"], "share": a["dE"] / totE})
    share_above = lambda x: sum(a["dE"] for k_, a in acc.items() if (k_[0] != "bin2") or k_[1] >= x) / totE
    res = {"window_shells": [A, B], "T_H_ref": THref, "total_dE_out": totE, "share_above_10TH": share_above(10), "share_above_24TH": share_above(24),
           "share_above_26TH": share_above(26), "rows": rows,
           "label": "outgoing content of kappa 1 and 2 relative to the instantaneous vacuum, exterior shells beyond 24 r_s, 2 T_H^ref bins (the stored 0.5 T_H bins summed; above 40 T_H the stored bins double to the band top)"}
    json.dump(res, open(out, "w"), indent=1)
    for r in rows:
        print(r["bin_TH"], "%.2e" % r["occupation_per_out_mode"] if r["occupation_per_out_mode"] is not None else "-", "%.3f" % r["share"])
    print("share above 10, 24, 26 T_H: %.3f %.3f %.3f; total dE_out %.4f" % (res["share_above_10TH"], res["share_above_24TH"], res["share_above_26TH"], totE))


if __name__ == "__main__":
    main()
