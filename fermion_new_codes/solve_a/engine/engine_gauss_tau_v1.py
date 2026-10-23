#!/usr/bin/env python
"""engine_gauss_tau_v1.py (builder analyst; engine spec 25.2: "The attribution instrument is 20.1's Gauss balance per interval ... run on the
set's tau 16, 8, 4, 2 and reading B members, the term that does not fall with tau is the model's"): per member (a collapse run's json and
fields npz with the handoff npz carrying its per-record <N^2>_n and the sea's, R2's and the dust's shell energies: an engine_replay_v2
replay for an engine_collapse_v17 run, the run's own for v19/v24/v25), the Gauss balance per interval as engine_collapse_records_v1 forms
it (with the interval's means: evolution = sum_n <N^2>_n Delta(eps of the propagated sea and dust), reset = -sum_n <N^2>_n Delta Tr[R2 h_n],
lapse = sum_n Delta<N^2>_n eps_n, closure = dE - their sum), accumulated over the windows --windows (T/rs) and over the whole run; then for
the members ordered by tau each window's terms side by side with the ratio of consecutive members (a term that falls with tau halves or
quarters as tau halves; one that does not is the model's). Every number is printed by the script. Args: --members (comma list of
label:run_dir/tag:handoff_npz) --windows (e.g. 0-16,16-22,22-40,0-40) --md."""
import argparse
import json
import os

import numpy as np


def member_terms(run, handoff):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    H = np.load(handoff, allow_pickle=True)
    rs = float(J["args"]["rs"]); tau = float(J["args"]["tau"])
    recs = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    Tz = np.asarray(Z["T"], float); TH = np.asarray(H["T"], float)
    rows = []; prev = None
    for k in range(len(TH)):
        T = float(TH[k]); kz = int(np.argmin(np.abs(Tz - T)))
        if abs(Tz[kz] - T) > 1e-9 or round(T, 6) not in recs:
            prev = None; continue
        eps = np.asarray(Z["eps_tot"][kz], float); N2 = np.asarray(H["N2avg"][k], float)
        eC = np.asarray(H["eps_sea"][k], float) + np.asarray(H["eps_dust"][k], float); eR = np.asarray(H["eps_R2"][k], float)
        E = float(recs[round(T, 6)]["E"])
        if prev is not None:
            a_ = 0.5 * (prev["N2"] + N2)
            evo = float(np.sum(a_ * (eC - prev["eC"]))); rst = -float(np.sum(a_ * (eR - prev["eR"])))
            lap = float(np.sum((N2 - prev["N2"]) * 0.5 * (prev["eps"] + eps)))
            dE = E - prev["E"]
            rows.append((prev["T"] / rs, T / rs, dE, evo, rst, lap, dE - (evo + rst + lap)))
        prev = {"T": T, "N2": N2, "eC": eC, "eR": eR, "eps": eps, "E": E}
    return tau, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--windows", default="0-16,16-22,22-40,0-40")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    mem = []
    for item in a.members.split(","):
        lab, run, hand = item.split(":")
        tau, rows = member_terms(run, hand)
        mem.append((lab, tau, rows))
    wins = [tuple(float(x) for x in w.split("-")) for w in a.windows.split(",")]
    names = ["dE", "evolution", "reset", "lapse", "closure"]
    L = ["## The Gauss balance per interval across the step sizes (engine_gauss_tau_v1.py)", "",
         "Per member the sums over the intervals inside each window (T/rs) of dE and its three parts (the interval's means; the closure the remainder). Members: %s." % (
             "; ".join("%s (tau %g, %d intervals, T/rs %.3f to %.3f)" % (lab, tau, len(rows), rows[0][0] if rows else float("nan"), rows[-1][1] if rows else float("nan")) for lab, tau, rows in mem)), ""]
    for w0, w1 in wins:
        L += ["### Window T/rs %g to %g" % (w0, w1), "", "| member (tau) | " + " | ".join(names) + " | intervals |", "|---|" + "---|" * (len(names) + 1)]
        sums = []
        for lab, tau, rows in mem:
            sel = [r for r in rows if r[0] >= w0 - 1e-9 and r[1] <= w1 + 1e-9]
            s = [sum(r[2 + i] for r in sel) for i in range(len(names))]
            sums.append((lab, tau, s, len(sel)))
            L.append("| %s (%g) | %s | %d |" % (lab, tau, " | ".join("%+.5e" % x for x in s), len(sel)))
        ordered = sorted([x for x in sums if x[3] > 0], key=lambda x: -x[1])
        for i in range(len(ordered) - 1):
            a_, b_ = ordered[i], ordered[i + 1]
            L.append("| ratio %s/%s | %s | - |" % (a_[0], b_[0], " | ".join(("%.3f" % (x / y)) if y != 0 else "-" for x, y in zip(a_[2], b_[2]))))
        L.append("")
    L += ["- a term that falls with tau shrinks by about 2 per halving (first order); a term that does not is the model's (engine 25.2); every number above is printed by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
