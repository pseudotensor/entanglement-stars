#!/usr/bin/env python
"""engine_replay_crosshw_v1.py (builder analyst; engine spec 20.1's cross-hardware check, "the two replays' shell totals compared", and
23.2 (3): COLLAPSE_records_v1.1's exactness table): two engine_replay_v2 runs of the same stored history on different hardware (the H100
and the box's A6000), per record the largest relative difference of the replayed shell quantities between them (the sea's and R2's shell
energies, the dust's, <N^2>_n, the stresses' face flux and centrifugal force of the dust and of the pairs, the smoothed S_r and S_t), and
each replay's own source difference from the stored shell totals (their exactness tables). Args: --a npz-prefix --b npz-prefix --a-json
--b-json --labels A,B --md."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--a-json", required=True); ap.add_argument("--b-json", required=True)
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--md", required=True)
    x = ap.parse_args()
    la, lb = x.labels.split(",")
    A = np.load(x.a + "_handoff.npz", allow_pickle=True); B = np.load(x.b + "_handoff.npz", allow_pickle=True)
    TA = np.asarray(A["T"], float); TB = np.asarray(B["T"], float)
    keys = ["eps_sea", "eps_R2", "eps_dust", "N2avg", "F_face_dust", "fc_dust", "F_face_pairs", "fc_pairs", "S_r_ell", "S_t_ell"]
    L = ["## The production's replay on two GPUs compared (engine_replay_crosshw_v1.py; %s against %s)" % (la, lb), "",
         "Per record the largest difference of each replayed shell quantity between the two replays, relative to its largest value; beside them each replay's source difference from the stored shell totals (eps, J).", "",
         "| T | " + " | ".join(keys) + " | %s against the stored | %s against the stored |" % (la, lb), "|---|" + "---|" * (len(keys) + 2)]
    worst = {k: 0.0 for k in keys}
    ra = {round(r["T_over_rs"], 6): r for r in json.load(open(x.a_json))["rows"]}
    rb = {round(r["T_over_rs"], 6): r for r in json.load(open(x.b_json))["rows"]}
    ta = sorted(ra); tb = sorted(rb)
    for i, T in enumerate(TA):
        j = int(np.argmin(np.abs(TB - T)))
        if abs(TB[j] - T) > 1e-9:
            continue
        row = []
        for k in keys:
            a = np.asarray(A[k][i], float); b = np.asarray(B[k][j], float)
            d = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-300)
            worst[k] = max(worst[k], d); row.append("%.1e" % d)
        qa = ra.get(ta[i]) if i < len(ta) else None; qb = rb.get(tb[j]) if j < len(tb) else None
        fa = ("%.1e, %.1e" % (qa["eps_rel"], qa["J_rel"])) if qa else "-"; fb = ("%.1e, %.1e" % (qb["eps_rel"], qb["J_rel"])) if qb else "-"
        L.append("| %.1f | %s | %s | %s |" % (T, " | ".join(row), fa, fb))
    L += ["", "- the largest relative differences over the records: " + ", ".join("%s %.1e" % (k, v) for k, v in worst.items()) + ".",
          "- each replay's own exactness against the stored shell totals is in its replay table; every number above is printed by the script."]
    open(x.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))


if __name__ == "__main__":
    main()
