#!/usr/bin/env python
"""engine_rotated_blocks_v1.py (builder analyst; engine spec 25.1: "count 9 rotated channel-records: BUILDER reads each block's position from
the stored eigenvalues and mask; a block wholly occupied or wholly empty is immaterial; a straddling one is replayed with ops v3"): per run
(a v19/v24 json with its spectra npz), every (record, channel) whose continuation record has rotated_blocks > 0: the record's stored
eigenvalues and occupation mask for that channel, the degenerate blocks at the record's block tolerance (engine_ops_v2.blocks_of, the rule
that rotated them, and engine_ops_v3.blocks_of_v3 beside), and per block of more than one level: its index range, its eigenvalues' range
and width, whether it straddles zero, and its occupation (all, none, or k of n). Args: --runs (comma list of dir/tag) --md."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v2 as OPS2  # noqa: E402
import engine_ops_v3 as OPS3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## The rotated degenerate blocks in the stored records (engine_rotated_blocks_v1.py)", "",
         "| run | T/rs | channel | tol_block | block (indices) | eigenvalues | width | straddles zero | occupied | v3 blocks there |", "|---|---|---|---|---|---|---|---|---|---|"]
    n_straddle = n_partial = n_blocks = 0
    for run in [r for r in a.runs.split(",") if r.strip()]:
        d, tag = os.path.split(run)
        J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
        S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
        rs = float(J["args"]["rs"]); TS = np.asarray(S["T"], float)
        for r in J["records"]:
            c = r.get("continuation")
            if r.get("E") is None or not isinstance(c, dict):
                continue
            for k, ci in c.items():
                if not isinstance(ci, dict) or not (ci.get("rotated_blocks") or 0):
                    continue
                i = int(np.argmin(np.abs(TS - r["T"])))
                if abs(TS[i] - r["T"]) > 1e-9:
                    L.append("| %s | %.3f | %s | no stored spectrum |" % (tag, r["T"] / rs, k)); continue
                e = np.asarray(S["e_%s" % k][i], float)
                mask = np.unpackbits(np.asarray(S["mask_%s" % k][i], np.uint8))[:len(e)].astype(bool)
                tol = float(ci.get("tol_block") or 1e-8)
                b2 = [(p, q) for (p, q) in OPS2.blocks_of(e, tol) if q - p > 1]
                b3 = [(p, q) for (p, q) in OPS3.blocks_of_v3(e, tol) if q - p > 1]
                for (p, q) in b2:
                    n_blocks += 1
                    occ = int(mask[p:q].sum()); n = q - p
                    st = bool(e[p] < 0.0 < e[q - 1])
                    n_straddle += int(st); n_partial += int(0 < occ < n)
                    v3 = ", ".join("[%d, %d)" % (x, y) for (x, y) in b3 if x < q and y > p) or "none"
                    L.append("| %s | %.3f | %s | %.0e | [%d, %d) | %.6e to %.6e | %.2e | %s | %s | %s |" % (
                        tag, r["T"] / rs, k, tol, p, q, e[p], e[q - 1], e[q - 1] - e[p], st, ("all" if occ == n else ("none" if occ == 0 else "%d of %d" % (occ, n))), v3))
    L += ["", "- %d rotated blocks; straddling zero: %d; partly occupied: %d (a block wholly occupied or wholly empty is immaterial to the projector; engine 25.1)." % (n_blocks, n_straddle, n_partial),
          "- every number above is read from the runs' jsons and spectra by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
