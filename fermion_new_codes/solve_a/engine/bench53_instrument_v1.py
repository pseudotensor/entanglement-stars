#!/usr/bin/env python
"""bench53_instrument_v1.py (the lane's analyst builder; engine spec 40.4 (2) (a), inbox #53): the instrumentation from the running engines'
own logs (no restart, nothing changed in a running engine): per record, the wall time, the dense eigendecompositions (calls, seconds, seconds
per call), the continuation's seconds, the field solves (calls, seconds) and, where the log prints it, the Newton evaluations of the record's
solve; derived: eigendecompositions per channel per evaluation. The host-device transfer is not separated in these logs (it sits inside the
eigh and continuation phases); a restart with timing hooks separates it (the benchmark's step (b)). Usage: --log NAME=PATH:CHANNELS ... --out."""
import argparse
import os
import re

import numpy as np

REC = re.compile(r"\((\d+) s; eigh (\d+) in ([\d.]+) s, continuation ([\d.]+) s, densities ([\d.]+) s, solves (\d+) in ([\d.]+) s")
EV = re.compile(r"T/rs ([\d.]+) newton \(direct\): converged; (\d+) evaluations in (\d+) s")
TR = re.compile(r"^T/rs ([\d.]+) E/E0")


def parse(path):
    rows = []; evals = {}
    T = None
    for ln in open(path, errors="replace"):
        m = EV.search(ln)
        if m:
            evals[m.group(1)] = int(m.group(2))
        m = TR.search(ln)
        if m:
            T = m.group(1)
        m = REC.search(ln)
        if m:
            rows.append({"T": T, "wall": int(m.group(1)), "eigh_n": int(m.group(2)), "eigh_s": float(m.group(3)), "cont_s": float(m.group(4)),
                         "dens_s": float(m.group(5)), "solve_n": int(m.group(6)), "solve_s": float(m.group(7))})
    for r in rows:
        r["evals"] = evals.get(r["T"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("refusing to overwrite %s" % a.out)
    L = ["# #53 instrumentation, version 1 (engine spec 40.4 (2) (a)): the dominant phases per record from the engines' own logs", "",
         "Written by code/engine/bench53_instrument_v1.py; every number printed by the script from the logs named. The phases overlap (their sum may exceed the record's wall); "
         "the host-device transfer is inside the eigh and continuation phases and is not separated by these logs.", ""]
    for spec in a.log:
        name, rest = spec.split("=", 1)
        path, ch = rest.rsplit(":", 1); ch = int(ch)
        rows = parse(path)
        L += ["## %s (%s; %d channels; %d records with a timing line)" % (name, os.path.basename(path), ch, len(rows)), "",
              "| T/r_s | wall s | eigh calls | eigh s | s per eigh | continuation s | field solves; s | Newton evaluations | eigh per channel per evaluation |", "|---|---|---|---|---|---|---|---|---|"]
        show = rows if len(rows) <= 12 else rows[:2] + rows[-10:]
        for r in show:
            epe = (r["eigh_n"] / (ch * r["evals"])) if r["evals"] else None
            L.append("| %s | %d | %d | %.1f | %.3f | %.1f | %d; %.1f | %s | %s |" % (r["T"], r["wall"], r["eigh_n"], r["eigh_s"], r["eigh_s"] / max(r["eigh_n"], 1), r["cont_s"],
                                                                         r["solve_n"], r["solve_s"], r["evals"] if r["evals"] else "-", ("%.2f" % epe) if epe else "-"))
        if rows:
            w = np.array([r["wall"] for r in rows], float); es = np.array([r["eigh_s"] for r in rows]); cs = np.array([r["cont_s"] for r in rows]); ss = np.array([r["solve_s"] for r in rows])
            pc = np.array([r["eigh_s"] / max(r["eigh_n"], 1) for r in rows])
            L += ["", "- over the %d records: the eigh share of the wall median %.2f, the continuation's %.2f, the field solves' %.2f; seconds per eigh median %.3f (range %.3f to %.3f)" % (
                len(rows), float(np.median(es / w)), float(np.median(cs / w)), float(np.median(ss / w)), float(np.median(pc)), float(pc.min()), float(pc.max())), ""]
    open(a.out, "x").write("\n".join(L) + "\n")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
