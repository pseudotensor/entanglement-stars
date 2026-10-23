#!/usr/bin/env python
"""Strand 2b sec. 15 item 6(x) (the lead's ruling in JUDGMENT_strand2b_row.md revision 6; builder analyst): the uniqueness bookkeeping rerun
on the stored start records of driver v13 (R4, pass2v13) and v7 (the cube, c8encl_v7), no new solves.  A start is accepted when its returned
vector's own residual is at most thr = max(1e-11, 10 x the case's residual noise floor) (the drivers accepted at the Newton criterion 1e-11);
accepted starts with a non-positive component count as negative; the positive ones are clustered at 1e-8 (the drivers' distinctness
tolerance); the wording follows the drivers' ("scan inconclusive: 0 accepted starts", "no second positive solution found in N converged
starts", "K distinct positive solutions in N converged starts").  Per case: the noise floor, thr, the drivers' and the restated counts and
wording, whether the positive cluster contains the continuation solution (distance below 1e-8), and the change of verdict.  Numbers emitted,
not retyped.
(version 2: the drivers' wording counts distinct solutions among the starts only; when the accepted positive roots do not include the
continuation solution, the case has a second positive solution.  Version 2 counts the continuation solution with them (distinct including
it), lists each accepted positive root's distance from it, and words such a case "K distinct positive solutions, the continuation solution
included, in N converged starts".  Version 1 printed "no second positive solution found in 1 converged starts" for the cube's (2,0) k = 1
scan, whose one accepted root lies 0.327 from the continuation solution.)"""
import argparse
import glob
import json
import os

import numpy as np


def f(v):
    return "-" if v is None else "%.2e" % v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    rows, summ, recs = [], [], []
    for d in a.dirs.split(","):
        changed = inconclusive = 0
        ncase = 0
        for p in sorted(glob.glob(os.path.join(d, "case_*.json"))):
            r = json.load(open(p))
            u = r.get("uniqueness") or {}
            st = u.get("starts_record") or []
            if not st:
                continue
            ncase += 1
            nf = r.get("residual_noise_floor")
            thr = max(1e-11, 10 * nf) if nf is not None else 1e-11
            sols, nconv, neg, near = [], 0, 0, 0
            for x in st:
                res = x.get("own_residual")
                if res is None or x.get("returned") is None:
                    continue
                if res > thr:
                    continue
                zz = np.array(x["returned"], float)
                if not x.get("ok"):
                    near += 1
                if np.any(zz <= 0):
                    neg += 1
                    continue
                nconv += 1
                if not any(np.linalg.norm(zz - s) < 1e-8 for s in sols):
                    sols.append(zz)
            ref_in = None
            dmin = []
            if sols:
                for s in sols:
                    for x in st:
                        if x.get("returned") is not None and np.linalg.norm(np.array(x["returned"], float) - s) < 1e-14:
                            dmin.append(x.get("distance_to_reference"))
                            break
                ref_in = bool(any(v is not None and v < 1e-8 for v in dmin))
            dwith = len(sols) + (0 if (ref_in or not sols) else 1)
            if nconv == 0:
                wording = "scan inconclusive: 0 accepted starts"
            elif dwith <= 1:
                wording = "no second positive solution found in %d converged starts" % nconv
            else:
                wording = "%d distinct positive solutions, the continuation solution included, in %d converged starts" % (dwith, nconv)
            old = u.get("wording")
            ch = old != wording
            changed += int(ch)
            inconclusive += int(nconv == 0)
            case = os.path.basename(p).replace("case_", "").replace(".json", "")
            rec = {"record": os.path.basename(d.rstrip("/")), "case": case, "noise_floor": nf, "threshold": thr, "starts": len(st),
                   "old_converged_positive": u.get("converged_positive"), "old_distinct": u.get("distinct_positive"), "old_wording": old,
                   "new_converged_positive": nconv, "new_distinct": len(sols), "new_negative": neg, "newly_accepted_near_root": near,
                   "contains_continuation_solution": ref_in, "distinct_including_continuation": dwith, "distances_from_continuation": dmin,
                   "new_wording": wording, "verdict_changed": ch}
            recs.append(rec)
            rows.append("| %s | %s | %s | %s | %d | %s | %d | %d | %d | %s | %d | %s | %s | %s |" % (
                rec["record"], case, f(nf), f(thr), len(st), old, nconv, len(sols), near, ref_in, dwith, "; ".join(f(v) for v in dmin) or "-", wording,
                "yes" if ch else "no"))
        two = [r["case"] for r in recs if r["record"] == os.path.basename(d.rstrip("/")) and r["distinct_including_continuation"] >= 2]
        summ.append("%s: %d cases; the verdict wording changes in %d; scans with no accepted start after the rerun: %d; cases with a second positive "
                    "solution (the continuation solution included): %d%s." % (os.path.basename(d.rstrip("/")), ncase, changed, inconclusive, len(two),
                                                                               (" (" + ", ".join(two) + ")") if two else ""))
    out = ["# Strand 2b sec. 15 item 6(x): the uniqueness bookkeeping at the threshold max(1e-11, 10 x noise floor) (generated by "
           "solve/code/strand2/strand2b_sec15_item6x_v2.py)", "", "## S Summary (emitted)", ""] + ["- " + s for s in summ] + [
        "", "## I6x Per case: the residual noise floor, the threshold, the starts, the drivers' wording, the restated counts (accepted positive, "
            "distinct, of them accepted only by the new threshold), whether the positive cluster contains the continuation solution, the restated "
            "wording and whether it changed; version 2 also counts the continuation solution: the distinct positive solutions including it and the "
            "accepted roots' distances from it", "",
        "| record | case | noise floor | threshold | starts | drivers' wording | accepted positive | distinct | newly accepted | contains continuation | distinct incl. continuation | distances from it | restated wording | changed |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + rows
    open(a.md, "w").write("\n".join(out) + "\n")
    json.dump({"cases": recs}, open(a.json, "w"), indent=1)
    print("\n".join("- " + s for s in summ))
    for r in recs:
        if r["verdict_changed"] or r["new_converged_positive"] == 0:
            print(r["record"], r["case"], "|", r["old_wording"], "->", r["new_wording"])
    print("wrote", a.md)


if __name__ == "__main__":
    main()
