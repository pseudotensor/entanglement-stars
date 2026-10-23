#!/usr/bin/env python
"""compare_resume_gate_v1.py (busi lane, Computation C; the resume gate of the resumability audit, inbox #86): a run restarted from a stored
complex128 orbitals slice (engine_twofield_v3 --resume-slice --resume-run, version 34 (a) with light_resume_v36) against the stored run over the
records after the slice's clock. Every number of each record json (minus gate_regression_v2b's SKIP_KEYS: labels, wall clocks, the kernel cache's
counters, the resume's own head) must be identical, and every time-sliced array of the fields npz at those clocks. Verdict BITWISE; otherwise the
first differing field, and the floor table of gate_regression_v2b (E/E0 3.0e-7 relative, R_t and R_sh 3.4e-6, min(N - |v|) 4.6e-8, the masks
equal), verdict WITHIN-FLOOR or FAIL. The stored run is read from a snapshot (pod_launch_C_v3.sh copies it), never from a live run's files.
Usage: python compare_resume_gate_v1.py --stored-dir SNAP --gate-dir GATE --tag TAG --slice-T-over-rs TS [--out report.json]"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate_regression_v2b import compare_tree, arrays_equal, floor_table  # noqa: E402

TIME_SLICED = ("N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied", "r_nodes", "N_nodes", "v_nodes")


def recs_after(J, ts):
    return [r for r in J["records"] if r.get("T_over_rs") is not None and r["T_over_rs"] > ts + 1e-9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stored-dir", required=True)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--slice-T-over-rs", type=float, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    js = json.load(open(os.path.join(a.stored_dir, "collapse_%s.json" % a.tag)))
    jg = json.load(open(os.path.join(a.gate_dir, "collapse_%s.json" % a.tag)))
    rg = recs_after(jg, a.slice_T_over_rs)
    rs_all = {round(r["T_over_rs"], 9): r for r in recs_after(js, a.slice_T_over_rs)}
    rows = []; first = None; leaves = 0; matched_s = []; matched_g = []
    for q in rg:
        p = rs_all.get(round(q["T_over_rs"], 9))
        if p is None:
            first = first or "T/rs %.4f: no stored record at this clock" % q["T_over_rs"]
            continue
        st = {"leaves": 0}
        d = compare_tree(p, q, "record[T/rs %.4f]" % q["T_over_rs"], st)
        leaves += st["leaves"]
        rows.append("T/rs %.4f: %s (%d numbers)" % (q["T_over_rs"], "identical" if d is None else d, st["leaves"]))
        first = first or d
        matched_s.append(p); matched_g.append(q)
    if not rg:
        first = "the gate run holds no record after the slice's clock"
    fa = np.load(os.path.join(a.stored_dir, "collapse_%s_fields.npz" % a.tag), allow_pickle=True)
    fb = np.load(os.path.join(a.gate_dir, "collapse_%s_fields.npz" % a.tag), allow_pickle=True)
    Ta = np.asarray(fa["T"], float); Tb = np.asarray(fb["T"], float); rsv = float(js["args"]["rs"])
    arrays = 0
    for i, t in enumerate(Tb):
        if t / rsv <= a.slice_T_over_rs + 1e-9:
            continue
        j = np.where(np.abs(Ta - t) < 1e-9)[0]
        if not len(j):
            first = first or "fields: no stored row at T/rs %.4f" % (t / rsv)
            continue
        for k in TIME_SLICED:
            if k in fa.files and k in fb.files:
                arrays += 1
                if not arrays_equal(fa[k][j[0]], fb[k][i]):
                    first = first or "fields %s at T/rs %.4f differ" % (k, t / rsv)
    verdict = "BITWISE" if first is None else None
    ftab = []
    if verdict is None and matched_s:
        ok, ftab = floor_table(matched_s, matched_g, None, None, len(matched_s))
        verdict = "WITHIN-FLOOR (masks not compared)" if ok else "FAIL"
    elif verdict is None:
        verdict = "FAIL"
    rep = {"tag": a.tag, "slice_T_over_rs": a.slice_T_over_rs, "records_compared": len(matched_g), "json_numbers": leaves,
           "field_arrays": arrays, "first_difference": first, "verdict": verdict, "rows": rows, "floor_table": ftab}
    print("\n".join(rows + ftab))
    print("resume gate %s: %d records after T/rs %.4f, %d json numbers, %d field arrays; first difference: %s; VERDICT %s" % (
        a.tag, len(matched_g), a.slice_T_over_rs, leaves, arrays, first, verdict))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)
    sys.exit(0 if verdict.startswith(("BITWISE", "WITHIN")) else 1)


if __name__ == "__main__":
    main()
