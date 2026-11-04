#!/usr/bin/env python
"""compare_member_floor_v1.py (the lane's analyst builder; engine spec 41.3 (2), (4)): two collapse runs of the same arguments from the same
initial data (the v35 --nk-jvp exact run against the v32 or v34 run) compared at every common record clock at twice the floor (two runs
compared with each other, 41.3 (1): E/E0 3.0e-07, the roots 3.4e-06 lattice units, min(N - |v|) 4.6e-08; the floor itself, 1.5e-07, 1.7e-06,
2.3e-08, printed beside): E/E0, min(N - |v|), the trapping zeros where present (count and each root), the masks per channel (the spectra npz's
mask rows; equal or not), the Newton evaluations of each run and the first run's jvp_v35 entry; a record within twice the floor with every
mask equal reads "within"; the summary gives the last common clock and whether every common record is within. Every number printed by the
script. Usage: --run DIR/TAG --ref DIR/TAG [--label-run --label-ref] --md --out-json."""
import argparse
import json
import os
import time

import numpy as np

FLOOR = {"E_over_E0": 1.5e-07, "roots": 1.7e-06, "min_N_minus_absv": 2.3e-08}
TWICE = {k: 2.0 * v for k, v in FLOOR.items()}


def load(p):
    d, t = os.path.split(p)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % t)))
    sp = os.path.join(d, "collapse_%s_spectra.npz" % t)
    S = np.load(sp) if os.path.exists(sp) else None
    return J, S


def mask_at(S, T, k):
    if S is None or ("mask_%d" % k) not in S.files:
        return None
    Ts = np.asarray(S["T"], float); i = int(np.argmin(np.abs(Ts - T)))
    if abs(Ts[i] - T) > 1e-9:
        return None
    return np.asarray(S["mask_%d" % k][i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--label-run", default="run"); ap.add_argument("--label-ref", default="reference")
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    J, S = load(a.run); Jr, Sr = load(a.ref)
    rs = float(J["args"]["rs"]); kap = list(range(1, int(J["args"]["kappa_sea"]) + 1))
    R = {round(r["T"], 6): r for r in J["records"] if r.get("E") is not None}
    Q = {round(r["T"], 6): r for r in Jr["records"] if r.get("E") is not None}
    common = sorted(set(R) & set(Q))
    rows = []
    for t in common:
        r, q = R[t], Q[t]
        zr = r.get("trapping_zeros") or []; zq = q.get("trapping_zeros") or []
        dE = r["E_over_E0"] - q["E_over_E0"]; dm = r["min_N_minus_absv"] - q["min_N_minus_absv"]
        dz = [a_ - b_ for a_, b_ in zip(zr, zq)] if len(zr) == len(zq) else None
        meq = []
        for k in kap:
            x, y = mask_at(S, t, k), mask_at(Sr, t, k)
            meq.append(None if (x is None or y is None) else bool(np.array_equal(x, y)))
        masks_ok = all(m is not False for m in meq)
        within = bool(abs(dE) <= TWICE["E_over_E0"] and abs(dm) <= TWICE["min_N_minus_absv"] and dz is not None and all(abs(d) <= TWICE["roots"] for d in dz) and masks_ok)
        ev = lambda x: ((x.get("solve_v32") or {}).get("evaluations") if x.get("solve_v32") else x.get("iterations"))  # noqa: E731
        rows.append({"T_over_rs": t / rs, "E_over_E0": r["E_over_E0"], "E_over_E0_ref": q["E_over_E0"], "dE": dE, "min": r["min_N_minus_absv"], "min_ref": q["min_N_minus_absv"],
                     "dmin": dm, "roots": zr, "roots_ref": zq, "droots": dz, "masks_equal": meq, "within_twice_floor": within, "evaluations": ev(r), "evaluations_ref": ev(q),
                     "seconds": r.get("seconds"), "seconds_ref": q.get("seconds"), "jvp_v35": r.get("jvp_v35")})
    allw = bool(rows) and all(x["within_twice_floor"] for x in rows)
    first_fail = next((x["T_over_rs"] for x in rows if not x["within_twice_floor"]), None)
    L = ["# compare_member_floor_v1: %s against %s at twice the floor (engine spec 41.3 (1), (2), (4))" % (a.label_run, a.label_ref), "",
         "Composed %s by code/engine/compare_member_floor_v1.py from %s and %s; every number printed by the script. Twice the floor: E/E0 %.1e, roots %.1e, min(N - |v|) %.1e "
         "(the floor %.1e, %.1e, %.1e)." % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), a.run, a.ref, TWICE["E_over_E0"], TWICE["roots"], TWICE["min_N_minus_absv"],
                                             FLOOR["E_over_E0"], FLOOR["roots"], FLOOR["min_N_minus_absv"]), "",
         "- common records: %d (T/r_s %s to %s); every one within twice the floor with the masks equal: %s%s" % (
             len(rows), ("%.4f" % rows[0]["T_over_rs"]) if rows else "-", ("%.4f" % rows[-1]["T_over_rs"]) if rows else "-", "yes" if allw else "no",
             ("" if allw or first_fail is None else " (first not within: T/r_s %.4f)" % first_fail)), "",
         "| T/r_s | E/E0 run | difference | min(N - abs(v)) run | difference | roots run | root differences | masks equal | within | evaluations run; ref | wall s run; ref | exact products; s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        jv = x["jvp_v35"] or {}
        L.append("| %.4f | %.8f | %+.2e | %+.6f | %+.2e | %s | %s | %s | %s | %s; %s | %s; %s | %s; %s |" % (
            x["T_over_rs"], x["E_over_E0"], x["dE"], x["min"], x["dmin"], ", ".join("%.6f" % z for z in x["roots"]) or "none",
            ", ".join("%+.2e" % d for d in x["droots"]) if x["droots"] is not None else ("count %d against %d" % (len(x["roots"]), len(x["roots_ref"]))),
            "%d of %d" % (sum(1 for m in x["masks_equal"] if m), len(x["masks_equal"])) if all(m is not None for m in x["masks_equal"]) else "not stored",
            "yes" if x["within_twice_floor"] else "no", x["evaluations"], x["evaluations_ref"], ("%.0f" % x["seconds"]) if x["seconds"] else "-",
            ("%.0f" % x["seconds_ref"]) if x["seconds_ref"] else "-", jv.get("products", "-"), ("%.1f" % jv["seconds"]) if jv.get("seconds") is not None else "-"))
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"script": "compare_member_floor_v1.py", "run": a.run, "ref": a.ref, "twice_floor": TWICE, "all_within": allw, "first_not_within": first_fail, "rows": rows},
              open(a.out_json, "w"), indent=1, default=float)
    print("compared %d records; all within twice the floor: %s" % (len(rows), allw))


if __name__ == "__main__":
    main()
