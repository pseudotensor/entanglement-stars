#!/usr/bin/env python
"""gate_v36d_relaunch_v1.py (solve builder 11, analyst; engine spec 53.1 (3)): the v36d relaunch's own gate, light parsing only.
(a) Records 17.125 to 17.5 against the v36c run's: every numeric leaf of each record (timing keys excluded: names containing 'second',
    'utc', 'wall', 'time' or ending in '_s'; the kernel caches' hit and miss counts; the keys v36d adds) and every row of the spectra and fields npz arrays at those clocks; BITWISE when all are equal;
    else the E/E0, min(N - |v|) and root differences against the floor (1.5e-07, 2.3e-08, 1.7e-06): ABOVE-FLOOR is the stop condition.
(b) The 17.625 reading: the relaunch's record set against its 17.5 record's (equals_previous_record), its v36d entry (the other solve), the
    solve's v36c label and kept round. Prints one summary line and writes --out-json and --md.
Usage: gate_v36d_relaunch_v1.py --run DIR/TAG --ref DIR/TAG --md F --out-json F"""
import argparse
import json
import os
import time

import numpy as np

FLOOR = {"E_over_E0": 1.5e-07, "roots": 1.7e-06, "min_N_minus_absv": 2.3e-08}
SKIP = ("second", "utc", "wall", "time", "hits", "misses", "kcache")     # timings and the kernel caches' counts (warm-up after a restart)
V36D_KEYS = ("kept_gap", "flips_vs_previous", "cross_lower", "e_prev_after")      # keys version 36d adds


def skip(k):
    k = k.lower()
    return any(s in k for s in SKIP) or k.endswith("_s") or k in V36D_KEYS


def leaves(x, y, p, out):
    if isinstance(x, dict):
        for k in x:
            if skip(k):
                continue
            leaves(x[k], y.get(k) if isinstance(y, dict) else None, p + "/" + k, out)
    elif isinstance(x, list):
        if not isinstance(y, list) or len(x) != len(y):
            out["structural"].append(p); return
        for i, (u, v) in enumerate(zip(x, y)):
            leaves(u, v, p + "[%d]" % i, out)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out["n"] += 1
        if x != y:
            out["unequal"].append((p, x, y))


def load(p):
    d, t = os.path.split(p)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % t)))
    arr = {}
    for suf in ("spectra", "fields"):
        f = os.path.join(d, "collapse_%s_%s.npz" % (t, suf))
        arr[suf] = np.load(f, allow_pickle=True) if os.path.exists(f) else None
    return J, arr


def rows_at(Z, T):
    if Z is None or "T" not in Z.files:
        return None
    Ts = np.asarray(Z["T"], float); i = np.nonzero(np.abs(Ts - T) < 1e-9)[0]
    return int(i[0]) if i.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--md", required=True); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    J, A = load(a.run); Jr, Ar = load(a.ref)
    rs = float(J["args"]["rs"])
    R = {round(r["T"] / rs, 6): r for r in J["records"] if r.get("E") is not None}
    Q = {round(r["T"] / rs, 6): r for r in Jr["records"] if r.get("E") is not None}
    rows = []
    for t in sorted(x for x in R if 17.0 < x <= 17.5 + 1e-9):
        r, q = R[t], Q.get(t)
        if q is None:
            continue
        out = {"n": 0, "unequal": [], "structural": []}
        leaves(r, q, "", out)
        arr_n, arr_neq = 0, []
        for suf in ("spectra", "fields"):
            Z, Zr = A[suf], Ar[suf]
            i, j = rows_at(Z, r["T"]), rows_at(Zr, q["T"])
            if i is None or j is None:
                arr_neq.append("%s: row missing" % suf); continue
            for k in Z.files:
                if k == "T" or k not in Zr.files:
                    continue
                x, y = np.asarray(Z[k]), np.asarray(Zr[k])
                if x.ndim == 0 or x.shape[0] != Z["T"].shape[0] or y.shape[0] != Zr["T"].shape[0]:
                    continue
                arr_n += 1
                if not np.array_equal(x[i], y[j]):
                    arr_neq.append("%s/%s" % (suf, k))
        zr = r.get("trapping_zeros") or []; zq = q.get("trapping_zeros") or []
        dE = r["E_over_E0"] - q["E_over_E0"]; dm = r["min_N_minus_absv"] - q["min_N_minus_absv"]
        dz = [x - y for x, y in zip(zr, zq)] if len(zr) == len(zq) else None
        above = bool(abs(dE) > FLOOR["E_over_E0"] or abs(dm) > FLOOR["min_N_minus_absv"] or dz is None or any(abs(d) > FLOOR["roots"] for d in dz))
        bit = not out["unequal"] and not out["structural"] and not arr_neq
        rows.append({"T_over_rs": t, "numbers": out["n"], "unequal": len(out["unequal"]), "first_unequal": out["unequal"][:5],
                     "structural": out["structural"][:5], "array_rows": arr_n, "array_rows_unequal": arr_neq[:10], "bitwise": bit,
                     "dE": dE, "dmin": dm, "droots": dz, "above_floor": above})
    r625 = R.get(17.625); r5 = R.get(17.5)
    read = None
    if r625 is not None:
        s = (r625.get("solve_v32") or {}).get("solves") or [{}]
        s = s[-1]
        vc = s.get("v36c") or {}
        read = {"record_set": (r625.get("v36c") or {}).get("record_set"), "set_17_5": (r5.get("v36c") or {}).get("record_set") if r5 else None,
                "equals_previous_record": (r625.get("v36c") or {}).get("equals_previous_record"), "label": (r625.get("v36c") or {}).get("label"),
                "v36d": r625.get("v36d"), "solve_v36d": s.get("v36d"), "kept_round": vc.get("kept_round"), "status": s.get("status"),
                "kept_residual": (r625.get("solve_v32") or {}).get("kept_residual"), "confirming_residual": (r625.get("solve_v32") or {}).get("confirming_residual"),
                "E_over_E0": r625.get("E_over_E0"), "min_N_minus_absv": r625.get("min_N_minus_absv"), "trapping_zeros": r625.get("trapping_zeros"),
                "converged": r625.get("converged")}
    allbit = bool(rows) and all(x["bitwise"] for x in rows)
    anyabove = any(x["above_floor"] for x in rows)
    verdict = ("BITWISE" if allbit else ("ABOVE-FLOOR" if anyabove else "WITHIN-FLOOR")) + " on %d of 4 records 17.125-17.5" % len(rows)
    if read is not None:
        verdict += "; 17.625 record set %s 17.5's (kept round %s, label %s)" % ("EQUALS" if read["equals_previous_record"] else "DIFFERS FROM", read["kept_round"], read["label"])
    L = ["# gate_v36d_relaunch_v1: the v36d relaunch against the v36c run (engine spec 53.1 (3))", "",
         "Composed %s by code/engine/gate_v36d_relaunch_v1.py from %s and %s." % (time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), a.run, a.ref), "",
         "- verdict: " + verdict, "",
         "| T/r_s | json numbers | unequal | array rows | rows unequal | bitwise | dE/E0 | d min(N-|v|) | root differences | above the floor |", "|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        L.append("| %.4f | %d | %d | %d | %s | %s | %+.2e | %+.2e | %s | %s |" % (x["T_over_rs"], x["numbers"], x["unequal"], x["array_rows"], ", ".join(x["array_rows_unequal"]) or "0",
                 "yes" if x["bitwise"] else "no", x["dE"], x["dmin"], ", ".join("%+.2e" % d for d in x["droots"]) if x["droots"] is not None else "count differs",
                 "YES" if x["above_floor"] else "no"))
    if read is not None:
        L += ["", "## 17.625", "", "```", json.dumps(read, indent=1, default=float), "```"]
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump({"verdict": verdict, "bitwise_17125_175": allbit, "above_floor": anyabove, "rows": rows, "read_17625": read}, open(a.out_json, "w"), indent=1, default=float)
    print(verdict)


if __name__ == "__main__":
    main()
