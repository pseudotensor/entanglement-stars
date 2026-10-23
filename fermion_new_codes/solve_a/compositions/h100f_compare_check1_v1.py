#!/usr/bin/env python
"""h100f_compare_check1_v1.py (solve 8a7b5274; adapted from box <internal>/compare_engine_test_v1.py, paths only, plus
leaf-level listing of every v3/v8 difference):  the records written by engine_twofield_v3.py and by engine_twofield_v8.py (--reference-solve implicit, its default)
restarted from the same version-8-written control slice at T/r_s 60 on the box CPU, compared leaf by leaf (json numbers, bitwise) and the fields
npz rows; both also compared against R2's stored (R2 GPU) records at the same clocks, where the hardware difference is measured."""
import json, sys, os
import numpy as np

BASE = "<internal>/out"
TAG = "C32_s4_ell4_tau2_twofield_v3_off"
R2J = "<internal>/collapse_%s.json" % TAG
R2F = "<internal>/collapse_%s_fields.npz" % TAG
SKIP = ("seconds", "timing", "slice_saved", "light_slice_saved", "engine_v34", "lag_s", "jvp_v35")


def leaves(o, path=(), out=None):
    if out is None:
        out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            if any(s in str(k) for s in SKIP):
                continue
            leaves(v, path + (str(k),), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            leaves(v, path + (str(i),), out)
    else:
        out[".".join(path)] = o
    return out


def recs_at(J, clocks):
    return {t: next(r for r in J["records"] if abs(r["T_over_rs"] - t) < 1e-9) for t in clocks}


def compare(A, B, labA, labB, clocks):
    for t in clocks:
        la, lb = leaves(A[t]), leaves(B[t])
        keys = sorted(set(la) | set(lb))
        rels = []; nnum = 0; ndiff = 0; worst = (0.0, None); missing = [k for k in keys if (k in la) != (k in lb)]
        for k in keys:
            if k in missing:
                continue
            a, b = la[k], lb[k]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
                nnum += 1
                if a != b:
                    ndiff += 1
                    rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
                    rels.append(rel)
                    if labA.startswith("v3") and labB.startswith("v8"):
                        print("   DIFF %s: %r vs %r (rel %.3g)" % (k, a, b, rel))
                    if rel > worst[0]:
                        worst = (rel, k, a, b)
            elif a != b:
                ndiff += 1
                if worst[1] is None:
                    worst = (0.0, k, a, b)
        print("%s vs %s at T/r_s %g: %d leaves, %d numbers, %d differ, missing on one side %d; worst %s" % (labA, labB, t, len(keys), nnum, ndiff, len(missing), worst))
        if rels:
            q = np.sort(np.array(rels)); print("   relative differences of differing numbers: median %.3g, 90th pct %.3g, 99th pct %.3g, max %.3g" % (q[len(q) // 2], q[int(0.9 * (len(q) - 1))], q[int(0.99 * (len(q) - 1))], q[-1]))
        if labA.startswith("v3") and labB.startswith("v8"):
            for k in keys:
                if k in la and k in lb and not (isinstance(la[k], (int, float)) and isinstance(lb[k], (int, float))) and la[k] != lb[k]:
                    print("   NONNUM DIFF %s: %r vs %r" % (k, str(la[k])[:120], str(lb[k])[:120]))
        if missing[:5]:
            print("   missing examples:", missing[:5])


def main():
    J3 = json.load(open(os.path.join(BASE, "check1_v3", "collapse_%s.json" % TAG)))
    J8 = json.load(open(os.path.join(BASE, "check1_v8", "collapse_%s.json" % TAG)))
    JR = json.load(open(R2J))
    print("scripts:", J3["script"], "|", J8["script"], "|", JR["script"])
    print("records written:", len(J3["records"]), len(J8["records"]), "last clocks", J3["records"][-1]["T_over_rs"], J8["records"][-1]["T_over_rs"])
    clocks = [r["T_over_rs"] for r in J3["records"] if r["T_over_rs"] > 60 + 1e-9]
    print("new records:", clocks)
    A3, A8, AR = recs_at(J3, clocks), recs_at(J8, clocks), recs_at(JR, clocks)
    compare(A3, A8, "v3 (h100-f)", "v8 implicit (h100-f)", clocks)
    compare(A3, AR, "v3 (h100-f)", "R2 stored (v8, GPU)", clocks)
    compare(A8, AR, "v8 (h100-f)", "R2 stored (v8, GPU)", clocks)
    # the fields npz rows
    F3 = np.load(os.path.join(BASE, "check1_v3", "collapse_%s_fields.npz" % TAG), allow_pickle=True)
    F8 = np.load(os.path.join(BASE, "check1_v8", "collapse_%s_fields.npz" % TAG), allow_pickle=True)
    FR = np.load(R2F, allow_pickle=True)
    T3, T8, TR = np.asarray(F3["T"], float), np.asarray(F8["T"], float), np.asarray(FR["T"], float)
    for t in clocks:
        i3, i8, iR = int(np.argmin(np.abs(T3 - t * 32))), int(np.argmin(np.abs(T8 - t * 32))), int(np.argmin(np.abs(TR - t * 32)))
        n_arr = 0; n_bit = 0; worst38 = 0.0; worst3R = 0.0
        for k in F3.files:
            a = F3[k]
            if a.ndim >= 1 and a.shape[0] == len(T3) and k != "T" and a.dtype != object:
                n_arr += 1
                b, c = F8[k][i8], FR[k][iR]
                if np.array_equal(a[i3], b):
                    n_bit += 1
                else:
                    worst38 = max(worst38, float(np.max(np.abs(a[i3] - b))))
                worst3R = max(worst3R, float(np.max(np.abs(a[i3] - c))))
                if not np.array_equal(a[i3], b):
                    print("   fields DIFF v3/v8 %s max %.3g" % (k, float(np.max(np.abs(a[i3] - b)))))
        print("fields npz at T/r_s %g: %d arrays, v3 vs v8 bitwise %d, worst |v3 - v8| %.3g, worst |v3 - R2| %.3g" % (t, n_arr, n_bit, worst38, worst3R))


if __name__ == "__main__":
    main()
