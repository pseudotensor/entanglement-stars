#!/usr/bin/env python
"""rd1c_readB_v1.py (busi, PERF RD-1c (c) part (B), as declared in busi/notes/PERF_RD1c_build_v1.md's amendment): from tolcmp_v2's outputs
v2_X_vs_P.json and v2_D_vs_P.json (X = the change arm, D = the default arm, P = the stored production):
  (1) every class of both arms within C at every record: E/E0 3.0e-7, roots 3.4e-6, depth 4.6e-8, the site fields (N, v, N_applied, v_applied,
      N_nodes, v_nodes) 2.5e-5;
  (2) the decision leaves differing X-P are the same set (record and key; X's and D's own values reported) as D-P (the change adds no decision difference against
      the production; the shared ones are listed by key as the ruling sec. 5 confound);
  (3) informative: X's distance from P at most twice D's per class and record (or both 0).
    python rd1c_readB_v1.py TOLCMP_OUT_DIR"""
import json
import os
import sys

C = {"E_over_E0": 3.0e-7, "roots": 3.4e-6, "depth": 4.6e-8}
FIELDS = ("fields/N", "fields/v", "fields/N_applied", "fields/v_applied", "fields/N_nodes", "fields/v_nodes")


def main():
    d = sys.argv[1]
    XP = json.load(open(os.path.join(d, "v2_X_vs_P.json"))); DP = json.load(open(os.path.join(d, "v2_D_vs_P.json")))
    oks = []; ratios = []
    for nm, J in (("X-P", XP), ("D-P", DP)):
        worst = {c: max(float(p.get(c, 0.0)) for p in J["per_record"].values()) for c in C}
        fw = max(float(J["rows"][k]["max_abs"]) for k in FIELDS if k in J["rows"])
        ok = all(worst[c] <= C[c] for c in C) and fw <= 2.5e-5
        oks.append(ok)
        print("(1) %s within C: %s (E/E0 %.2e, roots %.2e, depth %.2e, site fields %.2e; %d records)" % (
            nm, "PASS" if ok else "FAIL", worst["E_over_E0"], worst["roots"], worst["depth"], fw, len(J["per_record"])))
    # the declared rule: the same set of (record, key); the arm-side values are reported (a float leaf the comparator files as a decision,
    # such as the primitive's crossing samples, moves with the fields)
    kx = {q[0] for q in XP["decisions"]}; kd = {q[0] for q in DP["decisions"]}
    same = kx == kd
    oks.append(same)
    sx = {(q[0], json.dumps(q[2][0], default=str)) for q in XP["decisions"]}; sd = {(q[0], json.dumps(q[2][0], default=str)) for q in DP["decisions"]}
    print("(2) decision leaves X-P %d, D-P %d; the same (record, key) set: %s; only X-P %d, only D-P %d" % (len(kx), len(kd), "PASS" if same else "FAIL", len(kx - kd), len(kd - kx)))
    for q in sorted(kx - kd)[:10]:
        print("     key only X-P: %s" % q)
    for q in sorted(kd - kx)[:10]:
        print("     key only D-P: %s" % q)
    vk = sorted({q[0] for q in (sx ^ sd)})
    print("     reported: leaves where X's and D's own values differ (both differ from P): %d, e.g. %s" % (len(vk), vk[:4]))
    keys = {}
    for q in DP["decisions"]:
        k = "/" + q[0].split("/", 1)[1]; keys[k] = keys.get(k, 0) + 1
    print("     the shared D-P keys (records): %s" % ", ".join("%s %d" % kv for kv in sorted(keys.items())))
    bad = []
    for t, px in XP["per_record"].items():
        pd = DP["per_record"].get(t, {})
        for c in C:
            x, y = float(px.get(c, 0.0)), float(pd.get(c, 0.0))
            r = 0.0 if x == 0.0 else (float("inf") if y == 0.0 else x / y)
            ratios.append(r)
            if r > 2.0:
                bad.append((t, c, x, y))
    for k in FIELDS:
        if k in XP["rows"] and k in DP["rows"]:
            for t, x in XP["rows"][k]["per_record_abs"].items():
                y = DP["rows"][k]["per_record_abs"].get(t, 0.0)
                r = 0.0 if x == 0.0 else (float("inf") if y == 0.0 else x / y)
                ratios.append(r)
                if r > 2.0:
                    bad.append((t, k, x, y))
    print("(3) informative: X-P / D-P per class and record at most %.3f; above 2: %d %s" % (max(ratios) if ratios else 0.0, len(bad), bad[:6]))
    print("rd1c_readB_v1 (B): %s" % ("PASS" if all(oks) else "FAIL"))


if __name__ == "__main__":
    main()
