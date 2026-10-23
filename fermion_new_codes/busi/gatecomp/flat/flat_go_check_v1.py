#!/usr/bin/env python
"""flat_go_check_v1.py (busi, GATECOMP; ReviewA_devs_v1 (b), (d)): the GO read for a flat member. BOTH comparators before GO:
the card's (solve's dust_comparator_v1d on the engine's own projected T 0 profile: the member's dustcmp/run.out "fold ... caustic" line)
and the box's (the same comparator on the no-projection profile at the same sigma: its out/*.json). PASS when each has margin
T_c - T_birth >= 0.3 r_s and R_birth/2GE_0 >= 0.8, the two agree within 0.01 r_s in margin and 0.005 in R/2GE_0 (a larger difference is a
comparator defect, reported), the caustic sits inside the member's 0.25 r_s slice window [3.25, 6.0], and (for a member held behind
another) the leader's newest record has reached --after T.
    flat_go_check_v1.py CARD_RUN_OUT BOX_JSON [--leader-json J --after 3.25]"""
import argparse, json, re, sys

ap = argparse.ArgumentParser(); ap.add_argument("card"); ap.add_argument("box")
ap.add_argument("--leader-json", default=None); ap.add_argument("--after", type=float, default=None)
ap.add_argument("--window", default="3.25,6.0")
a = ap.parse_args()
L = [l for l in open(a.card) if re.search(r"control_continuum_read: shells", l)]
if not L:
    sys.exit("card comparator: no run line yet: NOT READY")
m = re.search(r"fold ([0-9.]+) at R ([0-9.]+) a \(([0-9.]+) of 2gE0\).*caustic ([0-9.]+)", L[-1])
cb, cR, cq, cc = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
J = json.load(open(a.box)); bb = J["fold"]["T_over_rs"]; bq = J["fold"]["R_over_2gE0"]; bc = J["centre"]["caustic"]["T_over_rs"]
cm, bm = cc - cb, bc - bb
lo, hi = [float(x) for x in a.window.split(",")]
checks = [("card margin >= 0.3", cm >= 0.3), ("card root >= 0.8", cq >= 0.8), ("box margin >= 0.3", bm >= 0.3), ("box root >= 0.8", bq >= 0.8),
          ("margins agree within 0.01 r_s (else a comparator defect)", abs(cm - bm) <= 0.01), ("roots agree within 0.005 (else a comparator defect)", abs(cq - bq) <= 0.005),
          ("caustic inside the 0.25 r_s window [%g, %g] (else extend the slices)" % (lo, hi), lo <= cc <= hi and lo <= bc <= hi)]
if a.leader_json:
    t = max(r["T_over_rs"] for r in json.load(open(a.leader_json))["records"])
    checks.append(("the leader has reached T %g (newest record %.4f)" % (a.after, t), t >= a.after - 1e-9))
print("card: T_birth %.5f R %.2f a (%.4f of 2GE) caustic %.5f margin %.4f | box: T_birth %.5f (%.4f) caustic %.5f margin %.4f | difference: margin %.4f r_s, root %.4f" % (
    cb, cR, cq, cc, cm, bb, bq, bc, bm, abs(cm - bm), abs(cq - bq)))
for n, ok in checks:
    print("  %-70s %s" % (n, "PASS" if ok else "FAIL"))
ok = all(c[1] for c in checks)
print("GO check: %s" % ("PASS: GO may be written" if ok else "FAIL: no GO"))
sys.exit(0 if ok else 1)
