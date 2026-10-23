#!/usr/bin/env python
"""flat_dry_read_v1.py (busi, GATECOMP flat build): reads flat_dry_v1.sh's outputs.
(1) per staged line and directory: the child's parsed vars(a) equals the parent's on every parent key, and its extra keys are exactly
    prep=well, prep_R=None, prep_edge=hard, prep_taper=None;
(2) per fixture: the child's printed preparation / restart lines equal the parent's stored prep.log line for line (the lines that carry
    numbers: 'initial data', 'dust set', 'version 34', 'restart', 'PREP OK', 'refus'), timing fields '(N s)' masked; the child's log has no
    'flat prep' line.
    python flat_dry_read_v1.py KITS_DIR FLAT_DRY_DIR"""
import json, os, re, sys

K, D = sys.argv[1], sys.argv[2]
NEW = {"prep": "well", "prep_R": None, "prep_edge": "hard", "prep_taper": None}
bad = 0; n = 0
for L in ("lines", "lines_dense"):
    for nm in sorted(os.listdir(os.path.join(D, "parse", L))):
        a = json.load(open(os.path.join(D, "parse", L, nm, "P", "dry_args_parse.json")))
        b = json.load(open(os.path.join(D, "parse", L, nm, "X", "dry_args_parse.json")))
        extra = {k: b[k] for k in b if k not in a}
        diff = [k for k in a if a[k] != b.get(k)]
        ok = (not diff) and extra == NEW
        n += 1; bad += (not ok)
        print("parse %-11s %-12s parent %d keys, child %d; differing %s; extra %s -> %s" % (L, nm, len(a), len(b), diff, extra, "OK" if ok else "FAIL"))
KEY = re.compile(r"initial data|dust set|version 34|restart|PREP OK|refus|REFUS|the run's argument|state rebuilt", re.I)
mask = lambda s: re.sub(r"\(\d+ s\)|in [\d.]+ s|[\d.]+ s\b", "(s)", s)  # noqa: E731
for nm in ("FXP_cv", "FXP_w", "FX_tau", "FX_plain", "FX_point", "FX_tight", "FX_tau_noview"):
    pa = [mask(l.rstrip("\n")) for l in open(os.path.join(K, "dry", "fx", nm, "prep.log")) if KEY.search(l)]
    pb_all = open(os.path.join(D, "prep", nm, "prep.log")).read()
    pb = [mask(l) for l in pb_all.splitlines() if KEY.search(l)]
    pb = [l.replace(os.path.join(D, "prep", nm), "OUT") for l in pb]; pa = [l.replace(os.path.join(K, "dry", "fx", nm), "OUT") for l in pa]
    ok = pa == pb and "flat prep" not in pb_all and len(pa) > 0
    n += 1; bad += (not ok)
    print("prep  %-14s %d keyed lines parent, %d child; equal %s; flat lines %d -> %s" % (nm, len(pa), len(pb), pa == pb, pb_all.count("flat prep"), "OK" if ok else "FAIL"))
    if pa != pb:
        for x, y in zip(pa, pb):
            if x != y:
                print("   parent: %s\n   child:  %s" % (x[:300], y[:300])); break
print("flat_dry_read_v1: %d items, %d FAIL -> %s" % (n, bad, "PASS" if bad == 0 else "FAIL"))
sys.exit(0 if bad == 0 else 1)
