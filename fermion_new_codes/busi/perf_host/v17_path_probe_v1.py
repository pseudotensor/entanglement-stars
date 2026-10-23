#!/usr/bin/env python
"""v17_path_probe_v1.py (busi, analyst; inbox #343): prints, per record of an h2_field_gate_v1 artifact made with engine_fields_v17, the
info["module"] and info["empty_interior"]["label"/"r_e"] leaves, i.e. which path v17 took (its own rule or v16's solve).
usage: python v17_path_probe_v1.py ART.pkl"""
import pickle
import sys
from collections import Counter

A = pickle.load(open(sys.argv[1], "rb"))
paths = Counter()
for rec in sorted(A):
    x = A[rec]
    mod = x.get("result[3]['module']")
    re_ = x.get("result[3]['empty_interior']['r_e']")
    lab = x.get("result[3]['empty_interior']['label']")
    key = (mod[1] if mod else None, None if re_ is None else re_[0])
    paths[key] += 1
    print(rec, "module:", mod[1] if mod else None, "| r_e:", re_, "| label:", (lab[1][:90] if lab else None))
print("paths:", dict(paths))
