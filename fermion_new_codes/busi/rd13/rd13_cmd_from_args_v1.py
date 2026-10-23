#!/usr/bin/env python
"""rd13_cmd_from_args_v1.py (busi, analyst; RD-1/RD-3 replay): a run's command line from its json's args (w2_cmd_from_args.py's rule:
None, False and empty strings dropped, True as a bare flag), the resume and segment keys dropped, KEY=VALUE overrides applied.
    python rd13_cmd_from_args_v1.py RUN_JSON [key=value ...]"""
import json
import sys

J = json.load(open(sys.argv[1])); A = dict(J["args"])
A.update(dict(q.split("=", 1) for q in sys.argv[2:]))
drop = {"resume_from", "resume_T_over_rs", "resume_grandparent", "resume_slice", "resume_run", "segment_end_T_over_rs"}
parts = []
for k, v in A.items():
    if v is None or v is False or k in drop or (isinstance(v, str) and v == ""):
        continue
    flag = "--" + k.replace("_", "-")
    parts.append(flag if v is True else "%s %s" % (flag, v))
print(" ".join(parts))
