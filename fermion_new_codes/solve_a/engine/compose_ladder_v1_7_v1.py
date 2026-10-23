#!/usr/bin/env python
"""compose_ladder_v1_7_v1.py (solve builder 12; engine spec 54.1 (2)): COLLAPSE_ladder_v1.7 = the sigma/a 2 point's seam and the flip's
record-step term (results/engine/V36D/seam_v36d_vs_member_to18125_v1.md, whole) as the head, version 1.6 below as the record, unchanged.
Refuses to overwrite an existing delivery."""
import os
D = "<internal>/solve_a"
H = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(H, "..", "..", "results", "engine")
out = os.path.join(D, "COLLAPSE_ladder_v1.7.md")
assert not os.path.exists(out), out
seam = open(os.path.join(E, "V36D", "seam_v36d_vs_member_to18125_v1.md")).read().split("\n", 1)[1]
v16 = open(os.path.join(D, "COLLAPSE_ladder_v1.6.md")).read()
L = ["# COLLAPSE_ladder_v1.7: the sigma/a 2 point's seam to T/r_s 18.125 and the flip's record-step term (engine spec 54.1); version 1.6 below as the record", "",
     "Version 1.6 stands as the record. This version adds the point's run from 17.625 (v36d, its label as ruled in 54.1 (2)), the seam table to 18.125 "
     "with both kept residuals (seam terms at 17.625 to 17.875, within at 18.0 and 18.125) and the stopped run's departure as the flip's record-step term. "
     "Past 18.125, 52.1 (2) governs.", "", "## The point's seam (results/engine/V36D/seam_v36d_vs_member_to18125_v1.md, whole)", seam.rstrip("\n"), "",
     "## Version 1.6 (the record, unchanged)", "", v16]
open(out, "w").write("\n".join(L))
print("written", out, os.path.getsize(out))
