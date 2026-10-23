#!/usr/bin/env python
"""read_v6j_gen_acceptance_v1.py (solve builder 5; inbox #74 solve task 2's acceptance read on the stored v6jA_s4gen run, light parsing only):
over every collecting checkpoint and channel of a page_history_v6j/k summary run with --collector-select gen: the range of the collected
modes' generalized eigenvalues (outside_weight_min/max) and the largest unselected one (unselected_g_max) against [0, 1] within 1e-12; the
2 residual columns (the archive's Gram deviation archive_gram_dev_F and the energy identity residual) and whether they differ; the joint
outside weight's smallest value; S_esc_weighted at the last checkpoint against a reference run's (the raw selector's v6bA_s4, 0.00114744).
Usage: read_v6j_gen_acceptance_v1.py <summary.json> <ref S_col at 40> <out.json>"""
import json, sys
d = json.load(open(sys.argv[1])); ref = float(sys.argv[2])
lo, hi, uhi, jmin = 1e300, -1e300, -1e300, 1e300; diff = 0; n = 0; gd = []; er = []
for r in d["rows"]:
    if not r.get("collect"):
        continue
    for k, c in r["per_channel"].items():
        if c.get("collector_select") != "gen":
            continue
        n += 1
        for key, f in (("outside_weight_min", min), ("outside_weight_max", max)):
            x = c.get(key)
            if x is not None:
                lo = min(lo, x) if key.endswith("min") else lo; hi = max(hi, x) if key.endswith("max") else hi
        if c.get("unselected_g_max") is not None:
            uhi = max(uhi, c["unselected_g_max"])
        if c.get("joint_outside_weight") is not None:
            jmin = min(jmin, c["joint_outside_weight"])
        a_, b_ = c.get("archive_gram_dev_F"), c.get("energy_identity_residual")
        gd.append(a_); er.append(b_)
        if a_ != b_:
            diff += 1
last = [r for r in d["rows"] if r.get("collect")][-1]
out = {"checkpoint_channel_entries": n, "collected_g_min": lo, "collected_g_max": hi, "unselected_g_max": uhi,
       "within_0_1_tol_1e-12": bool(lo >= -1e-12 and hi <= 1 + 1e-12 and uhi <= 1 + 1e-12), "joint_outside_weight_min": jmin,
       "residual_columns_differ_entries": diff, "archive_gram_dev_F_max": max(x for x in gd if x is not None),
       "energy_identity_residual_max_abs": max(abs(x) for x in er if x is not None),
       "last_t_over_rs": last["t_over_rs"], "S_esc_weighted_last": last["S_esc_weighted"], "ref_S_col": ref,
       "change": last["S_esc_weighted"] - ref, "change_rel": last["S_esc_weighted"] / ref - 1.0}
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[3], "w"), indent=1)
