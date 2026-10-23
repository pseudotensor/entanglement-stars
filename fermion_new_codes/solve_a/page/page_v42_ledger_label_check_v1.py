#!/usr/bin/env python
"""PAGE v4.2 issue 6 (inbox #205): which quantity v4.1 T7(a)'s second_term_D is. From the stored per-channel windows of the B2 production
kappa 1 and 2 d13 json (weights 16 and 32, the v4.1 producer's sum): the stored second_term_D (page_d13_readout_v1d.ledger_windows:
crossing_current + crossing_jumps - vacuum_change), the endpoint difference inventory_D_end - inventory_D_start, the telescoping residual
change_C - crossing_total, and the identity second_term_D = endpoint difference - telescoping residual. Stored data only.
Usage: python page_v42_ledger_label_check_v1.py PROD_DIR OUT_JSON"""
import sys, json, os

PROD = sys.argv[1]; OUT = sys.argv[2]
W = {1: 16.0, 2: 32.0}
D = {k: json.load(open(os.path.join(PROD, "v6o_B2_k%d_d13.json" % k)))["windows"][str(k)] for k in W}
idx = {k: {(w["t0_over_rs"], w["t1_over_rs"], w["cut"]): w for w in D[k] if w["cut"] in ("16", "30")} for k in W}
F = ("second_term_D", "inventory_D_start", "inventory_D_end", "telescoping_residual", "crossing_current", "crossing_jumps", "crossing_total", "vacuum_change", "change_C")
rows = []
for key in sorted(idx[1]):
    if key not in idx[2]:
        continue
    r = {"t0": key[0], "t1": key[1], "cut": key[2]}
    ok = True
    for f in F:
        v = [idx[k][key].get(f) for k in W]
        if any(x is None for x in v):
            ok = False; break
        r[f] = sum(W[k] * idx[k][key][f] for k in W)
    if not ok:
        continue
    r["endpoint_difference"] = r["inventory_D_end"] - r["inventory_D_start"]
    r["second_minus_endpoint"] = r["second_term_D"] - r["endpoint_difference"]
    r["identity_check"] = r["second_minus_endpoint"] + r["telescoping_residual"]
    r["rel_gap"] = abs(r["second_minus_endpoint"]) / abs(r["endpoint_difference"]) if r["endpoint_difference"] else None
    rows.append(r)
json.dump(rows, open(OUT, "w"), indent=1)
print("t0 t1 cut | second_term_D | endpoint diff | telescoping residual | second - endpoint | identity (should be 0) | rel gap | crossing_current")
for r in rows:
    if r["t0"] >= 20:
        print("%5.1f %5.1f %s | %+.10e | %+.10e | %+.10e | %+.10e | %+.2e | %.3e | %+.10e" % (r["t0"], r["t1"], r["cut"], r["second_term_D"], r["endpoint_difference"],
              r["telescoping_residual"], r["second_minus_endpoint"], r["identity_check"], r["rel_gap"], r["crossing_current"]))
