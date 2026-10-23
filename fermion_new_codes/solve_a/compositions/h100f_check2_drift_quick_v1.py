#!/usr/bin/env python
"""h100f_check2_drift_quick_v1.py (solve 8a7b5274): quick drift look for Check 2 while it runs: copies the json first (the engine rewrites it per
record), then prints h100-f minus R2 relative differences of E/E0, H_in and the heavy energy beyond 16 and 24 r_s per quarter clock."""
import json, shutil
W = "<internal>/comp1"; TAG = "C32_s4_ell4_tau2_twofield_v3_off"
shutil.copy(W + "/out/check2_v3/collapse_%s.json" % TAG, W + "/out/check2_snapshot.json")
JH = json.load(open(W + "/out/check2_snapshot.json")); JR = json.load(open(W + "/in/R2run/collapse_%s.json" % TAG))
R = {round(r["T_over_rs"], 4): r for r in JR["records"]}
rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)
print("T/rs   E/E0      H_in      beyond16  beyond24  (relative |h100-f - R2|)")
for r in JH["records"]:
    t = round(r["T_over_rs"], 4)
    if t <= 50 or abs(t * 4 - round(t * 4)) > 1e-9 and t != 50.0625:
        continue
    q = R[t]
    s16 = (r["s11"]["ball_beyond_16"]["energy_density_route"], q["s11"]["ball_beyond_16"]["energy_density_route"])
    s24 = (r["s11"]["ball_beyond_24"]["energy_density_route"], q["s11"]["ball_beyond_24"]["energy_density_route"])
    print("%7.4f %.2e %.2e %.2e %.2e" % (t, rel(r["E_over_E0"], q["E_over_E0"]), rel(r["H_in"], q["H_in"]), rel(*s16), rel(*s24)))
