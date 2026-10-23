# Probe: the solve_v32 entry of the NK member at T 13.0, and the engine/label of records past 16.0
import sys, json
j = json.load(open(sys.argv[1])); R = j["records"]
for r in R:
    if abs(r["T_over_rs"] - 13.0) < 1e-9: print(json.dumps(r["solve_v32"])[:6000])
for r in R:
    if r["T_over_rs"] >= 15.75: print(r["T_over_rs"], r.get("engine"), r.get("converged"), r.get("iterations"), str(r.get("record_label"))[:200])
