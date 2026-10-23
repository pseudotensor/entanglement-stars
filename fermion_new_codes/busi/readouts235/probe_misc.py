# Probe: npz T units for the s8 member; W2 json record keys and E; v3b definitions of W_R
import json, numpy as np, sys
z = np.load(sys.argv[1], allow_pickle=True); print("T first/last", z["T"][:3], z["T"][-1], "R", z["R"][:3], z["R"][-1])
j = json.load(open(sys.argv[2])); R = j["records"]; print("W2 n", len(R), "keys", list(R[0].keys())[:90])
for r in R[:2] + R[-4:]: print(r.get("T_over_rs"), r.get("E"), r.get("E_over_E0"), r.get("converged"), str(r.get("record_label"))[:150])
print("W2 args", json.dumps(j.get("args"))[:1200]); print("head E0", j.get("head", {}).get("E0"))
v = json.load(open(sys.argv[3])); print(json.dumps(v["definitions"])[:3000])
