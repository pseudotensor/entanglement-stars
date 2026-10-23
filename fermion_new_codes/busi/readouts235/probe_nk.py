# Probe: NK member copies, record counts, keys of the record at T 13.0 that relate to the Newton-Krylov solve
import sys, json
for p in sys.argv[1:]:
    j = json.load(open(p)); R = j["records"]
    print("==", p, "n", len(R), "T", R[0]["T_over_rs"], R[-1]["T_over_rs"], "script", j.get("script"))
print("args", json.dumps(j["args"])[:1500])
for r in R:
    if abs(r["T_over_rs"] - 13.0) < 1e-9:
        print("keys", list(r.keys()))
        for k in r:
            if any(s in k.lower() for s in ("nk", "newton", "resid", "iter", "krylov", "gmres", "line")):
                print(k, json.dumps(r[k])[:2500])
