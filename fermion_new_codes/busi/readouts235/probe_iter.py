# Probe: per member json, records count, T range, solver args, and one record's iteration_history / solver structure
import sys, json
for p in sys.argv[1:]:
    j = json.load(open(p)); R = j["records"]
    print("==", p); print("  n", len(R), "T", R[0]["T_over_rs"], R[-1]["T_over_rs"], "mix", j["args"].get("mix"), "script", j.get("script"))
    for r in R:
        if abs(r["T_over_rs"] - 13.0) < 1e-9:
            ih = r.get("iteration_history"); print("  iteration_history type", type(ih).__name__, (len(ih) if ih is not None else None))
            print("  ", json.dumps(ih)[:1500]); print("  solver", json.dumps(r.get("solver"))[:1500]); print("  iterations", r.get("iterations"), "converged", r.get("converged"))
