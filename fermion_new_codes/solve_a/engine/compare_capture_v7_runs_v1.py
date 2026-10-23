"""compare_capture_v7_runs_v1.py (builder analyst): two engine_capture_v7 JSONs of the same arguments (e.g. --workers 1 against
--workers N): the channel lists, every per-channel unitary reading and monitor value (bitwise count and largest difference), the absorbing
port (largest difference), the summed sigma and the settle decisions."""
import json
import sys

A = json.load(open(sys.argv[1])); B = json.load(open(sys.argv[2]))
UNIT = ("A", "R_out", "outside_plain", "inward_inside", "plain_inside", "inward_transit", "beyond_Rmax")
for ra in A["results"]:
    rb = [r for r in B["results"] if r["background"] == ra["background"] and r["probe"] == ra["probe"]]
    if not rb:
        print("missing in the second:", ra["background"], ra["probe"]); continue
    rb = rb[0]
    ka = [c["kappa"] for c in ra["channels"]]; kb = [c["kappa"] for c in rb["channels"]]
    n = nbit = 0; dmax = 0.0; dabs = 0.0; nabs = 0
    for ca, cb in zip(ra["channels"], rb["channels"]):
        for s in ("plus", "minus"):
            xa, xb = ca[s], cb[s]
            if ("error" in xa) != ("error" in xb):
                print("error flag differs at", ca["kappa"], s); continue
            if "error" in xa:
                continue
            vals = [(xa["I"], xb["I"]), (xa["bound"], xb["bound"])]
            for f in UNIT:
                if set(xa[f]) != set(xb[f]):
                    print("reading times differ", f, ca["kappa"], s)
                vals += [(xa[f][k], xb[f].get(k)) for k in xa[f]]
            for ma, mb in zip(xa["monitor"], xb["monitor"]):
                vals += [(ma["t"], mb["t"]), (ma["inside_outward"], mb["inside_outward"]), (ma["inside_total"], mb["inside_total"])]
            if len(xa["monitor"]) != len(xb["monitor"]):
                print("monitor lengths differ", ca["kappa"], s)
            for u, w in vals:
                n += 1; nbit += (u == w)
                if w is not None and u != w:
                    dmax = max(dmax, abs(u - w))
            for k in xa.get("absorbed", {}):
                nabs += 1; dabs = max(dabs, abs(xa["absorbed"][k] - xb["absorbed"][k]))
    sig_bit = all(ra["sigma"][k] == rb["sigma"][k] for k in ra["sigma"]) and set(ra["sigma"]) == set(rb["sigma"])
    dec_a = [(s["T_over_rs"], s["settled"], s["channels"]) for s in ra["steps"]]; dec_b = [(s["T_over_rs"], s["settled"], s["channels"]) for s in rb["steps"]]
    print("%s | %s: channel lists equal %s (%d, %d); unitary values compared %d, bitwise equal %d, largest difference of the rest %.3e; absorbing port values %d, largest difference %.3e; sigma bitwise equal %s; the steps' decisions equal %s"
          % (ra["background"][:12], ra["probe"], ka == kb, len(ka), len(kb), n, nbit, dmax, nabs, dabs, sig_bit, dec_a == dec_b))
    for s in ra["steps"]:
        print("   first run: T %g r_s, channels %d, step %.1f s, probe %.1f s" % (s["T_over_rs"], s["channels"], s["step_s"], s["probe_s"]))
    for s in rb["steps"]:
        print("   second run: T %g r_s, channels %d, step %.1f s, probe %.1f s" % (s["T_over_rs"], s["channels"], s["step_s"], s["probe_s"]))
