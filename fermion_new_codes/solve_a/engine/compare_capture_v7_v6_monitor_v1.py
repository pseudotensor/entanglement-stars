"""compare_capture_v7_v6_monitor_v1.py (builder analyst): v7's incident weights and conversion monitor against version 6's JSON on the
same slice and probe, at the monitor times where both used the same 1 r_s Chebyshev steps (t below version 6's first reading time)."""
import json
import sys

v6 = json.load(open(sys.argv[1])); v7 = json.load(open(sys.argv[2]))
print("slices", v6["slice"], v7["slice"], "E equal:", v6["E"] == v7["E"], "R_t equal:", v6["R_t"] == v7["R_t"])
for r7 in v7["results"]:
    r6 = [r for r in v6["results"] if r["background"] == r7["background"] and r["probe"] == r7["probe"]][0]
    t6first = min(r6["T_list"])
    nI = nmon = nbit = 0
    maxrel = 0.0
    for c7 in r7["channels"]:
        c6 = [c for c in r6["channels"] if c["kappa"] == c7["kappa"]][0]
        for s in ("plus", "minus"):
            a6, a7 = c6[s], c7[s]
            if "error" in a6 or "error" in a7:
                print("kappa", c7["kappa"], s, "error v6:", "error" in a6, "v7:", "error" in a7); continue
            nI += 1
            print("kappa %+d: I v6 %.17g v7 %.17g equal %s" % (a7["kappa"], a6["I"], a7["I"], a6["I"] == a7["I"]))
            m6 = {round(m["t"], 6): m for m in a6["monitor"] if m["t"] < t6first - 1e-9}
            for m in a7["monitor"]:
                t = round(m["t"], 6)
                if t in m6:
                    for f in ("inside_outward", "inside_total"):
                        nmon += 1
                        x6, x7 = m6[t][f], m[f]
                        nbit += (x6 == x7)
                        if x6 != x7:
                            maxrel = max(maxrel, abs(x7 - x6) / max(abs(x6), 1e-300))
    print("%s | %s: incident weights compared %d; monitor values compared %d (t < %.4f), bitwise equal %d, max relative difference of the rest %.3e"
          % (r7["background"][:12], r7["probe"], nI, nmon, t6first, nbit, maxrel))
