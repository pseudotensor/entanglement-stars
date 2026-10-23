#!/usr/bin/env python
"""capture_cost_v11_gate_c_v1.py (builder analyst; the lane's ruling on engine spec 38.5 (3), gate (c): "its cost per step against v10's,
so the 300 r_s run's wall time is stated before launch"): per background and probe, from the 40 r_s runs' JSONs of versions 9, 10 and 11
(the same slice, lane and workers), the probe's length and its mean seconds per 2 r_s reading step, version 11's against version 10's and
version 9's; and the 300 r_s run's wall time stated two ways: (i) version 11's mean step per probe times the 300 r_s run's readings (150
per probe at one reading per 2 r_s), summed; (ii) version 9's measured 300 r_s run, probe by probe, scaled by version 11's 40 r_s length over
version 9's. Every number printed from the JSONs. Args: --v9 --v10 --v11 --v9-t300."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    for k in ("v9", "v10", "v11", "v9_t300"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    a = ap.parse_args()
    D = {k: json.load(open(getattr(a, k))) for k in ("v9", "v10", "v11", "v9_t300")}
    print("runs: version 9 %s (%.0f s), version 10 %s (%.0f s), version 11 %s (%.0f s); version 9 to 300 r_s %s (%.0f s)" % (
        D["v9"]["start_utc"], D["v9"]["run_length_s"], D["v10"]["start_utc"], D["v10"]["run_length_s"], D["v11"]["start_utc"], D["v11"]["run_length_s"],
        D["v9_t300"]["start_utc"], D["v9_t300"]["run_length_s"]))
    print("| background | probe | channels | length v9, v10, v11 (s) | mean step v9, v10, v11 (s per 2 r_s) | v11/v10 | v11/v9 | the 300 r_s run: (i) v11 step x 150 (s) | (ii) v9's 300 r_s length x v11/v9 (s) |")
    t1 = t2 = 0.0
    for r9, r10, r11, r3 in zip(D["v9"]["results"], D["v10"]["results"], D["v11"]["results"], D["v9_t300"]["results"]):
        assert (r9["probe"], r9["background"]) == (r11["probe"], r11["background"]) == (r10["probe"], r10["background"]) == (r3["probe"], r3["background"])
        ms = [sum(s["step_s"] for s in r["steps"]) / len(r["steps"]) for r in (r9, r10, r11)]
        n300 = len(r3["steps"])
        e1 = ms[2] * n300; e2 = r3["length_s"] * r11["length_s"] / r9["length_s"]
        t1 += e1; t2 += e2
        print("| %s | %s | %d | %.0f, %.0f, %.0f | %.1f, %.1f, %.1f | %.3f | %.3f | %.0f | %.0f |" % (
            r11["background"][:12], r11["probe"], len(r11["channels"]), r9["length_s"], r10["length_s"], r11["length_s"], ms[0], ms[1], ms[2],
            r11["length_s"] / r10["length_s"], r11["length_s"] / r9["length_s"], e1, e2))
    print("whole runs: version 11 over version 10 %.3f, over version 9 %.3f; the 300 r_s run's wall time if no probe settles: (i) %.0f s (%.2f h), (ii) %.0f s (%.2f h); version 9's 300 r_s run took %.0f s (%.2f h)" % (
        D["v11"]["run_length_s"] / D["v10"]["run_length_s"], D["v11"]["run_length_s"] / D["v9"]["run_length_s"], t1, t1 / 3600.0, t2, t2 / 3600.0,
        D["v9_t300"]["run_length_s"], D["v9_t300"]["run_length_s"] / 3600.0))


if __name__ == "__main__":
    main()
