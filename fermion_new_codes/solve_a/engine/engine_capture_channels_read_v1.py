#!/usr/bin/env python
"""engine_capture_channels_read_v1.py (builder analyst; engine spec 35.2 (4): "A reading script, no run (a new file in code/engine/;
CAP7/capture_v9_min_channels_v1.md): from the version 9 minimum JSON at 40 r_s, per background, probe and channel |kappa|: A, the
in-transit weight and the inside weight, the critical channel p b_c marked; read in turn 44 (whether the lingering weight is near-critical or
spread)"): per background and probe, at the run's last reading time, every channel |kappa| with A(+), A(-), the in-transit inward weight on
[R_t + 2, R_max) and the inward weight inside R_t + 2 at each sign (fractions of the channel's incident weight, as the run stores them),
and its share of the summed in-transit weight (pi/p^2) kappa [W(+) + W(-)]; the critical channel p b_c(omega, m) (the massive capture
impact parameter from the run's own label) marked, and the share of the in-transit weight within 10 percent of it printed. Every number
printed by the script from the JSON. Args: --json --md."""
import argparse
import json
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    d = json.load(open(a.json))
    L = ["## C-capture version 9's channels at the last reading time (engine_capture_channels_read_v1.py; engine spec 35.2 (4); %s)" % a.json, "",
         "- per channel: A(+) and A(-) (the non-return fraction at each sign), the in-transit inward weight on [R_t + 2, R_max) and the inward weight inside R_t + 2 at each sign, fractions of the channel's incident weight; \"share\" is the channel's part of the summed in-transit weight (pi/p^2) kappa [W(+) + W(-)]; the critical channel kappa_c = p b_c(omega, m) is marked \"c\" (the channel nearest it) and the in-transit share of the channels within 10 percent of kappa_c is printed per probe.", ""]
    for r in d["results"]:
        T = r["T_list"][-1]; key = "%.4f" % T
        p = r["p"]
        bcm = (r.get("massive") or {}).get("b_c_massive")
        kc = p * bcm if bcm else None
        chans = r["channels"]
        def g(c, s_, f):
            return (c[s_].get(f) or {}).get(key)
        tr = [(c["kappa"], c["kappa"] * ((g(c, "plus", "inward_transit") or 0.0) + (g(c, "minus", "inward_transit") or 0.0))) for c in chans]
        tot = sum(x for _, x in tr)
        near = sum(x for k, x in tr if kc and abs(k - kc) <= 0.1 * kc)
        kmax = max(tr, key=lambda x: x[1])[0] if tr else None
        cn = min((c["kappa"] for c in chans), key=lambda k: abs(k - kc)) if kc else None
        L += ["### %s | %s (omega %.5f, p %.6f, T %.2f r_s after launch)" % (r["background"], r["probe"], r["omega"], p, T / d["r_s"]), "",
              "- the critical channel p b_c(omega, m) = %s; the channel carrying the largest in-transit share: |kappa| %s; the in-transit share within 10 percent of the critical channel: %.4f; channels %d" % (
                  ("%.2f" % kc) if kc else "-", kmax, (near / tot) if tot > 0 else float("nan"), len(chans)), "",
              "| |kappa| | A(+) | A(-) | in transit (+) | in transit (-) | inside (+) | inside (-) | share of the in-transit sum | mark |", "|---|---|---|---|---|---|---|---|---|"]
        f = lambda x: ("%.4e" % x) if x is not None else "-"  # noqa: E731
        for c, (_, x) in zip(chans, tr):
            L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (c["kappa"], f(g(c, "plus", "A")), f(g(c, "minus", "A")), f(g(c, "plus", "inward_transit")), f(g(c, "minus", "inward_transit")),
                                                                 f(g(c, "plus", "inward_inside")), f(g(c, "minus", "inward_inside")), ("%.4f" % (x / tot)) if tot > 0 else "-", "c" if c["kappa"] == cn else ""))
        L.append("")
    L.append("- every number above is printed by the script from %s; no propagation." % a.json)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("wrote %s" % a.md)


if __name__ == "__main__":
    main()
