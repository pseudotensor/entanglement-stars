#!/usr/bin/env python
"""engine_capture_read_v2.py (builder analyst; engine spec 37.6 (4): "engine_capture_read_v2 (new; record-only) prints (1) to (3) on the
40 r_s records now and on the 300 r_s records when they land"): per C-capture version 9 JSON (--json, repeatable), per background and
probe: (1) the comparator named as the massive critical disk pi b_c^2(omega, m), with C_m = pi b_c^2(omega, m)/(27 pi (g E)^2) printed;
the weights named "inward-classified" (37.6 (1)'s words); the launch channels the engine excluded (taken as A = 0) and the channel cap
printed and labelled uncertified tails; (2) the lattice group speed against the continuum sqrt(1 - 1/e^2), its departure in percent,
"lattice-kinematic" above 5 percent (the lead's threshold, 37.6 (2)); (3) the settle state by the run's own rule: an unsettled probe
prints as such; this version makes no barrier-top fit (no version 9 probe has settled; a settled probe is listed as waiting for one).
Added by the builder for the lead's reading (labelled, a reading aid, not a rule): on each run, the chain-end arrival, the first reading
time at which the plain norm beyond R_max = R_N - 16, summed over the channels as sigma is and divided by pi b_c^2(omega, m), exceeds
--arrival (0.01, the builder's threshold); the chain's outer end has no absorber (the run's absorbing port is the inner one, inside
R_t - 2), so readings at and after the arrival carry weight that left the counted region outward and may return from the end; beside
it, the smallest sigma/(pi b_c^2) before the arrival with its time and in-transit share, and the largest after it. Every number printed
by the script from the JSONs. Args: --json (repeatable) --md [--arrival 0.01]."""
import argparse
import json
import math


def tails(r, cap):
    ex = {}
    for k, why in r.get("excluded_channels") or []:
        ex.setdefault(abs(k), set()).add(1 if k > 0 else -1)
    s = ", ".join(("+-%d" % k) if len(v) == 2 else ("%+d" % (k * next(iter(v)))) for k, v in sorted(ex.items()))
    n = sum(len(v) for v in ex.values())
    kmax = max((c["kappa"] for c in r["channels"]), default=0)
    return ("%s (%d channels)" % (s, n)) if n else "none", ("at the cap %d" % cap) if kmax >= cap else "below the cap %d (largest |kappa| %d)" % (cap, kmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True); ap.add_argument("--md", required=True)
    ap.add_argument("--arrival", type=float, default=0.01)
    a = ap.parse_args()
    L = ["## C-capture version 9 read against the massive critical disk (engine_capture_read_v2.py; engine spec 37.6 (1) to (4))", "",
         "- the comparator is pi b_c^2(omega, m), the massive critical disk (b_c from each run's own label), C_m = pi b_c^2(omega, m)/(27 pi (g E)^2) times the massless disk; sigma/(pi b_c^2) is a summed ratio, (pi/p^2) sum_kappa kappa [f(+kappa) + f(-kappa)] over that constant, not a packet probability.",
         "- \"inward-classified weight\": the positive-band group-velocity sign applied to both spin components; not a spinor-band projection or an integrated flux (37.6 (1)); the runs' columns headed \"inward-branch\" carry this weight.",
         "- uncertified tails: the launch channels the engine excluded (taken as A = 0), and the channels above the cap where a probe's channel list reaches it.",
         "- lattice-kinematic: the probe's lattice group speed below the continuum sqrt(1 - 1/e^2) by more than 5 percent (the lead's threshold, 37.6 (2)).",
         "- the chain-end arrival (the builder's reading aid, threshold %g of pi b_c^2(omega, m)): the first reading time at which the plain norm beyond R_max = R_N - 16, summed as sigma, exceeds it; the chain's outer end has no absorber (the absorbing port is the inner one), so readings from there on carry weight that left the counted region outward; \"none\" means the threshold is not reached within the run." % a.arrival, ""]
    for jp in a.json:
        d = json.load(open(jp))
        sr = d["settle_rule"]; cap = d["args"]["kappa_cap"]
        state = ("complete, %s to %s UTC" % (d["start_utc"], d["end_utc"])) if d.get("end_utc") else ("in progress (started %s UTC): %d probes done" % (d["start_utc"], len(d["results"])))
        L += ["### %s" % jp, "",
              "- slice %s (T %.1f), E %.6f, r_s %.4f, R_t %.4f; readings every %g r_s to %g r_s; the run %s." % (d["slice"], d["T"], d["E"], d["r_s"], d["R_t"], sr["T_step_over_rs"], sr["T_max_over_rs"], state), "",
              "| background | probe | e = omega/m | C_m | group speed, lattice | sqrt(1 - 1/e^2) | departure (percent) | label | channels | excluded at launch (uncertified, A = 0) | the cap | settled (the run's rule) | sigma/(pi b_c^2) at the last reading (T/r_s) | chain-end arrival (T/r_s; the plain norm beyond R_max there) | smallest sigma/(pi b_c^2) before it (T/r_s; in-transit share) | largest after it (T/r_s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in d["results"]:
            mm = r["massive"]; e = mm["e"]; vl = mm["group_speed"]; vc = math.sqrt(1.0 - 1.0 / e ** 2)
            dep = 100.0 * (1.0 - vl / vc); Cm = mm["pi_bc2_massive"] / r["27_pi_gE2"]; nrm = r["norm_ratio"]
            ex, cp = tails(r, cap)
            st = r["settle"]; steps = r["steps"]
            sett = ("settled at %.1f r_s (barrier-top fit waits for a later version)" % st["T_over_rs"]) if st.get("settled") else "unsettled"
            arr = None
            for i, s_ in enumerate(steps):
                b = r["beyond_Rmax_summed"]["%.4f" % s_["T"]] / nrm
                if b > a.arrival:
                    arr = (i, s_["T_over_rs"], b); break
            pre = steps[:arr[0]] if arr else steps
            mn = min(pre, key=lambda s_: s_["ratio"]) if pre else None
            post = steps[arr[0]:] if arr else []
            mx = max(post, key=lambda s_: s_["ratio"]) if post else None
            L.append("| %s | %s | %.4f | %.4f | %.5f | %.5f | %.2f | %s | %d | %s | %s | %s | %.4f (%.1f) | %s | %s | %s |" % (
                r["background"], r["probe"], e, Cm, vl, vc, dep, "lattice-kinematic" if dep > 5.0 else "-", len(r["channels"]), ex, cp, sett,
                steps[-1]["ratio"], steps[-1]["T_over_rs"], ("%.1f; %.4f" % (arr[1], arr[2])) if arr else "none",
                ("%.4f (%.1f; %.4f)" % (mn["ratio"], mn["T_over_rs"], mn["transit_share"])) if mn else "-",
                ("%.4f (%.1f)" % (mx["ratio"], mx["T_over_rs"])) if mx else "-"))
        L.append("")
    L.append("- every number above is printed by the script from %s; no propagation." % ", ".join(a.json))
    open(a.md, "w").write("\n".join(L) + "\n")
    print("wrote %s" % a.md)


if __name__ == "__main__":
    main()
