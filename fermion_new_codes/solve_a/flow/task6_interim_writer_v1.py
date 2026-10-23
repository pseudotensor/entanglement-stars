#!/usr/bin/env python
"""Task 6 interim result writer (builder analyst): solve/results/RESULT_task6_interim.md from task6_report_v1.py's tables over the first
channels' records (flow_page_v1.py and flow_page_v2.py), with the header and the builder's notes (labelled; numbers read from the JSONs).
Args: --report-md --jsons (comma list) --stamp --out."""
import argparse
import json


def f(v, d=4):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def main():
    ap = argparse.ArgumentParser()
    for k in ("--report-md", "--jsons", "--stamp", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    chans = []
    for p in a.jsons.split(","):
        d = json.load(open(p))
        for c in d["channels"]:
            c["_script"] = d["script"]
            chans.append(c)
    orth = max(c["checks"]["PhiH_Phi_minus_1_max"] for c in chans)
    eb = max(abs(c["checks"]["energy_balance_rel"]) for c in chans)
    p0 = [c["p0"] for c in chans if "p0" in c]
    pid = max(abs(v["partner"]["identity_error"]) for c in chans for r in c["packets"] for v in r.get("hold", {}).values() if "partner" in v)
    ex = []
    for c in chans:
        k = sorted(c["excess"]["%d" % round(c["rs"])].items(), key=lambda kv: float(kv[0]))
        ex.append("%s j %s kappa %+g: %s" % (c["_script"], f(c["j"]), c["kappa_j"], ", ".join("%s: %s" % (f(float(t)), f(v, 3)) for t, v in k)))
    late = []
    for c in chans:
        if c["mass"] != 0 or c["j"] != 0.5:
            continue
        for r in c["packets"]:
            if "hold" not in r:
                continue
            hs = sorted(r["hold"].items(), key=lambda kv: float(kv[0]))
            lastn = hs[-1][1]["n"]
            late.append("%s kappa %+g omega/kappa %s: n at the last hold (%s/kappa) %s, F_w(1) %s, ratio %s" % (
                c["_script"], c["kappa_j"], f(r["omega_over_kappa"]), f(float(hs[-1][0])), f(lastn, 3), f(r["F_w(1)"], 3), f(lastn / r["F_w(1)"], 3)))
    rm = []
    for c in chans:
        for r in c["packets"]:
            if r.get("raymap"):
                q = r["raymap"]
                rm.append("%s kappa %+g omega/kappa %s: window points %s, kappa_p/kappa from x %s, from k %s" % (
                    c["_script"], c["kappa_j"], f(r["omega_over_kappa"]), q.get("n_points", 0),
                    f((q.get("kappa_p_from_x") or float("nan")) / c["kappa"], 4), f((q.get("kappa_p_from_k") or float("nan")) / c["kappa"], 4)))
    L = ["# Task 6 interim (%s): the formation state on the flow chart, the first channels" % a.stamp, "",
         "Builder: analyst. Spec: solve/specs/SPEC_task6_page_curve.md (lead, turn 11 T56). Code: flow_page_v1.py (the first three channels) and "
         "flow_page_v2.py (version 2: the back-trace range of P3(a) and the holds 64, 80, 96/kappa); records in solve/results/task6/; tables by "
         "task6_report_v1.py; this file by task6_interim_writer_v1.py. Every number: \"prescribed flow (simpler version); PG radial; v declared; "
         "formation history declared (the ramp)\". An interim for the lead's ruling; nothing delivered.", "",
         "## Builder's notes (labelled; for the lead)",
         "- (1) The instrument: the flat sea of h_0 from one dense eigendecomposition; the ramp on the orbital matrix by the commutator-free "
         "fourth-order Magnus scheme with Chebyshev exponentials of the banded h (BUILDER (C1); tested on a 256-site chain: errors 2.3e-6, 1.3e-7, "
         "7.2e-9 at dt = 1, 0.5, 0.25, fourth order; the backward ramp inverts the forward one to 2e-15); the hold exactly in h_f's eigenbasis.",
         "- (2) The identities over these channels: Phi^H Phi - 1 at most %s; the energy balance E(t_r) - E(0) - W at most %s relative to W; "
         "P0: %s; the partner identity ||w_p||^2 - n(1 - n) at most %s." % (
             f(orth, 2), f(eb, 2), "; ".join("S_A %s from G %s, S_{A'} - S_A %s" % (f(x["S_A_direct"], 8), f(x["difference"], 2), f(-x["S_A_minus_S_complement"], 2)) for x in p0),
             f(pid, 2)),
         "- (3) The excess cut entropy at R_c = r_s per hold time (t kappa: excess): " + "; ".join(ex) + ". There is no rising branch at these cuts: "
         "the formation transient (the ramp of t_r = 2 r_s) raises the excess to its maximum within 6/kappa, and it falls monotonically after; "
         "builder's reading: on this chart the declared regular centre returns what crosses (task 5 E3), so the interior's correlations with the "
         "outside do not accumulate.",
         "- (4) The occupations of the outgoing packets (j = 1/2, massless) at the last hold: " + "; ".join(late) + ". At omega/kappa >= 2 the "
         "occupations are the ramp's transient (falling through the holds); at omega/kappa <= 1 they reach plateaus well below F_w(1) (builder's "
         "reading: a j = 1/2 greybody suppression of a thermal occupation, to be divided out with the channel's transmission; task 5's E2 does not "
         "supply it at these frequencies).",
         "- (5) The partners of the outgoing packets lie mostly outside r_s at negative Killing frequency (the partner lines of the tables).",
         "- (6) The ray map of P3(a): " + ("; ".join(rm) if rm else "no record") + ". Version 1's back-trace stopped at 8/kappa while the builder's "
         "packets (C3) start at x_0 up to 27 r_s; version 2 runs to 8/kappa + 1.5 (x_0 - 4 r_s).", ""]
    L += ["#" + x if x.startswith("#") else x for x in open(a.report_md).read().split("\n")]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (a.out, len(L)))


if __name__ == "__main__":
    main()
