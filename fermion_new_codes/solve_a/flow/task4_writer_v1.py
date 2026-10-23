#!/usr/bin/env python
"""Task 4 result writer, revision 1 (builder analyst): solve/results/RESULT_task4_injection.md from task4_report_v1.py's tables and the
records of task4_static_v4.py, with the header and the builder's notes (labelled; numbers read from the JSONs).  Args: --report-md
--report-json --records (folder) --stamp --out."""
import argparse
import glob
import json
import os


def f(v, d=4):
    if v is None:
        return "-"
    v = float(v)
    return "%.*g" % (d, v) if (v == 0 or 1e-3 <= abs(v) < 1e6) else "%.3e" % v


def main():
    ap = argparse.ArgumentParser()
    for k in ("--report-md", "--report-json", "--records", "--stamp", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    rep = json.load(open(a.report_json))
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.records, "*.json")))]
    m0 = recs[0]["meta"]
    fits = {(c["Ns"], c["j"], c["sign"], c["mass"]): c for c in rep["configs"]}
    s001 = [(k, c["fits"]["delay"]["slope"] if c["fits"]["delay"] else None) for k, c in fits.items() if k[0] == 0.01 and k[3] == 0]
    stag001 = [c["fits"]["wkb_staggered"]["slope"] for k, c in fits.items() if k[0] == 0.01 and c["fits"]["wkb_staggered"]]
    open01 = []
    for r in recs:
        if r["meta"]["Ns"] == 0.1:
            for q in r["packets"]:
                open01.append((r["meta"]["j"], r["meta"]["sign"], q["omega_rh"], q["inside_shell_max"]))
    below = [(r["meta"]["mass"], q["omega_rh"], q["omega"], q["reflected"]) for r in recs if r["meta"].get("mass", 0) > 0 for q in r["packets"]
             if q["omega"] <= r["meta"]["mass"]]
    eps = sorted({f(r["meta"].get("eps_max"), 4) for r in recs})
    L = ["# Task 4 result, revision 1 (%s): packet injection on the static chart (the locked background L frozen at the shell lapse N_s; the "
         "lattice at rest)" % a.stamp, "",
         "Builder: analyst. Spec: solve/specs/SPEC_task4_injection.md sec. 1, read with the lead's comparator of SPEC_task3_prescribed_flow.md sec. "
         "11.6 (the band-top ruling). Code: solve/code/flow/task4_static_v4.py (versions 1 to 3 superseded; records in solve/results/task4/v4/), "
         "the tables by task4_report_v1.py, this file by task4_writer_v1.py. The page code's polar law on this lane's locked background L (E_0 = %s, "
         "A_0 = %s, r_h = 4 A_0 = %s, h = %s); the flat chart (N_s = 1) is the zero of the scale. Static chart, diagnostics; nothing delivered." % (
             f(m0["E0"]), f(m0["A0"]), f(m0["r_h"]), f(m0["h"])), "",
         "## Builder's notes (labelled; for the lead)",
         "- (1) The band top: the page code's radial operator (the staggered int-half difference) has its flat spectrum's maximum at %s per channel "
         "(the measured eps_max of the records), the sine law's 1/h does not apply; version 4 reports the sine law's turning lapse N = h omega psi^2 "
         "and the staggered law's N = h omega psi^2/2 with the WKB delay of its group velocity; the lead's sec. 11.6 adopts the staggered law." % ", ".join(eps),
         "- (2) The delay against log(1/omega) at N_s = 0.01 (four packets per channel): the slopes %s against the test 16 A_0 = %s and the staggered "
         "WKB integral's slopes %s." % ("; ".join("j %s kappa %+d: %s" % (f(k[1]), k[2], f(v, 5)) for k, v in s001), f(16 * m0["A0"], 5),
                                         ", ".join(f(x, 5) for x in sorted(set(round(s, 6) for s in stag001)))),
         "- (3) At N_s = 0.1 the packets enter the shell's interior (the largest norm inside the shell per packet: %s); their delays include the "
         "interior round trip and are not a near-surface delay; the open channels follow the lead's sec. 11.6 (c_s = 0.03025 at N_s = 0.1)." % (
             "; ".join("j %s kappa %+d omega r_h %s: %s" % (f(j), s, f(o), f(x, 3)) for j, s, o, x in open01)),
         "- (4) The massive run (m = 0.05, N_s = 0.01, j = 1/2, kappa -1): the packets with omega <= m are below the threshold (%s); their reflected "
         "fraction and delay measure no propagating packet." % ("; ".join("omega r_h %s (omega %s against m %s): reflected %s" % (f(o), f(om), f(m), f(r, 3))
                                                                         for m, o, om, r in below) or "none"),
         "- (5) The spec's release test (iv) (an outgoing band-top packet placed at the shell, its outward progress over 40 r_h against N psi^-2) is "
         "described in task4_static_v4.py's docstring but not implemented in version 4; it is open (the flowing chart's analogue is task 3's D4).",
         "- (6) The reflected fraction is the largest norm beyond r_ref = 3 r_h after its minimum (the first return, before any echo from the outer "
         "wall), with the end value beside it; the crossings are timed by the norm beyond r_ref (t_in at half its initial value, t_out at half its "
         "final value). Records carry the script name, never a hash.", ""]
    L += ["#" + x if x.startswith("#") else x for x in open(a.report_md).read().split("\n")]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (a.out, len(L)))


if __name__ == "__main__":
    main()
