#!/usr/bin/env python
"""Task 3 part (b), the conversion temperature with H_*'s massive light packet (SPEC_task3_prescribed_flow.md sec. 5; builder analyst): job
files for the record's conversion package and its report.

The package: review_output/layer_response_2026-10-02/spectro_work (lattice.py, campaign.py, tscan_report.py; unchanged copies in
<internal>/record_spectro), the prescribed-flow template the goal names: the planar chain with the sine hop, the
range-two flow hop -(i/4)(v_j + v_(j+2)) 1 (H_*'s operator per channel), the regular-centre tanh jet v = -v_max tanh(x/l_c) cut off at
the radius R = 344/kappa over the width 1.2/kappa (|dv/dx| = kappa at |v| = 1), the ramp sin^2 over 4/kappa, the fourth-order split
propagation at dt = 0.1, the initial normal-sea projector by the DST, outgoing packets (spatial Gaussians of width 12/kappa projected on the
positive normal band at the port 352/kappa, beyond which the lead is frozen), the occupations by backward propagation, the packet-measure
Fermi function; the record's time scan (kappa t = 90 to 250, the plateau over 140 to 200) and its box (n = 2048/kappa).
B0: the record's two time-scan jobs unchanged (kappa a = 1/8 and 1/32, the massless axial channel): the reproduction control (C5).
B1: the same jobs with the light packet's mass m = (m/kappa) kappa, m/kappa in {1/4, 1/2, 1}, and the ratios omega/kappa in {1/4, 1/2, 1, 2}
(the package reports a ratio below the port's threshold omega_th = m as kinematically closed); the label "prescribed flow (simpler version);
planar jet; v declared" in every job file.
Modes: make --record-jobs k8.json,k32.json --out DIR (writes DIR/k8 and DIR/k32 with named job files, no hashing); report DIR... --json OUT
(the plateau, its largest deviation, F_w(1), n/F_w(1), theta_r per packet with tscan_report.analyse, the closed ratios, the package's own fit,
the clean-window validity count, the massless control against the record's values)."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

RECORD = {0.125: {0.25: 0.99443, 0.5: 0.99364, 1.0: 0.99042}, 0.03125: {0.25: 0.99965, 0.5: 0.99960, 1.0: 0.99939}}
LABEL = "prescribed flow (simpler version); planar jet; v declared"


def make(a):
    out = Path(a.out)
    for src in a.record_jobs.split(","):
        cfg = json.loads(Path(src).read_text())
        kap = cfg["kappa"]
        tag = "k%d" % round(1 / kap)
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)
        jobs = [("B0_m0", dict(cfg))]
        for mk in (0.25, 0.5, 1.0):
            c = dict(cfg)
            c["mass"] = mk * kap
            c["ratios"] = [0.25, 0.5, 1.0, 2.0]
            jobs.append(("B1_m%gkappa" % mk, c))
        for name, c in jobs:
            c["variant"] = name
            c["label"] = LABEL
            c["job_id"] = "%s_%s" % (tag, name)
            (d / (c["job_id"] + ".job.json")).write_text(json.dumps(c, indent=2))
            print(c["job_id"], "mass", c["mass"], "ratios", c["ratios"], "n", c["n"])


def report(a):
    sys.path.insert(0, a.package)
    import numpy as np
    import tscan_report as TR
    rows = []
    for d in a.dirs:
        for f in sorted(Path(d).glob("*.result.json")):
            r = json.loads(f.read_text())
            c = r["config"]
            kap = c["kappa"]
            if r.get("arrays"):
                r["arrays"] = str(Path(d) / Path(r["arrays"]).name)
            times, res = TR.analyse(r, 140.0, 200.0) if r["packets"] else (None, [])
            for x in res:
                fermi = 1.0 / (math.exp(2 * math.pi * x["ratio"]) + 1.0)
                rec = RECORD.get(kap, {}).get(x["ratio"]) if c["mass"] == 0 else None
                rows.append({"job": c["job_id"], "kappa_a": kap, "m_over_kappa": c["mass"] / kap, "ratio": x["ratio"], "branch": x["branch"],
                             "plateau": x["plateau"], "plateau_maxdev": x["plateau_maxdev"], "F_w1": x["F_w1"], "n_over_Fw1": x["n_over_Fw1"],
                             "theta_r": x["theta_r"], "one_minus_theta_over_ka2": (1 - x["theta_r"]) / kap ** 2 if x["theta_r"] == x["theta_r"] else None,
                             "omega_over_kappa": x["omega_over_kappa"], "spread_over_kappa": x["spread_over_kappa"], "valid_times": int(sum(x["valid"])),
                             "Fermi_at_r": fermi, "record_theta": rec, "theta_minus_record": (x["theta_r"] - rec) if rec else None,
                             "seconds": r["elapsed_seconds"]})
            for cl in r.get("closed", []):
                rows.append({"job": c["job_id"], "kappa_a": kap, "m_over_kappa": c["mass"] / kap, "ratio": cl["target_ratio"], "closed": cl["status"]})
            if r.get("fit"):
                rows.append({"job": c["job_id"], "kappa_a": kap, "m_over_kappa": c["mass"] / kap, "package_fit": r["fit"]})
    Path(a.json).write_text(json.dumps({"label": LABEL, "rows": rows}, indent=1, default=float))
    L = ["# Task 3 part (b): the conversion temperature (flow_B_conversion_v1.py report; the record's package, %s)" % LABEL, "",
         "| job | kappa a | m/kappa | omega/kappa (target) | branch | plateau n | max dev | F_w(1) | n/F_w(1) | theta_r | (1 - theta_r)/(kappa a)^2 | record theta_r | difference | valid times | spread/kappa |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, d=6: "-" if v is None else ("%.*g" % (d, v))
    for r in rows:
        if "closed" in r:
            L.append("| %s | %s | %s | %s | - | closed (%s) | | | | | | | | | |" % (r["job"], fmt(r["kappa_a"]), fmt(r["m_over_kappa"]), fmt(r["ratio"]), r["closed"]))
        elif "plateau" in r:
            L.append("| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %d | %s |" % (
                r["job"], fmt(r["kappa_a"]), fmt(r["m_over_kappa"]), fmt(r["ratio"]), r["branch"], fmt(r["plateau"]), fmt(r["plateau_maxdev"], 2),
                fmt(r["F_w1"]), fmt(r["n_over_Fw1"], 5), fmt(r["theta_r"], 6), fmt(r["one_minus_theta_over_ka2"], 4), fmt(r["record_theta"]),
                fmt(r["theta_minus_record"], 2), r["valid_times"], fmt(r["spread_over_kappa"], 3)))
    for r in rows:
        if "package_fit" in r:
            L.append("- %s: the package's fit over its clean times: theta %s +- %s (chi2 %s / %s)." % (
                r["job"], fmt(r["package_fit"]["theta"]), fmt(r["package_fit"]["temporal_uncertainty"], 3), fmt(r["package_fit"]["chi2"], 3),
                r["package_fit"]["degrees_of_freedom"]))
    Path(a.md).write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("make")
    m.add_argument("--record-jobs", required=True)
    m.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--package", required=True)
    r.add_argument("--json", required=True)
    r.add_argument("--md", required=True)
    a = ap.parse_args()
    {"make": make, "report": report}[a.mode](a)


if __name__ == "__main__":
    main()
