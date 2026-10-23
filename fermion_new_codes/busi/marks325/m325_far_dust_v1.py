#!/usr/bin/env python
"""m325_far_dust_v1.py (busi f52b1152, 2026-10-20; inbox #325): the far dust's bare energy on the continued completions, in the form of the solve
lane's W3 producer (<internal>/w3_readout_v1.py, md5 bf493745fcfceed0511f1fc195cc27bd, lines 55 to 61):
per fields row the bare dust energy eps_tot - eps_pairs summed over the sites beyond 16 and 24 r_s (R > c r_s), and the pairs' eps_pairs beyond
the same cuts, with E/E0, the outer root R_t and min(N - |v|) of the json record at the same clock. Reads stored files only (box CPU, seconds).
Each --hist is NAME=DIR/TAG (the run's collapse_<TAG>.json and collapse_<TAG>_fields.npz). Prints the matched-clock table at the integer clocks
from --from to --to and every history's value with the spread max - min over the histories holding the clock; writes OUT.json and OUT.npz.
--check-w3 reruns the producer's numbers at 40 on the original runs and compares them with W3_completions_v1's printed values (known answer).
The count beyond the cuts (dust particles, its Killing split) needs the slices at 48 to 56 and is read by busi/code/dust/dust_tail_v3.py (the
record's own readout of 48 to 56); see busi/notes/marks325_stage_v1.md.
Usage: python m325_far_dust_v1.py --hist own=DIR/TAG --hist fs=DIR/TAG --hist pa=DIR/TAG --out PREFIX [--from 40 --to 56]
       python m325_far_dust_v1.py --check-w3"""
import argparse
import json
import os

import numpy as np

CUTS = (16.0, 24.0)
W3 = "<internal>/runs"
W3OWN = "<internal>/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40"
W3_AT40 = {"own": -3.72e-4, "frozen-sources": 2.36e-3, "paired-avg": 2.51e-3}       # W3_completions_v1 section 2, dust>16 at 40
W3_AT40_24 = {"own": 1.05e-4, "frozen-sources": 1.05e-4, "paired-avg": 9.20e-5}


def history(base):
    d, tag = os.path.split(base)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    rs = float(J["args"]["rs"])
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    R = np.asarray(Z["R"], float); Tf = np.asarray(Z["T"], float) / rs
    eps_d = np.asarray(Z["eps_tot"], float) - np.asarray(Z["eps_pairs"], float); eps_p = np.asarray(Z["eps_pairs"], float)
    E0 = float(J["records"][0]["E"])
    rec = {round(float(r["T"]) / rs, 6): r for r in J["records"] if r.get("T") is not None}
    rows = {}
    for i, t in enumerate(Tf):
        r = rec.get(round(float(t), 6), {})
        row = {"T": float(t), "EE0": (float(r["E"]) / E0 if r.get("E") is not None else None), "R_t": r.get("R_t"),
               "minNv": r.get("min_N_minus_absv"), "unconverged": bool(r.get("unconverged", False))}
        for c in CUTS:
            m = R > c * rs
            row["dust_%g" % c] = float(eps_d[i, m].sum()); row["pairs_%g" % c] = float(eps_p[i, m].sum())
        rows[round(float(t), 6)] = row
    return {"records": len(J["records"]), "last": float(J["records"][-1]["T"]) / rs, "E0": E0, "rows": rows, "script": J.get("script")}


def check_w3():
    tg = "w3_%s_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g"
    H = {"own": history(W3OWN)}
    for nm in ("frozen-sources", "paired-avg"):
        H[nm] = history(os.path.join(W3, "w3_" + nm, tg % nm))
    fails = 0
    for nm, h in H.items():
        r = h["rows"][40.0]
        for key, ref in (("dust_16", W3_AT40[nm]), ("dust_24", W3_AT40_24[nm])):
            ok = abs(r[key] - ref) <= 0.006 * abs(ref)          # the printed 3 digits
            fails += (not ok)
            print("%s: %s at 40 %.4e against W3's printed %.2e: %s" % ("PASS" if ok else "FAIL", nm, r[key], ref, key))
    print("CHECK-W3: %d failed" % fails)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--from", dest="t0", type=float, default=40.0)
    ap.add_argument("--to", dest="t1", type=float, default=56.0)
    ap.add_argument("--check-w3", action="store_true")
    a = ap.parse_args()
    if a.check_w3:
        raise SystemExit(1 if check_w3() else 0)
    H = {}
    for h in a.hist:
        nm, base = h.split("=", 1)
        H[nm] = history(base)
        print("%s: %d records, last T/rs %.4f, script %s" % (nm, H[nm]["records"], H[nm]["last"], H[nm]["script"]))
    clocks = [float(t) for t in np.arange(a.t0, a.t1 + 1e-9, 1.0)]
    table = []
    print("| T/r_s | " + " | ".join("%s: E/E0; dust>16; dust>24; pairs>16; R_t; min(N-|v|)" % nm for nm in H) + " | spread dust>16 (n) | spread dust>24 (n) |")
    for t in clocks:
        per = {nm: h["rows"].get(round(t, 6)) for nm, h in H.items()}
        sp = {}
        for k in ("dust_16", "dust_24"):
            v = [p[k] for p in per.values() if p is not None and not p["unconverged"]]
            sp[k] = ((max(v) - min(v)) if len(v) >= 2 else None, len(v))
        table.append({"T": t, "per": per, "spread": sp})
        cells = []
        for nm in H:
            p = per[nm]
            cells.append("-" if p is None else "%s; %.3e; %.3e; %.3e; %s; %s" % (
                ("%.5f" % p["EE0"]) if p["EE0"] is not None else "-", p["dust_16"], p["dust_24"], p["pairs_16"],
                ("%.3f" % p["R_t"]) if p["R_t"] is not None else "none", ("%.4f" % p["minNv"]) if p["minNv"] is not None else "-"))
        print("| %.3f | %s | %s | %s |" % (t, " | ".join(cells), *[("%.2e (%d)" % s) if s[0] is not None else "- (%d)" % s[1] for s in (sp["dust_16"], sp["dust_24"])]))
    if a.out:
        json.dump({"histories": {nm: {k: v for k, v in h.items() if k != "rows"} for nm, h in H.items()}, "table": table, "cuts_over_rs": CUTS},
                  open(a.out + ".json", "w"), indent=1)
        arr = {}
        for nm, h in H.items():
            ts = sorted(h["rows"])
            arr[nm + "__T"] = np.array(ts)
            for k in ("dust_16", "dust_24", "pairs_16", "pairs_24"):
                arr["%s__%s" % (nm, k)] = np.array([h["rows"][t][k] for t in ts])
        np.savez_compressed(a.out + ".npz", **arr)
        print("wrote %s.json and %s.npz" % (a.out, a.out))


if __name__ == "__main__":
    main()
