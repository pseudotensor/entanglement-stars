#!/usr/bin/env python
"""engine_v25_gate_v1.py (builder analyst; engine spec 25.1: "Gate: v25 (= v24 on ops v3) reproduces the three production records and the
first three tau 4 records bitwise in the masks and to 1e-11 in E"): the first --records records of an engine_collapse_v25 run from T = 0
against a stored run of the same member (engine_collapse_v19) at the same T: per record |dE/E| (criterion below 1e-11), the occupation
masks bit for bit per channel against the stored run's packed masks (criterion: all equal), the applied fields' largest differences and
the continuation modes beside (informational). Exit 0 when both criteria hold on every record. Args: --new dir/tag --ref dir/tag --records
--md."""
import argparse
import json
import os
import sys

import numpy as np


def load(run):
    d, tag = os.path.split(run)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % tag), allow_pickle=True)
    return tag, J, Z, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    tn, Jn, Zn, Sn = load(a.new); tr, Jr, Zr, Sr = load(a.ref)
    kap = [int(k.split("_")[1]) for k in Sn.files if k.startswith("mask_")]
    recn = {round(r["T"], 6): r for r in Jn["records"] if r.get("E") is not None}
    recr = {round(r["T"], 6): r for r in Jr["records"] if r.get("E") is not None}
    TSn = np.asarray(Sn["T"], float); TSr = np.asarray(Sr["T"], float)
    TZn = np.asarray(Zn["T"], float); TZr = np.asarray(Zr["T"], float)
    rs = float(Jn["args"]["rs"])
    L = ["## engine_collapse_v25's gate (engine_v25_gate_v1.py): %s against %s" % (tn, tr), "",
         "| T/rs | E (new) | E (stored) | dE/E | masks equal (channels) | max dN applied | max dv applied | continuation (new; stored) |", "|---|---|---|---|---|---|---|---|"]
    ok = True; n = 0
    for i, T in enumerate(TSn[:a.records]):
        j = int(np.argmin(np.abs(TSr - T)))
        if abs(TSr[j] - T) > 1e-9:
            L.append("| %.4f | the stored run has no record here |" % (T / rs)); ok = False; continue
        rn = recn.get(round(T, 6), {}); rr = recr.get(round(T, 6), {})
        En = float(rn.get("E", np.nan)); Er = float(rr.get("E", np.nan))
        dE = abs(En - Er) / abs(Er)
        eq = 0
        for k in kap:
            mn = np.unpackbits(np.asarray(Sn["mask_%d" % k][i], np.uint8)); mr = np.unpackbits(np.asarray(Sr["mask_%d" % k][j], np.uint8))
            eq += int(np.array_equal(mn, mr))
        kn = int(np.argmin(np.abs(TZn - T))); kr = int(np.argmin(np.abs(TZr - T)))
        dN = float(np.max(np.abs(np.asarray(Zn["N_applied"][kn], float) - np.asarray(Zr["N_applied"][kr], float))))
        dv = float(np.max(np.abs(np.asarray(Zn["v_applied"][kn], float) - np.asarray(Zr["v_applied"][kr], float))))
        good = (dE < 1e-11) and (eq == len(kap))
        ok = ok and good; n += 1
        L.append("| %.4f | %.12f | %.12f | %.2e | %d of %d | %.2e | %.2e | %s; %s |" % (T / rs, En, Er, dE, eq, len(kap), dN, dv,
                                                                              (rn.get("continuation_label") or "-")[:40], (rr.get("continuation_label") or "-")[:40]))
    L += ["", "- %s: %d records compared; the criteria |dE/E| < 1e-11 and every channel's mask equal bit for bit %s." % ("PASS" if (ok and n == a.records) else "FAIL", n,
                                                                                                                         "hold on every record" if (ok and n == a.records) else "do not all hold"),
          "- every number above is read from the two runs' files by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if (ok and n == a.records) else 1)


if __name__ == "__main__":
    main()
