#!/usr/bin/env python
"""s5_rows_extract_v1.py (the lane's analyst builder; engine spec 40.5 (4): "S5 (a), (b)'s reading from an extract the builder makes now,
results/engine/CERT3v9/s5_rows_extract_v1.md: each rows_v9_*.md's summary lines verbatim and, per run, over the certified records the range and
median of the share within 1/10 of R_hat = R^ref + R^zone against the direct residuals, the finite-zone R_t^zone/H^2 and R_r^zone/H^2 ranges, the
ledger remainders with -Delta W, and the clipped intervals' count"). The certified records of a run are read as those whose certificate
quadrature (the certificate's own reconstruction, the rows' tt "quadrature" entry) holds a share within 1/10 of at least 0.8 (the certificate's
criterion, which on the production reproduces its 281 of 321; stated in the file, not ruled); the extract also prints the same quantities over
all records. Every number is printed by the script from results/engine/CERT3v9/rows_v9_*.json and .md. Usage: s5_rows_extract_v1.py --dir --out."""
import argparse
import glob
import hashlib
import json
import os

import numpy as np


def rng(x):
    x = [v for v in x if v is not None and v == v]
    if not x:
        return "-"
    return "%.3g to %.3g (median %.3g; %d)" % (min(x), max(x), float(np.median(x)), len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("refusing to overwrite %s" % a.out)
    L = ["# S5 rows extract, version 1 (engine spec 40.5 (4); for the lead's turn 49 and COLLAPSE_certificate_statement_v1.4)", "",
         "Written by code/engine/s5_rows_extract_v1.py from the rows compositions in results/engine/CERT3v9 (engine_replay_v9.py --compose-rows); every number printed by the script. "
         "The certified records of a run are read as those whose certificate quadrature (the rows' tt quadrature entry, the certificate's own reconstruction) has a share within 1/10 of at least 0.8, "
         "the certificate's criterion (on the production it reproduces 281 of 321); the same quantities over all records beside. Nothing is ruled.", ""]
    for p in sorted(glob.glob(os.path.join(a.dir, "rows_v9_*.json"))):
        md = p[:-5] + ".md"
        J = json.load(open(p)); rows = J["rows"]
        name = os.path.basename(p)[:-5]
        L += ["## %s" % name, "", "- files: %s (md5 %s); %s (md5 %s); data %s" % (os.path.basename(p), hashlib.md5(open(p, "rb").read()).hexdigest(), os.path.basename(md),
                                                                                   hashlib.md5(open(md, "rb").read()).hexdigest(), J.get("data")), "",
              "### its summary lines, verbatim", ""]
        txt = open(md).read().split("\n")
        summ = [ln for ln in txt if ln.startswith("- tt, ") or ln.startswith("- rr, ") or ln.startswith("- every number above")]
        L += summ + [""]
        cert = [r for r in rows if (r.get("tt") or {}).get("quadrature", {}).get("share_within_0.1") is not None and r["tt"]["quadrature"]["share_within_0.1"] >= 0.8]
        for lab, sel in (("certified records (%d of %d)" % (len(cert), len(rows)), cert), ("all records (%d)" % len(rows), rows)):
            g = lambda f: [f(r) for r in sel]  # noqa: E731
            sh = lambda c, k: g(lambda r: ((r.get(c) or {}).get(k) or {}).get("share_within_0.1"))  # noqa: E731
            zc = lambda k: g(lambda r: ((((r.get("hooks") or {}).get("zone") or {}).get("centre") or {}).get(k)))  # noqa: E731
            iv = lambda k: g(lambda r: ((((r.get("hooks") or {}).get("ledger") or {}).get("interval") or {}).get(k)))  # noqa: E731
            clipped = sum(1 for r in sel if (((r.get("hooks") or {}).get("clipped") or {}).get("intervals")))
            clip_iv = sum(len(((r.get("hooks") or {}).get("clipped") or {}).get("intervals") or []) for r in sel)
            rem = [abs(x) for x in iv("remainder") if x is not None]
            L += ["### over the %s" % lab, "",
                  "- tt, the share within 1/10: R_hat %s; R^zone %s; R^ref %s; the quadrature %s" % (rng(sh("tt", "hat")), rng(sh("tt", "zone")), rng(sh("tt", "ref")), rng(sh("tt", "quadrature"))),
                  "- rr, the share within 1/10: R_hat %s; R^zone %s; R^ref %s; the quadrature %s" % (rng(sh("rr", "hat")), rng(sh("rr", "zone")), rng(sh("rr", "ref")), rng(sh("rr", "quadrature"))),
                  "- the finite-zone rows at the centre node: R_t^zone/H^2 %s; R_r^zone/H^2 %s" % (rng(zc("Rt_over_H2")), rng(zc("Rr_over_H2"))),
                  "- the ledger per interval: the remainder's absolute value %s (largest %s); -Delta W %s; Delta E_G %s; a + b + c %s" % (
                      rng(rem), ("%.3e" % max(rem)) if rem else "-", rng(iv("minus_dW")), rng(iv("dE_G")), rng(iv("abc"))),
                  "- the clipped intervals: %d records with clipped intervals, %d intervals in all" % (clipped, clip_iv), ""]
    open(a.out, "x").write("\n".join(L) + "\n")
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
