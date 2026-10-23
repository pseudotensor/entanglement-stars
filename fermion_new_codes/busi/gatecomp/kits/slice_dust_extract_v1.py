#!/usr/bin/env python
"""slice_dust_extract_v1.py (busi lane, GATECOMP kit (ii), inbox #400): the dust comparator's inputs from an engine slice, small.
The dust comparator (solve's dust_comparator_v1d.py, its local npz reader) reads from a slice only T, rs, R, kappas, weights, N_applied,
v_applied, dtype and per channel dust_K and f_K. A T = 0 slice at 4096 sites and K = 14 is about 15 GB (sea_K, R2_K, e_K, masks); the
extract keeps exactly the keys above (bitwise copies), so the 15 GB slice need not cross to the box. Every kept array is compared with
the source after writing (np.array_equal); the source's md5 and the extract's md5 are printed.

  slice_dust_extract_v1.py SLICE.npz OUT.npz [--no-md5]
"""
import argparse
import hashlib
import os
import sys

import numpy as np

KEEP = ("T", "rs", "R", "kappas", "weights", "N_applied", "v_applied", "dtype")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slice"); ap.add_argument("out"); ap.add_argument("--no-md5", dest="no_md5", action="store_true")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit("refused: %s exists" % a.out)
    Z = np.load(a.slice, allow_pickle=False)
    kap = [int(k) for k in np.atleast_1d(Z["kappas"])]
    keys = [k for k in KEEP if k in Z.files] + ["dust_%d" % k for k in kap if "dust_%d" % k in Z.files] + ["f_%d" % k for k in kap if "f_%d" % k in Z.files]
    arrs = {k: Z[k] for k in keys}
    np.savez(a.out, **arrs)
    Y = np.load(a.out, allow_pickle=False)
    bad = [k for k in keys if not np.array_equal(Y[k], arrs[k])]
    if bad:
        sys.exit("MISMATCH after write: %s" % bad)
    ncol = {k: int(arrs["dust_%d" % k].shape[1]) for k in kap if "dust_%d" % k in arrs}
    print("extract %s: T %.4f rs %g, %d sites, kappas %s, dust columns %s, %d keys, %.1f MB, all bitwise equal to the source" % (
        a.out, float(arrs["T"]), float(arrs["rs"]), len(arrs["R"]), kap, ncol, len(keys), os.path.getsize(a.out) / 1e6))
    if not a.no_md5:
        print("md5 source %s  %s" % (md5(a.slice), a.slice))
        print("md5 extract %s  %s" % (md5(a.out), a.out))


if __name__ == "__main__":
    main()
