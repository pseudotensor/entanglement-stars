#!/usr/bin/env python
"""Strand 2b control 3 (SPEC_strand2b_row.md sec. 11.3; builder analyst): the native scalar source e_x^nat(s) (the frozen
code's Coefficients.scalar_source, wick subtraction against the frozen s = 1 sea, at U = 2 g_M of the row; the native action
has no bond quartic) run through the identical 2b machinery (strand2b_row_R4_v8.solve_case with library v6) in place of
rho^P_x[psi], on R4 with the same K and k.  The native law has no excitation grading: its "block" is the whole charge sector
q (N = 4 + q), and the state is that sector's lowest multiplet (the native code's convention).  C13 does not apply.
One case per call: --q, --m, --U, --k.  Writes <out>/native_<tag>.json.  Also --variance: the per-site ruler variance
<S_x^2> - <S_x>^2 in the lowest state, from strand 0's saved operator solutions (full_v3/mats/native_*.npz)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2b_row_R4_v8 as RR  # noqa: E402
import native_core as nc  # noqa: E402
import native_sources as ns  # noqa: E402


class NativeRhoP:
    """build(s) -> {"rho": (n, D, D) array of e_x^nat(s) on sector Nc, "nexc": kexc * identity (one block = the sector)}."""

    def __init__(self, m, U, Nc, kexc, fops):
        self.lat_nat = nc.lattice("ring4", (0.0, 0.0, 0.0))
        self.n = 4
        self.m, self.U, self.Nc, self.kexc, self.f = m, U, Nc, kexc, fops
        G, nsea = ns.sea_contractions(self.lat_nat, m, 8)
        self.co = ns.Coefficients(self.lat_nat, fops.fock, Nc, m, U, G, "wick")
        self.cache = {}

    def build(self, s):
        key = tuple(np.round(np.asarray(s, float), 15))
        if key in self.cache:
            return self.cache[key]
        s = np.asarray(s, float)
        rho = np.array([self.co.scalar_source(x, s) for x in range(self.n)])
        rho = 0.5 * (rho + rho.conj().transpose(0, 2, 1))
        d = rho.shape[1]
        out = {"rho": rho, "nexc": self.kexc * np.eye(d)}
        if len(self.cache) > 4000:
            self.cache.clear()
        self.cache[key] = out
        return out


def variance(mats):
    rows = []
    for f in sorted(glob.glob(os.path.join(mats, "native_*.npz"))):
        z = np.load(f)
        S, rho = z["S"], z["rho"]
        v = [float(np.real(np.trace(rho @ Sx @ Sx)) - np.real(np.trace(rho @ Sx)) ** 2) for Sx in S]
        mean = [float(np.real(np.trace(rho @ Sx))) for Sx in S]
        rows.append({"file": os.path.basename(f), "mean_S": mean, "variance_S": v})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--q", type=int, default=0)
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--U", type=float, default=0.0)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--no-enclosure", action="store_true")
    ap.add_argument("--no-unique", action="store_true")
    ap.add_argument("--variance", default="", help="strand 0 mats dir: compute the ruler variances and exit")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.variance:
        with open(os.path.join(a.out, "native_ruler_variance.json"), "w") as fh:
            json.dump(variance(a.variance), fh, indent=1)
        print("variance done")
        return
    lat = S2.lattice("R4")
    K, _ = nc.K_matrix(nc.lattice("ring4")["positions"])
    fops = RR.B.FockOps(8)
    Nc = 4 + a.q

    RR.B.RhoP = lambda lat_, m, gM, gB, Lam, Nc_, fops_: NativeRhoP(a.m, a.U, Nc_, abs(a.q), fops_)
    RR.slater_route = lambda *args, **kw: {"skipped": "native control: no Slater route"}
    blk = (a.q, 0)
    t0 = time.time()
    rec = RR.solve_case(lat, K, blk, a.m, 2 * 0.0, 0.0, np.inf, a.k, fops, not a.no_enclosure, not a.no_unique)
    rec.update({"law": "native scalar source e_x^nat(s), wick, frozen s = 1 sea", "q": a.q, "U": a.U, "Nc": Nc,
                "host": socket.gethostname(), "note": "block = the whole sector; gM/gB fields of the record are unused"})
    tag = "q%d_m%g_U%g_k%g" % (a.q, a.m, a.U, a.k)
    with open(os.path.join(a.out, "native_%s.json" % tag), "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    print("native %s done in %.1f s, ok %s" % (tag, time.time() - t0, rec["continuation"].get("ok")), flush=True)


if __name__ == "__main__":
    main()
