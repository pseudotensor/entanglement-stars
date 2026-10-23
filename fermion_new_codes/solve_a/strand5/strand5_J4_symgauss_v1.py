#!/usr/bin/env python
"""Strand 5 J4 sec. 10(o) (spec solve/specs/SPEC_strand5_kernel_star.md; builder analyst): the line kernels symmetrized at the build and the
Gauss check after it, with strand5_lattice_star_v3's Star (imported, unchanged).

The kernel family's stencil as built by strand5_lattice_star_v3.setup_kernel (the table route or the per-packet route), then symmetrized,
K_n <- (K_n + K_-n)/2 for n != 0, K_0 re-set so that sum K_n = 0 (the zero-sum adjustment after the symmetrization; the same as symmetrizing
before it, since K_0 is re-set), and k_op = the symmetrized stencil's moment (1/2) sum n^2 K_n.  Reported per family: the pre-symmetrization
asymmetry max |K_n - K_-n| (and relative to |K_0|), the moment's change k_op(sym)/k_op - 1.  Per member: the member re-solved at the same z_c
with the symmetrized operator by Star.newton from its stored solution (profile and A, B, mu_inf), the Newton iterations and residual, the
relative changes of A, B and A/B - 1, and the Gauss checks (A_vol - A)/A and (B_vol - B)/B before (the family record) and after.  Kernel
operators only (the local rows' three-point stencils are symmetric by construction).  Writes <out>/symgauss_<family>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_lattice_star_v3 as S  # noqa: E402


def symmetrize(st):
    s = {n: 0.5 * (st[n] + st[-n]) for n in st if n != 0}
    s[0] = -sum(s.values())
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    Rf = json.load(open(a.family))
    M = Rf["meta"]
    if M["operator"] != "kernel":
        raise SystemExit("kernel operators only")
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = types.SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    st, kop, info = S.setup_kernel(args)
    if abs(kop / M["k_op"] - 1) > 1e-12:
        raise SystemExit("k_op mismatch")
    asym = max(abs(st[n] - st[-n]) for n in st)
    sst = symmetrize(st)
    kop_s = 0.5 * sum(n * n * v for n, v in sst.items())
    gM, gB = M["couplings"]
    coup = (gM, gB, 1.0, 1.0)
    e1 = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    res = {"meta": {"family": a.family, "k_op": kop, "k_op_symmetrized": kop_s, "k_op_change_rel": kop_s / kop - 1, "asymmetry_max": float(asym),
                    "asymmetry_rel_K0": float(asym / abs(st[0])), "R": int(max(st))}, "members": []}
    out = os.path.join(a.out, "symgauss_" + os.path.basename(a.family))
    for d in [d for d in Rf["members"] if d.get("newton_ok")]:
        zc = d["z_c"]
        key = "z%.6f" % zc
        r = prof[key + "_r"]
        psi = prof[key + "_psi"]
        N = prof[key + "_N"]
        J = len(r)
        star = S.Star(J, sst, kop_s, M["mlight"], coup, e1)
        x0 = np.concatenate([psi, N * psi, [d["A"], d["B"], d["mu_inf"]]])
        x, inf = star.newton(x0, float(zc))
        dd = star.diagnostics(x, float(zc))
        rec = {"z_c": zc, "J": J, "newton_ok": bool(inf["ok"]), "newton_it": inf["it"], "residual": inf["res"],
               "dA_rel": dd["A"] / d["A"] - 1, "dB_rel": dd["B"] / d["B"] - 1, "d_AoB": dd["A_over_B_minus_1"] - d["A_over_B_minus_1"],
               "gauss_A_before": d["A_vol_minus_A"] / d["A"], "gauss_B_before": d["B_vol_minus_B"] / d["B"],
               "gauss_A_after": dd["A_vol_minus_A"] / dd["A"], "gauss_B_after": dd["B_vol_minus_B"] / dd["B"]}
        res["members"].append(rec)
        print("z_c %.4f J %d: Newton %s it %d res %.1e; dA %.1e dB %.1e d(A/B-1) %.1e; Gauss A %.2e -> %.2e, B %.2e -> %.2e (%.0f s)" % (
            zc, J, inf["ok"], inf["it"], inf["res"], rec["dA_rel"], rec["dB_rel"], rec["d_AoB"], rec["gauss_A_before"], rec["gauss_A_after"],
            rec["gauss_B_before"], rec["gauss_B_after"], time.time() - t0), flush=True)
        with open(out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    res["meta"]["seconds"] = time.time() - t0
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
