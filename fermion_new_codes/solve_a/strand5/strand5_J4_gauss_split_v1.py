#!/usr/bin/env python
"""Strand 5 J4 sec. 10(m)(3), the split of the zero-sum/symmetry term of the Gauss decomposition (builder analyst; strand5_J4_exterior_v1's
identity): Z = sum_{j <= J+R} j (K * u)_j + k_mom A, u = r (psi - 1) on the grid (odd at r = 0), u = A beyond J, for (i) the stencil as used,
(ii) the stencil symmetrized, K_n -> (K_n + K_-n)/2, with K_0 re-set so that sum K_n = 0 in float64, (iii) the symmetrized stencil with the
whole sum j (K * u)_j and K_0's zero-sum accumulated in long double (numpy.longdouble), the same u.  Z(i) - Z(ii) is the asymmetry's
part, Z(ii) - Z(iii) the float64 zero-sum and summation rounding, Z(iii) the remainder; each over k_op A.  k_mom is the moment of the
stencil in each case.  Members by index (default: the dilute and the most compact).  Writes <out>/gauss_split_<family>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_lattice_star_v3 as S  # noqa: E402


def T_full(stencil, u_grid, A, J, R, dtype):
    """sum_{j=1}^{J+R} j sum_n K_n u_{j-n}, u odd at 0, u = A beyond J (and -A below -J), accumulated in dtype."""
    Jx = J + R
    ns = np.array(sorted(stencil))
    K = np.array([stencil[n] for n in ns], dtype=dtype)
    # u on -(Jx + R) .. (Jx + R)
    M = Jx + R + 1
    u = np.zeros(2 * M + 1, dtype=dtype)
    idx = np.arange(-M, M + 1)
    pos = idx > 0
    a = np.abs(idx)
    vals = np.where(a <= J, np.concatenate([[0.0], np.asarray(u_grid, float)])[np.minimum(a, J)], A)
    u = np.where(pos, vals, -vals).astype(dtype)
    u[idx == 0] = 0
    tot = dtype(0)
    for j in range(1, Jx + 1):
        uj = u[(j - ns) + M]
        tot += dtype(j) * np.sum(K * uj, dtype=dtype)
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--members", default="0,-1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Rf = json.load(open(a.family))
    M = Rf["meta"]
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = types.SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    st, kop, info = S.setup_kernel(args)
    R = max(st)
    sym = {n: 0.5 * (st[n] + st[-n]) for n in st}
    sym[0] = -sum(v for n, v in sym.items() if n != 0)
    symL = {n: np.longdouble(0.5) * (np.longdouble(st[n]) + np.longdouble(st[-n])) for n in st}
    symL[0] = -sum(v for n, v in symL.items() if n != 0)
    mom = lambda s_: sum(np.longdouble(n) * n * s_[n] for n in s_) / 2
    mem = [d for d in Rf["members"] if d.get("newton_ok")]
    res = {"meta": {"family": a.family, "k_op": kop, "R": R, "zero_sum_used": float(sum(st.values())), "zero_sum_sym64": float(sum(sym.values())),
                    "zero_sum_symL": float(sum(symL.values())), "asymmetry_max": float(max(abs(st[n] - st[-n]) for n in st)),
                    "K0": st[0]}, "members": []}
    for i in [int(x) for x in a.members.split(",")]:
        d = mem[i]
        key = "z%.6f" % d["z_c"]
        r = prof[key + "_r"]
        psi = prof[key + "_psi"]
        J = len(r)
        A = d["A"]
        ug = r * (psi - 1)
        out = {"z_c": d["z_c"], "J": J, "A": A}
        Zi = float(T_full(st, ug, A, J, R, np.float64) + mom(st) * A)
        Zii = float(T_full(sym, ug, A, J, R, np.float64) + mom(sym) * A)
        Ziii = float(T_full(symL, ug, np.longdouble(A), J, R, np.longdouble) + mom(symL) * np.longdouble(A))
        s = kop * A
        out.update({"Z_used_rel": Zi / s, "Z_sym_float64_rel": Zii / s, "Z_sym_longdouble_rel": Ziii / s,
                    "asymmetry_part_rel": (Zi - Zii) / s, "float64_zero_sum_and_rounding_part_rel": (Zii - Ziii) / s, "remainder_rel": Ziii / s,
                    "record_A_vol_minus_A_rel": d["A_vol_minus_A"] / A})
        res["members"].append(out)
        print("z_c %.4f J %d: -Z/(k A) as used %.3e; asymmetry part %.3e, float64 zero-sum and rounding %.3e, long-double remainder %.3e; record (A_vol - A)/A %.3e" % (
            d["z_c"], J, -Zi / s, -(Zi - Zii) / s, -(Zii - Ziii) / s, -Ziii / s, d["A_vol_minus_A"] / A), flush=True)
    with open(os.path.join(a.out, "gauss_split_" + os.path.basename(a.family)), "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
