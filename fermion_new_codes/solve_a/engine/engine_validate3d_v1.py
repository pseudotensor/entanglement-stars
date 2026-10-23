#!/usr/bin/env python
"""engine_validate3d_v1.py (lead's draft; SPEC_final4b_engine.md 2.4): the radial reduction against the full three-dimensional lattice at
small size.  Open L^3 box, the centre a site; prescribed N(r) = 1 - A exp(-r^2/2s^2), v(r) = -B (r/s) exp(-r^2/2s^2) radial; m a = 0.3;
the positive band of h_m[N, v] filled to a state count; the normal-ordered rho_x, J_{x,r} by form (i), binned in shells, against the chain
reduction at the same profile and E_F.  Dense eigh (4 L^3 up to 13500 on the GPU)."""
import argparse
import json
import os

import numpy as np

import engine_ops_v1 as OPS
import engine_fields_v1 as FLD


def profiles(L, A, B, s):
    c = (L - 1) / 2
    g = np.arange(L) - c
    X = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    r = np.linalg.norm(X, axis=1)
    N = 1 - A * np.exp(-r ** 2 / (2 * s * s))
    vmag = -B * (r / s) * np.exp(-r ** 2 / (2 * s * s))
    nhat = np.where(r[:, None] > 0, X / np.maximum(r[:, None], 1e-300), 0.0)
    return N, vmag[:, None] * nhat, r, nhat


def run(a):
    OPS.GPU["dev"] = a.gpu
    L = a.L
    h, Q = OPS.lattice3d_parts(L, a.m)
    N3, v3, r, nhat = profiles(L, a.A, a.B, a.s)
    hm = OPS.clocked3d(h, Q, N3, v3)
    e, U = OPS.spectrum(hm)
    pos = e > 0
    ep = e[pos]
    out = {"script": "engine_validate3d_v1.py", "args": vars(a), "positive_levels_below_m": ep[ep < a.m].tolist()}
    hd = h.toarray(); Qd = [q.toarray() for q in Q]
    res = {}
    for count in a.counts:
        EF = 0.5 * (ep[count - 1] + ep[count]) if count < len(ep) else ep[-1] + 1e-6
        Ts = 0.01 * a.m
        f = OPS.fermi(ep, EF, Ts)
        Up = U[:, pos]
        rho = OPS.site_density(hd, Up, f, ncomp=4)
        Jr = sum(OPS.site_density(Qd[d], Up, f, ncomp=4) * nhat[:, d] for d in range(3))
        rho_sh = OPS.shell_bin(L, rho); J_sh = OPS.shell_bin(L, Jr)
        # the chain reduction at the same profile: eps_n^tot, J_n^tot
        nsites = len(rho_sh)
        R = np.arange(1, nsites + 1, dtype=float)
        Nc = 1 - a.A * np.exp(-R ** 2 / (2 * a.s ** 2)); vc = -a.B * (R / a.s) * np.exp(-R ** 2 / (2 * a.s ** 2))
        eps_c = np.zeros(nsites); J_c = np.zeros(nsites); nb = 0.0; levels = []
        for kappa in range(1, 40):
            hc, Qc, _ = OPS.chain_parts(nsites, a.m, float(kappa), bool(a.half_offset))
            ec, Uc = OPS.spectrum(OPS.clocked(hc, Qc, Nc, vc))
            pc = ec > 0
            fc = OPS.fermi(ec[pc], EF, Ts)
            if fc.max() < 1e-12:
                break
            e1, J1 = OPS.sources_static(hc, Qc, Uc[:, pc], fc, 16.0 * kappa)
            eps_c += e1; J_c += J1; nb += 16.0 * kappa * fc.sum(); levels += [(kappa, float(x)) for x in ec[pc][fc > 1e-6]]
        res[str(count)] = {"EF": float(EF), "NB_3d": float(f.sum()), "NB_chain": nb, "rho_shell_3d": rho_sh.tolist(), "eps_shell_chain": eps_c.tolist(),
                           "J_shell_3d": J_sh.tolist(), "J_shell_chain": J_c.tolist(), "E_3d": float(rho.sum()), "E_chain": float(eps_c.sum()),
                           "rel_shell_diff_first8": ((eps_c[:8] - rho_sh[:8]) / np.maximum(np.abs(rho_sh[:8]), 1e-300)).tolist(), "chain_levels": levels}
    out["results"] = res
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "v3d_%s.json" % a.tag), "w") as fh:
        json.dump(out, fh, default=float)
    # BUILDER: the degeneracy of the 3D levels against the chain's 16 kappa multiplets (the cubic group splits them: report the splittings);
    # the count the chain must match is the 3D count at the same E_F, not the other way round; repeat at L = 15, s = 4, B = 0 and 0.3.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=11)
    ap.add_argument("--m", type=float, default=0.3)
    ap.add_argument("--A", type=float, default=0.3)
    ap.add_argument("--B", type=float, default=0.0)
    ap.add_argument("--s", type=float, default=3.0)
    ap.add_argument("--counts", type=int, nargs="+", default=[16, 48, 120])
    ap.add_argument("--half-offset", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
