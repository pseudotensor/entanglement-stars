#!/usr/bin/env python
"""engine_validate3d_v2.py (builder analyst, version 2 of the lead's draft: the site densities use the sparse h and Q (version 1 densified
them and multiplied 13500 x 13500 by 13500 x 6750 on one CPU thread, about 30 minutes per L = 15 case with the GPU idle; the sums are the
same), and the builder item of version 1 is done: the 3D positive levels below the largest count's E_F grouped into degenerate sets
(tolerance 1e-7) against the chain's levels with their 16 kappa multiplicities, and the counts per energy window.  SPEC_final4b_engine.md 2.4: the radial reduction against the full three-dimensional lattice at
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
    out = {"script": "engine_validate3d_v2.py", "args": vars(a), "positive_levels_below_m": ep[ep < a.m].tolist()}
    hd = h.tocsr(); Qd = [q.tocsr() for q in Q]          # version 2: sparse products (identical sums)
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
        # version 2: the degenerate sets of the 3D levels below E_F + 10 T_s against the chain's multiplets (energy, multiplicity)
        top = EF + 10 * Ts
        e3 = ep[ep < top]
        sets3 = []
        for x in e3:
            if sets3 and abs(x - sets3[-1][0]) < 1e-7:
                sets3[-1][1] += 1
            else:
                sets3.append([float(x), 1])
        chain_sets = sorted([(float(x), 16 * k) for k, x in levels if x < top])
        win = np.linspace(min(e3.min() if len(e3) else a.m, chain_sets[0][0] if chain_sets else a.m) - 1e-9, top, 9)
        cnt3 = np.histogram(e3, bins=win)[0].tolist()
        cntc = np.histogram([x for x, _ in chain_sets], bins=win, weights=[w for _, w in chain_sets])[0].tolist()
        res[str(count)] = {"EF": float(EF), "NB_3d": float(f.sum()), "NB_chain": nb, "levels_3d_sets": sets3, "levels_chain_sets": chain_sets,
                           "window_edges": win.tolist(), "count_3d_per_window": cnt3, "count_chain_per_window": cntc, "rho_shell_3d": rho_sh.tolist(), "eps_shell_chain": eps_c.tolist(),
                           "J_shell_3d": J_sh.tolist(), "J_shell_chain": J_c.tolist(), "E_3d": float(rho.sum()), "E_chain": float(eps_c.sum()),
                           "rel_shell_diff_first8": ((eps_c[:8] - rho_sh[:8]) / np.maximum(np.abs(rho_sh[:8]), 1e-300)).tolist(), "chain_levels": levels}
    out["results"] = res
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "v3d_%s.json" % a.tag), "w") as fh:
        json.dump(out, fh, default=float)
    for count, rr in res.items():
        print("count %s: E_F %.6f NB 3d %.2f chain %.2f | E 3d %.4f chain %.4f | 3d sets %s | chain sets %s" % (
            count, rr["EF"], rr["NB_3d"], rr["NB_chain"], rr["E_3d"], rr["E_chain"],
            " ".join("%.5f(%d)" % (x, d) for x, d in rr["levels_3d_sets"][:12]), " ".join("%.5f(%d)" % (x, d) for x, d in rr["levels_chain_sets"][:12])), flush=True)


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
