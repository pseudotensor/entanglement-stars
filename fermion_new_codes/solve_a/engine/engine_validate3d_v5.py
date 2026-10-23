#!/usr/bin/env python
"""engine_validate3d_v5.py (builder analyst; engine 17.4's ruling: the chain's current as a surrogate decided against the 3-dimensional ball;
version 5 = version 4 plus --chain-current {two-step, full}: "full" adds to the chain's two-step Q the nearest-neighbour term of the chain's own
current [X, h_0^2]/(2i), the bond (n, n + 1) block (i/4)(f_(n+1) - f_n) sigma_z with f_n = kappa/R_n (engine_vacflow_v5's term, the same
operator conventions), so that the shell-binned radial-momentum and energy densities and the level clusters at B = 0.3 compare both chain
currents with the 3-dimensional lattice in the same container. Nothing else changes. Version 4's docstring follows.
engine_validate3d_v4.py (builder analyst; SPEC_final4b_engine.md sec. 12.5, the lead's V3D redesign; numbered 4 because version 2 is
the sparse open-cube V3D and version 3 the builder's container-switch draft with its ball test).

Both operators in the same smooth container: a confining wall V(r) = 2 m (r/R_c)^6 entering as a site-dependent mass (m -> m + V on the
sites, clocked with N like the mass; the lead's Weyl formula p_F^2 = E^2 - (m + V)^2 reads it so; a scalar wall would let the negative band
through above V > E + m, labelled builder reading), sampled at |x - x_c| on the lattice and at R_n on the chain; the cube's open faces lie in
the forbidden region (the largest weight of the compared states on the faces printed). Primary: the even-L cube about a cube-centre point
(the 8 valleys degenerate, engine_valley_degeneracy_v1), the chain at R_n = n + 1/2; variant: the odd-L cube about a site, the chain at
R_n = n + 1 (the pair brackets the O(a) placement). R_c = L/2 - 2.5; m a = 0.3; N = 1 - 0.5 exp(-r^2/(2 s^2)), v = -B (r/s) exp(-r^2/(2 s^2)),
s = R_c/3, B in {0, 0.3}. Compared: (1) the level list: the 3D levels assigned to the chain's multiplets in order of energy (blocks of
16 kappa), each cluster's mean and width against the chain level, the lowest --nmult multiplets; (2) the Weyl count at the energies of the
whole-multiplet counts: the 3D count, the chain's 16 kappa count and 16 int 4 pi r^2 p_F^3/(6 pi^2) dr with p_F^2 = (E/N)^2 - (m + V)^2
(B = 0; with flow printed but not compared); (3) the shell-binned energy and radial-momentum densities and the totals at the counts 16, 48,
96, 160 (whole multiplets in the chain's order; the 3D count at the same E_F printed), the bins of unit width about x_c with the central
bin's lattice volume stated; (4) B = 0.3 against B = 0; (5) the L = 12 -> 16 trend read from the two records.  Dense eigh on the GPU.
Args: --L --centre (cube|site) --B --m --counts --nmult --gpu --out --tag."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v1 as OPS  # noqa: E402


def eig_dense(hm, gpu):
    import torch
    dev = "cuda:%d" % gpu if gpu is not None else "cpu"
    coo = hm.tocoo()
    idx = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=dev)
    a = torch.sparse_coo_tensor(idx, torch.as_tensor(coo.data, dtype=torch.complex128, device=dev), hm.shape).to_dense()
    e, U = torch.linalg.eigh(a)
    del a
    return e.real.cpu().numpy(), U


def run(a):
    t0 = time.time()
    L, m, B = a.L, a.m, a.B
    assert (a.centre == "cube") == (L % 2 == 0), "centre=cube needs an even L (a cube-centre point), centre=site an odd L"
    Rc = L / 2.0 - 2.5
    s = Rc / 3.0
    V = lambda r: 2.0 * m * (np.asarray(r, float) / Rc) ** 6
    Nf = lambda r: 1.0 - 0.5 * np.exp(-np.asarray(r, float) ** 2 / (2 * s * s))
    vf = lambda r: -B * (np.asarray(r, float) / s) * np.exp(-np.asarray(r, float) ** 2 / (2 * s * s))
    c = (L - 1) / 2.0
    g = np.indices((L, L, L)).reshape(3, -1).T - c
    r = np.sqrt((g ** 2).sum(1))
    h, Q = OPS.lattice3d_parts(L, m)
    nhat = np.where(r[:, None] > 0, g / np.maximum(r[:, None], 1e-300), 0.0)
    N3 = Nf(r); v3 = vf(r)[:, None] * nhat
    Nd = sp.diags(np.repeat(N3, 4))
    hm = 0.5 * (Nd @ h + h @ Nd)
    for d in range(3):
        Vd = sp.diags(np.repeat(v3[:, d], 4))
        hm = hm + 0.5 * (Vd @ Q[d] + Q[d] @ Vd)
    hm = hm + sp.kron(sp.diags(N3 * V(r)), sp.csr_matrix(OPS.BETA))
    e, U = eig_dense(hm.tocsr(), a.gpu)
    first = int(np.searchsorted(e, 0.0))
    nkeep = max(a.counts) + 400
    ep = e[first:first + nkeep]
    Up = U[:, first:first + nkeep].cpu().numpy() if hasattr(U, "cpu") else U[:, first:first + nkeep]
    del U
    # the faces' weight of the compared states
    face = (np.abs(g) >= c - 0.5).any(1)
    face_modes = (4 * np.nonzero(face)[0][:, None] + np.arange(4)[None, :]).ravel()
    face_weight = float(np.max(np.sum(np.abs(Up[face_modes, :max(a.counts) + 16]) ** 2, axis=0)))
    # the chain in the same container
    half = a.centre == "cube"
    nch = int(np.ceil(c * np.sqrt(3))) + 2
    Rn = np.arange(nch, dtype=float) + (0.5 if half else 1.0)
    chans = {}
    mult = []
    for kappa in range(1, 80):
        hc, Qc, _ = OPS.chain_parts(nch, m, float(kappa), half)
        if a.chain_current == "full":
            f = float(kappa) / Rn
            Qn = np.zeros_like(Qc)
            for n_ in range(nch - 1):
                Qn[2 * n_:2 * n_ + 2, 2 * n_ + 2:2 * n_ + 4] = 0.25j * (f[n_ + 1] - f[n_]) * OPS.SZ
            Qc = Qc + Qn + Qn.conj().T
        hcm = OPS.clocked(hc, Qc, Nf(Rn), vf(Rn)) + np.kron(np.diag(Nf(Rn) * V(Rn)), OPS.SZ)
        ec, Uc = OPS.spectrum(hcm)
        pc = ec > 0
        chans[kappa] = (hc, Qc, ec[pc], Uc[:, pc])
        mult += [(float(x), 16 * kappa, kappa, j) for j, x in enumerate(ec[pc])]
        if ec[pc].min() > ep[-1]:
            break
    mult.sort()
    # (1) the clusters: the 3D levels in blocks of the chain multiplets' weights, in order
    clusters = []
    i3 = 0
    for (x, w, k, j) in mult[: a.nmult]:
        blk = ep[i3:i3 + w]
        if len(blk) < w:
            break
        clusters.append({"chain": x, "kappa": k, "weight": w, "mean_3d": float(blk.mean()), "width_3d": float(blk.max() - blk.min()),
                         "mean_minus_chain": float(blk.mean() - x)})
        i3 += w
    # (2) and (3) at the whole-multiplet counts in the chain's order
    res = {}
    cum = np.cumsum([w for _, w, _, _ in mult])
    for count in a.counts:
        i_c = int(np.searchsorted(cum, count))
        if i_c >= len(mult) or cum[i_c] != count:
            res[str(count)] = {"note": "not a whole-multiplet count of the chain (cumulative weights %s)" % cum[: i_c + 2].tolist()}
            continue
        EF = 0.5 * (mult[i_c][0] + mult[i_c + 1][0])
        f3 = (ep < EF).astype(float)
        o3 = f3 > 0
        rho = OPS.site_density(h, Up[:, o3], f3[o3], ncomp=4) + 0.0
        Jr = sum(OPS.site_density(Q[d], Up[:, o3], f3[o3], ncomp=4) * nhat[:, d] for d in range(3))
        sh = np.floor(r).astype(int) if half else np.maximum(np.round(r - 1.0), 0).astype(int)
        rho_sh = np.bincount(sh, weights=rho, minlength=nch)[:nch]; J_sh = np.bincount(sh, weights=Jr, minlength=nch)[:nch]
        eps_c = np.zeros(nch); J_c = np.zeros(nch); nb = 0.0
        for kappa, (hc, Qc, ecp, Ucp) in chans.items():
            fc = (ecp < EF).astype(float)
            if fc.max() == 0:
                continue
            e1, J1 = OPS.sources_static(hc, Qc, Ucp[:, fc > 0], fc[fc > 0], 16.0 * kappa)
            eps_c += e1; J_c += J1; nb += 16.0 * kappa * fc.sum()
        weyl = None
        if B == 0.0:
            pf3 = lambda rr: max((EF / Nf(rr)) ** 2 - (m + V(rr)) ** 2, 0.0) ** 1.5
            weyl = 16.0 * quad(lambda rr: 4 * np.pi * rr * rr * pf3(rr) / (6 * np.pi ** 2), 0.0, 3 * Rc, limit=400)[0]
        res[str(count)] = {"EF": float(EF), "NB_3d": float(f3.sum()), "NB_chain": nb, "Weyl_continuum": weyl, "E_3d": float(rho.sum()), "E_chain": float(eps_c.sum()),
                           "rho_shell_3d": rho_sh.tolist(), "eps_shell_chain": eps_c.tolist(), "J_shell_3d": J_sh.tolist(), "J_shell_chain": J_c.tolist(),
                           "central_bin_sites": int(np.sum(sh == 0)),
                           "max_rel_shell_diff_eps_inner": float(np.max(np.abs(eps_c[:int(Rc)] - rho_sh[:int(Rc)])) / max(np.max(np.abs(rho_sh)), 1e-300))}
        print("count %d: E_F %.6f N_B 3D %.2f chain %.2f Weyl %s | E 3D %.5f chain %.5f | max inner shell diff %.3e" % (
            count, EF, f3.sum(), nb, ("%.2f" % weyl) if weyl else "-", rho.sum(), eps_c.sum(), res[str(count)]["max_rel_shell_diff_eps_inner"]), flush=True)
    out = {"script": "engine_validate3d_v5.py", "args": vars(a), "R_c": Rc, "s": s, "modes_3d": 4 * L ** 3, "face_weight_max": face_weight,
           "clusters": clusters, "results": res, "seconds": time.time() - t0,
           "wall": "site-dependent mass m -> m + 2 m (r/R_c)^6, clocked with N (builder reading of the lead's 'on-site' wall)"}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "v3d5_%s.json" % a.tag), "w"), default=float)
    for cl in clusters:
        print("multiplet kappa %d (weight %d): chain %.6f, 3D cluster mean %.6f width %.2e, mean - chain %.2e" % (
            cl["kappa"], cl["weight"], cl["chain"], cl["mean_3d"], cl["width_3d"], cl["mean_minus_chain"]))
    print("face weight max %.2e; %d modes; %.0f s" % (face_weight, 4 * L ** 3, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--centre", default="cube", choices=["cube", "site"])
    ap.add_argument("--B", type=float, default=0.0)
    ap.add_argument("--m", type=float, default=0.3)
    ap.add_argument("--counts", type=int, nargs="+", default=[16, 48, 96, 160])
    ap.add_argument("--nmult", type=int, default=10)
    ap.add_argument("--chain-current", default="two-step", choices=["two-step", "full"])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
