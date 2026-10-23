#!/usr/bin/env python
"""engine_validate3d_v3.py (builder analyst; SPEC_final4b_engine.md sec. 11.5, the lead's V3D redesign, numbered version 3 because the
name version 2 is taken by the sparse-product V3D of the open cube; the container is a switch until the lead rules on the builder's
valley-degeneracy test, turn 19).

Both operators in the same spherical container: --container ball_cc: the lattice ball |x - c| <= R_wall about a cube-centre point c (even L;
the 8 valleys degenerate, engine_valley_degeneracy_v1) with Dirichlet outside, the chain with R_n = n + 1/2 (the distances from a cube centre)
and its last site at R_wall; --container ball_site: the same about a site (odd L; the valleys split: the control of the container effect),
the chain R_n = n + 1; --container wall: a smooth mass wall U(r) beta = U_w beta/(1 + exp(-(r - R_wall)/w_U)) in a periodic even box
(no lattice boundary inside; U_w = 2, w_U = 1 by default), the chain with the same wall on site (U(R_n) sigma_z) and its end at R_wall + 6 w_U.
Profile N = 1 - A exp(-r^2/(2 s^2)), v = -B (r/s) exp(-r^2/(2 s^2)), s = R_wall/3 (A = 0.5, B in {0, 0.3}), m a = 0.3 (0.15 as the k a
trend).  Compared at the whole-multiplet counts 16, 48, 96, 160 (E_F between the count's level and the next; zero-temperature filling):
(1) the level sets of the 3D operator (energies within 1e-7) against the chain's levels with their 16 kappa multiplicities: per chain level
the nearest 3D cluster of the same total multiplicity, the cluster's mean and width; (2) the shell-binned energy and radial-momentum
densities of the filled states (shells of unit width about c: [n, n + 1) for ball_cc and wall, [n + 1/2, n + 3/2) about a site with the centre
site in shell 0 for ball_site) against the chain's eps_n, J_n; (3) the totals N_B and E.  The R_wall trend (7, 11, 13 if the matrix fits) is
the acceptance reading (sec. 2.4: differences falling as (a/R_wall)^2).  Dense eigh on the GPU (complex128 up to 22300 modes; complex64
beyond, labelled).  Args: --container --R-wall --A --B --m --counts --gpu --out --tag."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v1 as OPS  # noqa: E402


def container_sites(container, R_wall, wU=1.0):
    if container == "ball_cc":
        L = 2 * int(np.ceil(R_wall)) + 2
        c = (L - 1) / 2.0
    elif container == "ball_site":
        L = 2 * int(np.ceil(R_wall)) + 3
        c = (L - 1) / 2.0
    else:                                                       # wall: the periodic even box beyond the wall's tail
        L = 2 * int(np.ceil(R_wall + 8 * wU)) + 2
        c = (L - 1) / 2.0
    g = np.indices((L, L, L)).reshape(3, -1).T - c
    r = np.sqrt((g ** 2).sum(1))
    sel = np.nonzero(r <= R_wall)[0] if container != "wall" else np.arange(L ** 3)
    return L, c, g, r, sel


def build_3d(container, R_wall, m, A, B, s, wU, Uw):
    L, c, g, r, sel = container_sites(container, R_wall, wU)
    h, Q = OPS.lattice3d_parts(L, m, periodic=(container == "wall"))
    modes = (4 * sel[:, None] + np.arange(4)[None, :]).ravel()
    hs = h.tocsr()[modes][:, modes]
    Qs = [q.tocsr()[modes][:, modes] for q in Q]
    rs = r[sel]
    N = 1 - A * np.exp(-rs ** 2 / (2 * s * s))
    nhat = np.where(rs[:, None] > 0, g[sel] / np.maximum(rs[:, None], 1e-300), 0.0)
    vmag = -B * (rs / s) * np.exp(-rs ** 2 / (2 * s * s))
    vvec = vmag[:, None] * nhat
    # the clocked operator kept sparse (engine_ops_v1.clocked3d's formula without its dense conversion)
    Nd = sp.diags(np.repeat(N, 4))
    hm = 0.5 * (Nd @ hs + hs @ Nd)
    for d in range(3):
        Vd = sp.diags(np.repeat(vvec[:, d], 4))
        hm = hm + 0.5 * (Vd @ Qs[d] + Qs[d] @ Vd)
    if container == "wall":
        U = Uw / (1.0 + np.exp(-(rs - R_wall) / wU))
        hm = hm + sp.kron(sp.diags(U), sp.csr_matrix(OPS.BETA))
    if hm.shape[0] > 40000:
        raise ValueError("BUILDER: %d modes need a sparse eigensolver (shift-invert near m) before this container runs" % hm.shape[0])
    return hm.tocsr(), hs, Qs, rs, nhat, len(sel)


def eig_gpu(hm, gpu, n_keep):
    """the sparse operator densified on the device; all eigenvalues returned, the eigenvectors of the lowest n_keep positive levels only."""
    import torch
    n = hm.shape[0]
    dt = torch.complex128 if n <= 22300 else torch.complex64
    dev = "cuda:%d" % gpu if gpu is not None else "cpu"
    coo = hm.tocoo()
    idx = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=dev)
    a = torch.sparse_coo_tensor(idx, torch.as_tensor(coo.data, dtype=dt, device=dev), (n, n)).to_dense()
    e, U = torch.linalg.eigh(a)
    del a
    er = e.real.double().cpu().numpy()
    first = int(np.searchsorted(er, 0.0))
    keep = U[:, first:first + n_keep].to(torch.complex128).cpu().numpy()
    del e, U
    return (er, keep, first), ("complex128" if dt == torch.complex128 else "complex64")


def sets_of(ep, tol=1e-7):
    s = []
    for x in ep:
        if s and abs(x - s[-1][0]) < tol:
            s[-1][1] += 1
        else:
            s.append([float(x), 1])
    return s


def run(a):
    t0 = time.time()
    s = a.R_wall / 3.0
    hm, hs, Qs, rs, nhat, nsite = build_3d(a.container, a.R_wall, a.m, a.A, a.B, s, a.wU, a.Uw)
    n_keep = max(a.counts) + 200
    (e, Up, first), prec = eig_gpu(hm, a.gpu, n_keep)
    ep = e[first:first + n_keep]
    sets3 = sets_of(ep[: max(a.counts) + 200])
    # the chain in the same container
    half = a.container in ("ball_cc", "wall")
    nch = int(np.floor(a.R_wall - (0.5 if half else 1.0))) + 1 if a.container != "wall" else int(np.ceil(a.R_wall + 6 * a.wU))
    Rc = np.arange(nch, dtype=float) + (0.5 if half else 1.0)
    Nc = 1 - a.A * np.exp(-Rc ** 2 / (2 * s * s)); vc = -a.B * (Rc / s) * np.exp(-Rc ** 2 / (2 * s * s))
    chain_levels = []
    chans = {}
    for kappa in range(1, 60):
        hc, Qc, _ = OPS.chain_parts(nch, a.m, float(kappa), half)
        hcm = OPS.clocked(hc, Qc, Nc, vc)
        if a.container == "wall":
            hcm = hcm + np.kron(np.diag(a.Uw / (1.0 + np.exp(-(Rc - a.R_wall) / a.wU))), OPS.SZ)
        ec, Uc = OPS.spectrum(hcm)
        pc = ec > 0
        chans[kappa] = (hc, Qc, ec[pc], Uc[:, pc])
        chain_levels += [(float(x), 16 * kappa, kappa) for x in ec[pc]]
        if ec[pc].min() > ep[max(a.counts) + 100]:
            break
    chain_levels.sort()
    res = {}
    for count in a.counts:
        # the 3D E_F: between the count-th and the next level (the counts are whole multiplets: reported if not)
        EF = 0.5 * (ep[count - 1] + ep[count])
        f3 = (ep < EF).astype(float)
        whole3 = abs(ep[count] - ep[count - 1]) > 1e-7
        rho = OPS.site_density(hs, Up[:, f3 > 0], f3[f3 > 0], ncomp=4)
        Jr = sum(OPS.site_density(Qs[d], Up[:, f3 > 0], f3[f3 > 0], ncomp=4) * nhat[:, d] for d in range(3))
        if a.container == "ball_site":
            sh = np.maximum(np.round(rs - 1.0), 0).astype(int)
        else:
            sh = np.floor(rs).astype(int)
        rho_sh = np.bincount(sh, weights=rho, minlength=nch)[:nch]; J_sh = np.bincount(sh, weights=Jr, minlength=nch)[:nch]
        eps_c = np.zeros(nch); J_c = np.zeros(nch); nb = 0.0
        for kappa, (hc, Qc, ecp, Ucp) in chans.items():
            fc = (ecp < EF).astype(float)
            if fc.max() == 0:
                continue
            e1, J1 = OPS.sources_static(hc, Qc, Ucp[:, fc > 0], fc[fc > 0], 16.0 * kappa)
            eps_c += e1; J_c += J1; nb += 16.0 * kappa * fc.sum()
        res[str(count)] = {"EF": float(EF), "whole_multiplet_3d": bool(whole3), "NB_3d": float(f3.sum()), "NB_chain": nb, "E_3d": float(rho.sum()), "E_chain": float(eps_c.sum()),
                           "rho_shell_3d": rho_sh.tolist(), "eps_shell_chain": eps_c.tolist(), "J_shell_3d": J_sh.tolist(), "J_shell_chain": J_c.tolist(),
                           "max_rel_shell_diff_eps": float(np.max(np.abs(eps_c - rho_sh)) / max(np.max(np.abs(rho_sh)), 1e-300))}
        print("count %d: E_F %.6f (3D whole multiplet %s) N_B 3D %.2f chain %.2f | E 3D %.5f chain %.5f | max shell diff %.3e" % (
            count, EF, whole3, f3.sum(), nb, rho.sum(), eps_c.sum(), res[str(count)]["max_rel_shell_diff_eps"]), flush=True)
    out = {"script": "engine_validate3d_v3.py", "args": vars(a), "n_sites_3d": nsite, "modes_3d": 4 * nsite, "precision": prec,
           "levels_3d_sets": sets3[:60], "levels_chain": chain_levels[:60], "results": res, "seconds": time.time() - t0}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "v3d3_%s.json" % a.tag), "w"), default=float)
    print("3D sets: %s" % " ".join("%.5f(%d)" % (x, d) for x, d in sets3[:14]))
    print("chain:   %s" % " ".join("%.5f(%d,k%d)" % (x, w, k) for x, w, k in chain_levels[:10]))
    print("%d sites, %s, %.0f s" % (nsite, prec, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="ball_cc", choices=["ball_cc", "ball_site", "wall"])
    ap.add_argument("--R-wall", type=float, default=7.0)
    ap.add_argument("--A", type=float, default=0.5)
    ap.add_argument("--B", type=float, default=0.0)
    ap.add_argument("--m", type=float, default=0.3)
    ap.add_argument("--wU", type=float, default=1.0)
    ap.add_argument("--Uw", type=float, default=2.0)
    ap.add_argument("--counts", type=int, nargs="+", default=[16, 48, 96, 160])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
