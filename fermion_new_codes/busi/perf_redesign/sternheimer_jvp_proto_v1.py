#!/usr/bin/env python
"""sternheimer_jvp_proto_v1.py (busi lane, reviewer A redesign prototype R3): the Newton engines' exact J.v density response by shifted
banded solves (one per occupied state, the banded LU kernel of banded_eigh_gpu_v1) against the production formula in the eigenbasis
(engine_collapse_v36r.nk_stage_v32.jvp: Hm = Uu^H dh Uo, M = Hm / (e_o - e_u), X = Uu M, then the site densities of X against Uo).
The Sternheimer route: for each occupied o, b_o = -dh u_o deflated of u_o, y_o = (h - e_o)^{-1} b_o (the pivmin-regularized banded solve),
y_o deflated of u_o; the first-order projector change Y Uo^H + Uo Y^H equals X Uo^H + Uo X^H exactly, the occupied-occupied components of
Y cancelling pairwise (c_{o'o} = -conj(c_{oo'})), so the site densities agree; no n x n x n_o product appears.
usage: python sternheimer_jvp_proto_v1.py OUT_DIR CODE_DIR W2_SLICE_NPZ W2_FIELDS_NPZ [--channels 1,2,5] [--gpu 0]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import banded_eigh_gpu_v1 as BE
import banded_eigh_proto_v1 as PR


def dens_pair(A_t, Uo, X):
    """the engine's density response per site: sum_o Re[u_o^* (A x_o)] + Re[x_o^* (A u_o)], summed over the two components."""
    d_ = (Uo.conj() * A_t(X)).real.sum(dim=1) + (X.conj() * A_t(Uo)).real.sum(dim=1)
    return d_.reshape(-1, 2).sum(dim=1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("code_dir"); ap.add_argument("w2_slice"); ap.add_argument("w2_fields")
    ap.add_argument("--channels", default="1,2,5"); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--m", type=float, default=0.2)
    a = ap.parse_args()
    sys.path.insert(0, a.code_dir)
    import engine_vacflow_v1 as VF
    os.makedirs(a.out_dir, exist_ok=True)
    dev = torch.device("cuda:%d" % a.gpu); torch.cuda.set_device(dev)
    z = np.load(a.w2_slice, allow_pickle=False)
    zf = np.load(a.w2_fields, allow_pickle=False)
    R = np.asarray(z["R"], float); N = np.asarray(z["N_applied"], float); v = np.asarray(z["v_applied"], float)
    T_slice = float(z["T"])
    # the perturbation: the applied fields' change over the record before the slice (a realistic Newton direction scale)
    Ti = np.asarray(zf["T"], float); i1 = int(np.argmin(np.abs(Ti - T_slice))); i0 = max(0, i1 - 1)
    dN = np.asarray(zf["N_applied"][i1], float) - np.asarray(zf["N_applied"][i0], float)
    dv = np.asarray(zf["v_applied"][i1], float) - np.asarray(zf["v_applied"][i0], float)
    lines = ["sternheimer_jvp_proto_v1: slice T %.4f, perturbation = applied fields at record %d minus %d (max |dN| %.2e, max |dv| %.2e), GPU %s" % (
        T_slice, i1, i0, np.max(np.abs(dN)), np.max(np.abs(dv)), torch.cuda.get_device_name(dev))]
    out = []
    n = 2 * len(R)
    for kap in [int(c) for c in a.channels.split(",")]:
        h0, Q = VF.chain_sparse(R, a.m, float(kap))
        hm = PR.clocked_csr(h0, Q, N, v)
        dh = PR.clocked_csr(h0, Q, dN, dv)
        band = BE.band_from_csr(hm); band_t = torch.as_tensor(band, device=dev)
        hb = BE.band_from_csr(h0.tocsr()); qb = BE.band_from_csr(Q.tocsr()); dhb = BE.band_from_csr(dh)
        hb_t = torch.as_tensor(hb, device=dev); qb_t = torch.as_tensor(qb, device=dev); dhb_t = torch.as_tensor(dhb, device=dev)
        # the eigenbasis (the dense solver, the production's) and the occupied set: the stored sea mask plus the dust levels by overlap
        hd = torch.as_tensor(hm.toarray(), device=dev)
        e, U = torch.linalg.eigh(hd); e = e.real; del hd
        mask = PR.unpack_mask(z["mask_%d" % kap], n)
        dust = torch.as_tensor(np.asarray(z["dust_%d" % kap]), device=dev)
        n_d = dust.shape[1]
        if n_d:
            w = ((U.conj().T @ dust).abs() ** 2).sum(dim=1).cpu().numpy()
            w[mask] = -1.0
            top = np.argsort(-w)[:n_d]
            mask = mask.copy(); mask[top] = True
        occ = torch.as_tensor(mask, device=dev)
        Uo = U[:, occ]; Uu = U[:, ~occ]; eo = e[occ]; eu = e[~occ]
        rec = {"kappa": kap, "n_mat": n, "n_occ": int(occ.sum()), "n_dust": int(n_d)}
        # ---- the production formula
        torch.cuda.synchronize(); t0 = time.time()
        dhUo = BE.band_matvec(dhb_t, Uo)
        Hm = Uu.conj().T @ dhUo
        den = eo[None, :] - eu[:, None]
        small = den.abs() < 1e-12
        M = torch.where(small, torch.zeros_like(Hm), Hm / torch.where(small, torch.ones_like(den), den).to(Hm.dtype))
        X = Uu @ M
        d_eps_ref = dens_pair(lambda Y: BE.band_matvec(hb_t, Y), Uo, X); d_J_ref = dens_pair(lambda Y: BE.band_matvec(qb_t, Y), Uo, X)
        torch.cuda.synchronize(); t_ref = time.time() - t0
        rec["dropped_pairs"] = int(small.sum()); rec["min_abs_den"] = float(den.abs().min())
        # ---- the Sternheimer route (warm-up once for the kernel compile, then timed)
        for rep in range(2):
            torch.cuda.synchronize(); t0 = time.time()
            B = -BE.band_matvec(dhb_t, Uo)                 # the engine's M = Hm / (e_o - e_u) is the solution of (h - e_o) x = -dh u_o
            c = (Uo.conj() * B).sum(dim=0)
            B = B - Uo * c[None, :]
            Y = BE.solve_shifted(band, n, eo.cpu().numpy(), B)
            c2 = (Uo.conj() * Y).sum(dim=0)
            Y = Y - Uo * c2[None, :]
            d_eps_st = dens_pair(lambda Z: BE.band_matvec(hb_t, Z), Uo, Y); d_J_st = dens_pair(lambda Z: BE.band_matvec(qb_t, Z), Uo, Y)
            torch.cuda.synchronize(); t_st = time.time() - t0
        # the solve's residual (the deflated system)
        resid = BE.band_matvec(band_t, Y) - Y * eo[None, :].to(Y.dtype) - B
        rec["solve_residual_rel"] = float(torch.linalg.norm(resid) / torch.linalg.norm(B))
        # the unoccupied-block agreement: Uu^H Y against M
        MY = Uu.conj().T @ Y
        rec["unocc_block_max_abs_diff"] = float((MY - M).abs().max()); rec["M_max_abs"] = float(M.abs().max())
        sc_e = max(float(np.max(np.abs(d_eps_ref))), 1e-300); sc_j = max(float(np.max(np.abs(d_J_ref))), 1e-300)
        rec["deps_max_abs"] = sc_e; rec["dJ_max_abs"] = sc_j
        rec["deps_rel_diff"] = float(np.max(np.abs(d_eps_st - d_eps_ref))) / sc_e; rec["dJ_rel_diff"] = float(np.max(np.abs(d_J_st - d_J_ref))) / sc_j
        rec["t_eigenbasis_s"] = t_ref; rec["t_sternheimer_s"] = t_st; rec["factor_box"] = t_ref / t_st
        line = "kappa %d: n %d, occupied %d (dust %d), min |e_o - e_u| %.2e, dropped %d; eigenbasis route %.3f s, Sternheimer route %.3f s (factor %.1fx on this GPU); densities rel diff eps %.2e J %.2e; unoccupied block max |diff| %.2e (|M| max %.2e); solve residual rel %.1e" % (
            kap, n, rec["n_occ"], n_d, rec["min_abs_den"], rec["dropped_pairs"], t_ref, t_st, rec["factor_box"], rec["deps_rel_diff"], rec["dJ_rel_diff"], rec["unocc_block_max_abs_diff"], rec["M_max_abs"], rec["solve_residual_rel"])
        lines.append(line); print(line, flush=True)
        out.append(rec)
        del U, Uo, Uu, X, Y, B, Hm, M
        torch.cuda.empty_cache()
    with open(os.path.join(a.out_dir, "sternheimer_jvp_proto_v1.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out_dir, "sternheimer_jvp_proto_v1.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
