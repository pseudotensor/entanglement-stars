#!/usr/bin/env python
"""engine_massfluct_v2.py (builder analyst; version 2 of engine_massfluct_v1.py for the zero-temperature fixed-N_B members of engine_star_v5
(sec. 12.1's filling): the occupations rebuilt exactly as engine_star_v5.channel_pass does (all channels until two consecutive ones lie wholly
above E_top; the pooled sorted positive levels with the weights 16 kappa; the lowest N_B filled, the last multiplet fractional), by a first
pass over the channels' eigenvalues and a second pass for the eigenvectors and the fluctuation; the reading radii R_surf and R_999 (version 5's
records carry no r_star). Version 1 read engine_star_v2's EF_used, Ts and filling. Version 1's docstring follows.

engine_massfluct_v1.py (builder analyst; SPEC_final4b_engine.md sec. 11.6 (2) and (3), written from the lead's formulas): the Hartree
star's mass-function fluctuation and its monopole exchange energy.

For a converged member of engine_star_v2 (its fields N, v and its filling), per channel kappa the eigenmodes of h_m^{(kappa)}[N, v] are
recomputed and the occupations rebuilt (zeroT_EF: f = 1 below E_F; zeroT_NB: the member's N_B with the last degenerate set fractional;
fermi: the member's T_s); the sea (negative modes) has f = 1.  The chain operator of the mass function inside the site c is
m^{(kappa)}(R_c) = sum_{n <= c} hcal_n with hcal_n = (1/2){P_n, h_chain} (the unclocked chain operator: the energy density of spec 2.2).
  F^{(kappa)}(c) = sum_{k, k'} f_k (1 - f_k') |<k'| m^{(kappa)}(R_c) |k>|^2       (the quasi-free state's connected correlator)
split into the matter's part (k in the positive band) and the sea's (k in the sea); Var m(R_c) = sum_kappa 16 kappa F^{(kappa)}(c) (the
copies are independent: m(r) connects only states of one j, m_j and valley; the lead's sec. 11.6 (2)).  The monopole exchange estimate
  E_x^(0) = (1/2) sum_kappa 16 kappa sum_{k, k'} f_k (1 - f_k') sum_{n n'} V_nn' <k|hcal_n|k'><k'|hcal_n'|k>,   V_nn' = -g/max(R_n, R_n'),
by the cumulative form sum_{nn'} V_nn' x_n conj(x_n') = sum_n (-g/R_n) [|x_n|^2 + 2 Re(conj(x_n) sum_{n' < n} x_n')], all (k, k') pairs at
once with the weights sqrt(f_k) and sqrt(1 - f_k').  Both by a cumulative loop over the sites (torch on the GPU when --gpu is given).
Outputs: the radial profiles sqrt(Var m(r))/m(r) for the matter and the sea, the value at r_star, sqrt(Var_mat m(r_star))/E against
1/sqrt(N_B), E_x^(0) (matter, sea, total) against the Hartree gravitational energy estimate E - N_B m - (kinetic) (BUILDER-pending: the
kinetic split), and the per-channel shares.  Args: --star (engine_star_v2 JSON) --member (index, default the maximum member) --gpu --out --tag."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_ops_v1 as OPS  # noqa: E402
import engine_star_v2 as ST  # noqa: E402


def cumulative_channel(h, U, fw, R, g, dev=None):
    """per channel: the cumulative F(c) (matter and sea parts) for every site c and the exchange sum.  U: all eigenvectors (columns),
    fw: occupations of all modes; returns arrays F_mat[c], F_sea[c] and the exchange (matter, sea)."""
    import torch
    td = torch.device("cuda:%d" % dev) if dev is not None else torch.device("cpu")
    occ = np.nonzero(fw > 0)[0]
    un = np.nonzero(fw < 1)[0]
    Uo = torch.as_tensor(U[:, occ], device=td)
    Uu = torch.as_tensor(U[:, un], device=td)
    H = torch.as_tensor(h, device=td)
    Ao = H @ Uo                                              # h U_occ
    Bu = H @ Uu                                              # h U_un
    so = torch.as_tensor(np.sqrt(fw[occ]), device=td)
    su = torch.as_tensor(np.sqrt(1.0 - fw[un]), device=td)
    nsites = h.shape[0] // 2
    X = torch.zeros((len(un), len(occ)), dtype=Uo.dtype, device=td)   # weighted cumulative <k'| m(R_c) |k>
    return occ, un, Uo, Uu, Ao, Bu, so, su, X, nsites, td


def channel_fluct(h, U, fw, is_sea, R, g, dev=None):
    import torch
    occ, un, Uo, Uu, Ao, Bu, so, su, X, nsites, td = cumulative_channel(h, U, fw, R, g, dev)
    sea_cols = torch.as_tensor(is_sea[occ], device=td)
    mat_cols = ~sea_cols
    F_mat = np.zeros(nsites); F_sea = np.zeros(nsites)
    Ex_mat = 0.0; Ex_sea = 0.0
    for n in range(nsites):
        sl = slice(2 * n, 2 * n + 2)
        dM = 0.5 * (Uu[sl].conj().T @ Ao[sl] + Bu[sl].conj().T @ Uo[sl])    # <k'| hcal_n |k>
        dW = (su[:, None] * dM) * so[None, :]
        # the exchange: (-g/R_n) [|x_n|^2 + 2 Re(conj(x_n) X_{n-1})] summed with the weights, split by the column's band
        cross = (dW.conj() * X).real
        sq = (dW.conj() * dW).real
        e_n = (-g / R[n]) * (sq + 2.0 * cross)
        Ex_mat += float(e_n[:, mat_cols].sum()); Ex_sea += float(e_n[:, sea_cols].sum())
        X = X + dW
        a2 = (X.conj() * X).real
        F_mat[n] = float(a2[:, mat_cols].sum()); F_sea[n] = float(a2[:, sea_cols].sum())
    return F_mat, F_sea, 0.5 * Ex_mat, 0.5 * Ex_sea


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star", required=True)
    ap.add_argument("--member", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    D = json.load(open(a.star))
    mem_list = D["members"]
    if a.member is None:
        ok = [i for i, mm in enumerate(mem_list) if not mm.get("failed")]
        i_mem = max(ok, key=lambda i: mem_list[i]["E"])
    else:
        i_mem = a.member
    M = mem_list[i_mem]
    m, g, nsites = M["m"], M["g"], M["nsites"]
    half = bool(D["args"].get("half_offset", 0))
    R = np.arange(nsites, dtype=float) + (0.5 if half else 1.0)
    N = np.array(M["N"]); v = np.array(M["v"])
    OPS.GPU["dev"] = a.gpu
    t0 = time.time()
    NBt = float(M["N_B"])
    # pass 1: the eigenvalues per channel until two consecutive channels lie above the running E_top (engine_star_v5.channel_pass's bound)
    ep_all = []
    above = 0
    for kappa in range(1, 2000):
        h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half)
        e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
        ep = e[e > 0]
        ep_all.append((kappa, ep))
        lv = np.concatenate([x[1] for x in ep_all]); wv = np.concatenate([np.full(len(x[1]), 16.0 * x[0]) for x in ep_all])
        o = np.argsort(lv); cw = np.cumsum(wv[o])
        E_top_est = float(lv[o][min(int(np.searchsorted(cw, NBt - 1e-9)), len(lv) - 1)]) if cw[-1] >= NBt else np.inf
        above = above + 1 if ep.min() > E_top_est else 0
        if above >= 2:
            break
    lev = sorted([(float(x), 16.0 * k, ic, j) for ic, (k, ep) in enumerate(ep_all) for j, x in enumerate(ep)])
    occ_pos = [np.zeros(len(ep)) for _, ep in ep_all]
    left = NBt; i = 0; E_top = None
    while i < len(lev) and left > 1e-12:
        j = i
        while j < len(lev) and abs(lev[j][0] - lev[i][0]) < 1e-9:
            j += 1
        wsum = sum(x[1] for x in lev[i:j])
        frac = min(1.0, left / wsum)
        for x in lev[i:j]:
            occ_pos[x[2]][x[3]] = frac
        left -= frac * wsum
        E_top = lev[i][0]
        i = j
    EF_used = E_top
    print("filling rebuilt: %d channels, E_top %.8f (the record's %s)" % (len(ep_all), E_top, M.get("E_top")), flush=True)
    Var_mat = np.zeros(nsites); Var_sea = np.zeros(nsites); Ex_mat = 0.0; Ex_sea = 0.0; per = []
    NB = 0.0
    for (kappa, ep), fpos in zip(ep_all, occ_pos):
        h, Q, _ = OPS.chain_parts(nsites, m, float(kappa), half)
        e, U = OPS.spectrum(OPS.clocked(h, Q, N, v))
        fw = np.where(e < 0, 1.0, 0.0)
        fw[e > 0] = fpos
        is_sea = e < 0
        Fm, Fs, Xm, Xs = channel_fluct(h, U, fw, is_sea, R, g, a.gpu)
        w = 16.0 * kappa
        Var_mat += w * Fm; Var_sea += w * Fs; Ex_mat += w * Xm; Ex_sea += w * Xs
        NB += w * fpos.sum()
        per.append({"kappa": kappa, "Var_mat_at_end": float(w * Fm[-1]), "Var_sea_at_end": float(w * Fs[-1]), "Ex_mat": float(w * Xm), "Ex_sea": float(w * Xs)})
        print("kappa %d: Var_mat(end) %.4e Var_sea(end) %.4e Ex_mat %.4e Ex_sea %.4e (%.0f s)" % (kappa, w * Fm[-1], w * Fs[-1], w * Xm, w * Xs, time.time() - t0), flush=True)
    eps = np.array(M["eps_tot"])
    mprof = np.cumsum(eps)
    reads = {}
    for lab in ("R_surf", "R_999"):
        if M.get(lab):
            ic = min(int(np.searchsorted(R, float(M[lab]))), nsites - 1)
            reads[lab] = {"r": float(M[lab]), "fluct_mat_over_E": float(np.sqrt(max(Var_mat[ic], 0.0)) / M["E"]),
                          "fluct_sea_over_E": float(np.sqrt(max(Var_sea[ic], 0.0)) / M["E"])}
    out = {"script": "engine_massfluct_v2.py", "star": os.path.basename(a.star), "member": i_mem, "E_top_rebuilt": EF_used, "E_top_record": M.get("E_top"),
           "N_B": NB, "N_B_record": M["N_B"], "E": M["E"], "ell": M.get("ell"), "reads": reads, "one_over_sqrt_NB": float(1.0 / np.sqrt(NB)),
           "Ex0_matter": Ex_mat, "Ex0_sea": Ex_sea, "Ex0_total": Ex_mat + Ex_sea,
           "profile_r": R[::max(1, nsites // 256)].tolist(),
           "profile_fluct_mat_over_m": (np.sqrt(np.maximum(Var_mat, 0.0)) / np.where(mprof > 0, mprof, np.nan))[::max(1, nsites // 256)].tolist(),
           "profile_fluct_sea_over_m": (np.sqrt(np.maximum(Var_sea, 0.0)) / np.where(mprof > 0, mprof, np.nan))[::max(1, nsites // 256)].tolist(),
           "per_channel": per, "seconds": time.time() - t0}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "massfluct2_%s.json" % a.tag), "w"), indent=1, default=float)
    rs_ = reads.get("R_surf", {})
    print("member %d: N_B %.2f (record %s) E %.6f; at R_surf sqrt(Var_mat)/E %s against 1/sqrt(N_B) %.5f, sqrt(Var_sea)/E %s; E_x^(0) matter %.6e sea %.6e (%.0f s)" % (
        i_mem, NB, M["N_B"], M["E"], rs_.get("fluct_mat_over_E"), out["one_over_sqrt_NB"], rs_.get("fluct_sea_over_E"), Ex_mat, Ex_sea, time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
