#!/usr/bin/env python
"""page_hold_split_selftest_v1.py: manufactured checks of page_hold_split_v1's routines on small chains (at most 512 sites, dense everywhere),
without a record or a stored state. The chain is page_ops_v6's (chain_sparse, clocked_csr) with the declared exterior-plus-ball flow at a small
r_s so that R_cut = 24 r_s lies well inside the chain; the flow exceeds 1 inside the horizon, which traps an inner band (bound eigenvectors).
(1) D = 0 on the vacuum: P_-'s own columns give d = 0, E_inv = 0, ||D||_F = 0 (the Gram route). (2) E_inv equals the eigenbasis sum: on the
vacuum of another flow, Tr[h C] - Tr[h P_-] from the sparse h against sum e d; Tr D = n_occ - n_neg. (3) the weights are complete (sum = 2 sites
inside) and in [0, 1]; the split's parts sum to E_inv at every threshold, the counts to n2, the signed parts to each part. (4) a state built from
bound eigenvectors (the vacuum with the bound set flipped) gives an extended share 0 and a bound part sum |e| over the set. (5) the shift-invert
window reproduces the dense eigenvalues near the top and the bottom, its residuals, the window's invariant sums (the weights, the energy content),
and both estimate rules of the target routine. (6) the graft occupations d^g against the dense embedding E D_S E^dagger on a small long chain,
and their complete energy sum against Tr[h_L E D_S E^dagger]. (7) the exit scales on a uniform exterior: v_g = sqrt(1 - E^2) at the end, the
path time L/v_g, no propagation above the band, the exits flagged; the turning radii. (8) the band edges on the uniform chain. (9) the bins'
sums. (10) the envelope's ends. Prints "SELFTEST split v1 PASSED" or names each failure.
Usage: [--device cpu] [--nsites 384] [--kappa 1] [--chunk 64]."""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_ops_v6 as OPS            # noqa: E402
import page_hold_split_v1 as HS       # noqa: E402

CD = OPS.CD
RD = OPS.RD


def small_chain(nsites, kappa, rs, Rb, m=0.0, v_scale=1.0):
    """R_n = n + 1, N = 1, v = v_scale times the declared exterior-plus-ball flow (page_ops_v6.flow_dense); h = h[N, v] as a host CSR."""
    R = np.arange(nsites, dtype=float) + 1.0
    h0, Q = OPS.chain_sparse(R, m, float(kappa))
    v = v_scale * OPS.flow_dense(rs, Rb)(R)
    N = np.ones(nsites)
    return R, N, v, OPS.clocked_csr(h0, Q, N, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--nsites", type=int, default=384); ap.add_argument("--kappa", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    if a.nsites > 512:
        raise SystemExit("--nsites at most 512 (dense checks)")
    t_all = time.time()
    fails = []

    def check(name, ok, detail=""):
        print("  %s %s%s" % ("ok  " if ok else "FAIL", name, (": " + detail) if detail else ""), flush=True)
        if not ok:
            fails.append(name + ((": " + detail) if detail else ""))

    def run(name, fn):
        try:
            fn()
        except Exception as ex:
            check(name + " (exception)", False, "%s: %s" % (type(ex).__name__, ex))

    n = a.nsites; kappa = a.kappa; rs = 4.0; Rb = 1.0
    R_cut = 24.0 * rs; R_col = 16.0 * rs
    T_H = 1.0 / (4.0 * math.pi * rs)
    R, N, v, h = small_chain(n, kappa, rs, Rb)
    n2 = 2 * n
    nc = int(np.count_nonzero(R <= R_cut)); ncol = int(np.count_nonzero(R <= R_col))
    print("selftest: %d sites (n2 %d), kappa %d, r_s %g, R_cut %g (%d sites inside), collar %g (%d sites), max |v| %.3f, device %s" % (
        n, n2, kappa, rs, R_cut, nc, R_col, ncol, np.abs(v).max(), dev), flush=True)
    e_t, V, sec, peak = HS.dense_spectrum(h, dev)
    e = e_t.cpu().numpy().astype(float)
    neg = e < 0.0; nneg = int(neg.sum())
    bound_G = OPS.gershgorin(h)
    S = {}

    # (1) the vacuum
    def t1():
        Phi = V[:, :nneg].clone()
        n_a = HS.occupations(V, Phi, a.chunk).cpu().numpy()
        d = n_a - neg.astype(float)
        g = HS.gram_stats(Phi, a.chunk)
        D_F = math.sqrt(max(g["trace_C2"] - 2.0 * float(n_a[neg].sum()) + nneg, 0.0))
        check("T1 vacuum: max |d|", float(np.abs(d).max()) < 1e-12, "%.2e" % np.abs(d).max())
        check("T1 vacuum: E_inv", abs(float((e * d).sum())) < 1e-10 * float(np.abs(e).sum()), "%.2e" % (e * d).sum())
        check("T1 vacuum: Tr C = n_neg, Gram defect", abs(g["trace_C"] - nneg) < 1e-9 and g["gram_defect_F"] < 1e-10, "%.2e, %.2e" % (g["trace_C"] - nneg, g["gram_defect_F"]))
        check("T1 vacuum: ||D||_F", D_F < 1e-4, "%.2e" % D_F)
    run("T1", t1)

    # (2) E_inv equals the eigenbasis sum on a nontrivial state
    def t2():
        _, _, _, h2 = small_chain(n, kappa, rs, Rb, v_scale=0.7)
        e2_t, V2, _, _ = HS.dense_spectrum(h2, dev)
        nneg2 = int((e2_t < 0).sum())
        Phi2 = V2[:, :nneg2].clone(); del V2
        n_a = HS.occupations(V, Phi2, a.chunk).cpu().numpy()
        d2 = n_a - neg.astype(float)
        E_eig = float((e * d2).sum())
        P = Phi2.cpu().numpy()
        E_direct = float(np.sum(np.conj(P) * (h @ P)).real) - float(e[neg].sum())
        check("T2 E_inv: eigenbasis sum against Tr[hC] - Tr[hP_-]", abs(E_eig - E_direct) <= 1e-9 * (1.0 + abs(E_direct)), "%.10e vs %.10e" % (E_eig, E_direct))
        check("T2 Tr D = n_occ - n_neg", abs(float(d2.sum()) - (nneg2 - nneg)) < 1e-9, "%.2e" % (d2.sum() - (nneg2 - nneg)))
        check("T2 the state deviates (|E_inv| > 0)", abs(E_eig) > 1e-8, "%.3e" % E_eig)
        S["Phi2"] = Phi2; S["d2"] = d2; S["E_inv2"] = E_eig
    run("T2", t2)

    # (3) the weights and the split's parts
    def t3():
        w_cut = HS.weights_inside(V, 2 * nc, a.chunk).cpu().numpy(); w_col = HS.weights_inside(V, 2 * ncol, a.chunk).cpu().numpy()
        check("T3 weights complete: sum w_cut = 2 n_inside", abs(float(w_cut.sum()) - 2 * nc) < 1e-9, "%.2e" % (w_cut.sum() - 2 * nc))
        check("T3 weights in [0, 1]", bool((w_cut >= -1e-12).all() and (w_cut <= 1.0 + 1e-12).all() and (w_col <= w_cut + 1e-12).all()))
        S["w_cut"] = w_cut; S["w_col"] = w_col
        d2 = S["d2"]; E2 = S["E_inv2"]
        for th in (0.9, 0.99, 0.999):
            s_ = HS.split_energy(e, d2, w_cut, th)
            check("T3 parts sum to E_inv at theta %g" % th, abs(s_["E_bound"] + s_["E_extended"] - E2) <= 1e-12 * (1.0 + abs(E2)), "%.2e" % (s_["E_bound"] + s_["E_extended"] - E2))
            check("T3 counts sum to n2 at theta %g" % th, s_["n_bound"] + s_["n_extended"] == n2)
            check("T3 signed parts at theta %g" % th, abs(s_["E_bound_particles"] + s_["E_bound_holes"] - s_["E_bound"]) <= 1e-12 * (1.0 + abs(E2))
                  and abs(s_["E_extended_particles"] + s_["E_extended_holes"] - s_["E_extended"]) <= 1e-12 * (1.0 + abs(E2)))
    run("T3", t3)

    # (4) a state built from bound eigenvectors
    def t4():
        w_cut = S["w_cut"]
        B = w_cut >= 0.99
        check("T4 the manufactured chain has bound eigenvectors (w_cut >= 0.99)", int(B.sum()) >= 2, "%d of %d" % (B.sum(), n2))
        occ4 = neg ^ B
        idx = torch.as_tensor(np.nonzero(occ4)[0], dtype=torch.int64, device=dev)
        Phi4 = V[:, idx]
        n_a = HS.occupations(V, Phi4, a.chunk).cpu().numpy()
        d4 = n_a - neg.astype(float)
        s4 = HS.split_energy(e, d4, w_cut, 0.99)
        check("T4 extended share 0", abs(s4["E_extended"]) <= 1e-10 * (1.0 + float(np.abs(e).sum())), "%.2e" % s4["E_extended"])
        check("T4 bound part = sum_B |e|", abs(s4["E_bound"] - float(np.abs(e[B]).sum())) <= 1e-9 * (1.0 + float(np.abs(e[B]).sum())), "%.2e" % (s4["E_bound"] - np.abs(e[B]).sum()))
        check("T4 d = +-1 on the set, 0 elsewhere", float(np.abs(np.abs(d4[B]) - 1.0).max()) < 1e-12 and float(np.abs(d4[~B]).max()) < 1e-12)
    run("T4", t4)

    # (5) the shift-invert window against the dense spectrum
    def t5():
        A = h.tocsc()
        be = HS.band_edges_sites(N, v, R, kappa, 0.0, 2048)
        nev = 12
        vals, vecs, res, sec_, conv = HS.shift_invert_window(A, bound_G * (1.0 + 1e-6), nev, 1e-12, None, 4000)
        check("T5 window converged (%d pairs)" % nev, bool(conv) and vals.size == nev)
        check("T5 top eigenvalues against the dense ones", vals.size == nev and float(np.abs(vals - e[-nev:]).max()) < 1e-9, "%.2e" % (np.abs(vals - e[-nev:]).max() if vals.size == nev else float("nan")))
        check("T5 residuals", res.size and float(res.max()) < 1e-8, "%.2e" % (res.max() if res.size else float("nan")))
        wW = (np.abs(vecs[:2 * nc]) ** 2).sum(axis=0)
        check("T5 window weight sum invariant", abs(float(wW.sum()) - float(S["w_cut"][-nev:].sum())) < 1e-8, "%.2e" % (wW.sum() - S["w_cut"][-nev:].sum()))
        dgW, _, _ = HS.graft_occupations(vecs, n2, S["Phi2"], V[:, :nneg], a.chunk)
        ref = float((e[-nev:] * S["d2"][-nev:]).sum())
        check("T5 window energy content invariant", abs(float((vals * dgW).sum()) - ref) < 1e-8 * (1.0 + abs(ref)), "%.2e" % ((vals * dgW).sum() - ref))
        valsT, _, resT, info = HS.window_for_target(A, "top", bound_G, be, nev=nev, offset=1e-4, k0=4, tol=1e-12, tol_coarse=1e-6, ncv=None, maxiter=4000,
                                                     passes_max=3, res_tol=1e-8, first="symbol")
        check("T5 target top (symbol estimate) reproduces the dense top", valsT.size == nev and float(np.abs(valsT - e[-nev:]).max()) < 1e-9 and abs(info["extreme"] - e[-1]) < 1e-9,
              "%d pairs, %d passes" % (valsT.size, info.get("n_passes", 0)))
        valsG, _, _, infoG = HS.window_for_target(A, "top", bound_G, be, nev=nev, offset=1e-4, k0=4, tol=1e-12, tol_coarse=1e-6, ncv=None, maxiter=4000,
                                                   passes_max=3, res_tol=1e-8, first="gershgorin")
        check("T5 target top (Gershgorin estimate) reproduces the dense top", valsG.size == nev and float(np.abs(valsG - e[-nev:]).max()) < 1e-9, "%d passes" % infoG.get("n_passes", 0))
        valsB, _, _, infoB = HS.window_for_target(A, "bottom", bound_G, be, nev=nev, offset=1e-4, k0=4, tol=1e-12, tol_coarse=1e-6, ncv=None, maxiter=4000,
                                                   passes_max=3, res_tol=1e-8, first="symbol")
        check("T5 target bottom reproduces the dense bottom", valsB.size == nev and float(np.abs(valsB - e[:nev]).max()) < 1e-9 and abs(infoB["extreme"] - e[0]) < 1e-9)
        sig = float(be["end_top"])
        valsE, _, _, _ = HS.window_for_target(A, "ext_top", bound_G, be, nev=nev, offset=1e-4, k0=4, tol=1e-12, tol_coarse=1e-6, ncv=None, maxiter=4000,
                                               passes_max=3, res_tol=1e-8, first="symbol")
        near = np.sort(np.abs(e - sig))[:nev]
        check("T5 target ext_top returns the eigenvalues nearest the exterior top", valsE.size == nev and abs(float(np.sort(np.abs(valsE - sig))[-1]) - float(near[-1])) < 1e-9)
    run("T5", t5)

    # (6) the graft occupations against the dense embedding
    def t6():
        ns6 = 128; nl6 = 192
        Rl6, Nl6, vl6, hl6 = small_chain(nl6, kappa, rs, Rb)
        _, _, _, hs6 = small_chain(ns6, kappa, rs, Rb)
        es_t, Vs6, _, _ = HS.dense_spectrum(hs6, dev)
        nnegs = int((es_t < 0).sum())
        _, _, _, hs6b = small_chain(ns6, kappa, rs, Rb, v_scale=0.7)
        eb_t, Vb, _, _ = HS.dense_spectrum(hs6b, dev)
        Phi6 = Vb[:, :int((eb_t < 0).sum())].clone(); del Vb
        el_t, Vl6, _, _ = HS.dense_spectrum(hl6, dev)
        el = el_t.cpu().numpy().astype(float); Vl = Vl6.cpu().numpy()
        dg, dphi, dvac = HS.graft_occupations(Vl, 2 * ns6, Phi6, Vs6[:, :nnegs], a.chunk)
        Vn = Vs6[:, :nnegs]
        DS = (Phi6 @ Phi6.mH - Vn @ Vn.mH).cpu().numpy()
        Ws = Vl[:2 * ns6, :]
        dg_dense = np.einsum("ia,ij,ja->a", Ws.conj(), DS, Ws).real
        check("T6 graft occupations against the dense embedding", float(np.abs(dg - dg_dense).max()) < 1e-10, "%.2e" % np.abs(dg - dg_dense).max())
        hl_blk = hl6[: 2 * ns6, : 2 * ns6].toarray()
        Eg = float(np.sum(hl_blk.T * DS).real)
        check("T6 complete energy sum against Tr[h_L E D_S E^dagger]", abs(float((el * dg).sum()) - Eg) <= 1e-9 * (1.0 + abs(Eg)), "%.10e vs %.10e" % ((el * dg).sum(), Eg))
        check("T6 the Phi and vacuum parts are weights in [0, 1]", bool((dphi >= -1e-12).all() and (dphi <= 1 + 1e-12).all() and (dvac >= -1e-12).all() and (dvac <= 1 + 1e-12).all()))
    run("T6", t6)

    # (7) the exit scales on a uniform exterior
    def t7():
        nu = 400
        Ru = 1.0e6 + np.arange(nu, dtype=float) + 1.0; Nu = np.ones(nu); vu = np.zeros(nu)
        Eu = np.array([0.1, 0.3, 0.5, 0.9, -0.5, 1.5])
        ex = HS.exit_scales(Nu, vu, Ru, 1, 0.0, Eu, Ru[0], 50, 4096)
        vg = np.sqrt(1.0 - Eu[:5] ** 2)
        check("T7 v_g at the end = sqrt(1 - E^2) (slowest and fastest)", float(np.abs(ex["v_end_min"][:5] - vg).max()) < 1e-4 and float(np.abs(ex["v_end_max"][:5] - vg).max()) < 1e-4,
              "%.2e" % np.abs(ex["v_end_min"][:5] - vg).max())
        L = Ru[-1] - Ru[1]
        check("T7 path time = L / v_g", float(np.abs(ex["t_exit"][:5] - L / vg).max()) <= 1e-4 * float((L / vg).max()), "%.2e" % np.abs(ex["t_exit"][:5] - L / vg).max())
        check("T7 box time = (R_end - R_from) / v_g", float(np.abs(ex["t_box"][:5] - (Ru[-1] - Ru[0]) / vg).max()) <= 1e-4 * float(((Ru[-1] - Ru[0]) / vg).max()))
        check("T7 above the band: no exit, turning radius R_from", (not np.isfinite(ex["t_exit"][5])) and ex["R_turn"][5] == Ru[0] and (not bool(ex["exits"][5])))
        check("T7 propagating energies exit at the end", bool(ex["exits"][:5].all()) and bool((ex["R_turn"][:5] == Ru[-1]).all()))
        be_u = HS.band_edges_sites(Nu, vu, Ru, 1, 0.0, 4096)
        Rt = HS.turning_radii(Eu, be_u["top_site"], be_u["bottom_site"], Ru, Ru[0])
        check("T7 turning radii from the band edges", bool((Rt[:5] == Ru[-1]).all()) and Rt[5] == Ru[0])
        S["be_u"] = be_u
    run("T7", t7)

    # (8) the band edges on the uniform chain
    def t8():
        be_u = S["be_u"]
        check("T8 uniform band top 1, bottom -1", abs(be_u["global_top"] - 1.0) < 1e-9 and abs(be_u["global_bottom"] + 1.0) < 1e-9 and abs(be_u["end_top"] - 1.0) < 1e-9,
              "%.3e, %.3e" % (be_u["global_top"] - 1.0, be_u["global_bottom"] + 1.0))
        check("T8 uniform gap edges +-kappa/R", abs(float(be_u["upper_branch_min_site"][0]) - 1.0e-6) < 1e-9 and abs(float(be_u["lower_branch_max_site"][0]) + 1.0e-6) < 1e-9)
    run("T8", t8)

    # (9) the bins' sums
    def t9():
        d2 = S["d2"]; w_cut = S["w_cut"]
        edges = HS.bin_edges(0.5, 40.0, 10.0, max(bound_G, float(np.abs(e).max())) / T_H)
        b = HS.bin_arrays(e, d2, w_cut, 0.99, edges, T_H)
        tot = sum(float(b[k_].sum()) for k_ in b if k_.startswith("E_"))
        cnt = sum(float(b[k_].sum()) for k_ in b if k_.startswith("count_"))
        nz = int((e == 0.0).sum())
        check("T9 the bins' energy sums give E_inv", abs(tot - S["E_inv2"]) <= 1e-12 * (1.0 + abs(S["E_inv2"])), "%.2e" % (tot - S["E_inv2"]))
        check("T9 the bins' counts give n2 (minus exact zero modes)", int(round(cnt)) == n2 - nz, "%d vs %d" % (int(round(cnt)), n2 - nz))
        check("T9 the last edge exceeds every |e|/T_H", float(edges[-1]) > float(np.abs(e).max()) / T_H)
        S["bins"] = b; S["edges"] = edges
    run("T9", t9)

    # (10) the envelope's ends
    def t10():
        b = S["bins"]; edges = S["edges"]; nb = len(edges) - 1
        cen = b["centres_TH"] * T_H
        Es = np.concatenate([cen, -cen])
        ex = HS.exit_scales(N, v, R, kappa, 0.0, Es, R_cut, 8, 2048)
        tg = np.array([0.0, 10.0 * rs, 1.0e30])
        rem, trapped = HS.envelope_remaining(b["E_extended_pos"], b["E_extended_neg"], ex["t_exit"][:nb], ex["t_exit"][nb:], tg)
        E_ext = float(b["E_extended_pos"].sum() + b["E_extended_neg"].sum())
        check("T10 envelope at t = 0 is the whole extended part", abs(rem[0] - E_ext) <= 1e-12 * (1.0 + abs(E_ext)), "%.2e" % (rem[0] - E_ext))
        check("T10 envelope at t -> infinity is the trapped part", abs(rem[2] - trapped) <= 1e-12 * (1.0 + abs(trapped)))
        check("T10 envelope monotone", bool(rem[0] >= rem[1] - 1e-12 and rem[1] >= rem[2] - 1e-12))
    run("T10", t10)

    print("selftest %.1f s" % (time.time() - t_all), flush=True)
    if fails:
        print("SELFTEST split v1 FAILED (%d):" % len(fails), flush=True)
        for f in fails:
            print("  - " + f, flush=True)
        sys.exit(1)
    print("SELFTEST split v1 PASSED", flush=True)


if __name__ == "__main__":
    main()
