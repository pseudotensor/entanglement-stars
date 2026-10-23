#!/usr/bin/env python
"""packet_v1_bounds.py (busi lane, reviewer A slot, inbox #354): the two bounds of ReviewA_qsing_v1 sec. (2)(a) and (2)(b) checked numerically on every
stored record of the reference collapse.

(1) The per-site source bound.  The stored eps_tot[k, n] and J_tot[k, n] are the shell totals of the covariance traces
    sum_kappa w_kappa Re tr[Pi_n h0_kappa (C - P)] and sum_kappa w_kappa Re tr[Pi_n Q_kappa (C - P)] (engine_collapse_v19 sources(): dens(h, sea)
    - dens(h, ref) + dens(h, dust)), so rho_n = eps_tot/(4 pi R_n^2) and J_n = J_tot/(4 pi R_n^2) are the volume densities at radius R_n (a = 1).
    Checked for every record and site, with W = v_applied[k, n] (the shift that enters h_m and the charge row U' = 8 pi g r^2 (rho + q J)):
      cubic constant (reviewer A (2)(a), the paper's eq. sc-source-cap):  |rho_n +- W J_n| <= 4 sqrt(m^2 + 3/2 + W^2/8);
      the chain's own constant (the same Hoelder argument on the chain, |tr Pi_n A D| <= ||Pi_n A||_1 ||D||, ||D|| <= 1):
          |rho_n +- W J_n| <= B_chain,n(W) = sum_kappa w_kappa ||Pi_n (h0_kappa + W Q_kappa)||_1 / (4 pi R_n^2),
      ||Pi_n A||_1 the trace norm of the two rows of A at site n = sum of the square roots of the eigenvalues of the 2 x 2 Gram Pi_n A A^dag Pi_n.
    Also the kernel-averaged densities rho_ell(r) = sum_n K_0(r, R_n) eps_n of engine_fields_v12 (the row's sources) against the cubic constant.
(2) The norm bound on the clocked Hamiltonian.  Per record and channel ||h_m||_2 (ARPACK, largest magnitude) against
      max_n N_applied ||h0_kappa|| + max_n |v_applied| ||Q||   (||(1/2){N, A}|| <= ||N|| ||A||),
    and against the stored spectra e_kappa (collapse_*_spectra.npz, the engine's own eigenvalues at the accepted fields) where they exist.

Usage: python packet_v1_bounds.py --fields F_fields.npz --json F.json --spectra F_spectra.npz --out DIR [--engine-code DIR]
"""
import argparse
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import eigsh  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packet_v1_common as PC  # noqa: E402


def site_blocks(A, B, n):
    """the 2 x 2 site-diagonal blocks of M = A B^dag (sparse), G[i] = M[2i:2i+2, 2i:2i+2], read off the three diagonals of M."""
    M = (A @ B.conj().T).tocsr()
    d0 = M.diagonal(0); d1 = M.diagonal(1); dm1 = M.diagonal(-1)
    G = np.zeros((n, 2, 2), complex)
    G[:, 0, 0] = d0[0::2]; G[:, 1, 1] = d0[1::2]; G[:, 0, 1] = d1[0::2]; G[:, 1, 0] = dm1[0::2]
    return G


def site_blocks_direct(A, B, n, every=97):
    """the same blocks by explicit row extraction on every `every`-th site (the slow, unambiguous route; used to check site_blocks)."""
    M = (A @ B.conj().T).tocsr()
    idx = np.arange(0, n, every)
    G = np.zeros((len(idx), 2, 2), complex)
    for j, i in enumerate(idx):
        G[j] = M[2 * i:2 * i + 2, 2 * i:2 * i + 2].toarray()
    return idx, G


def trace_norm_2x2(G):
    """sum of the square roots of the eigenvalues of the Hermitian psd 2 x 2 blocks G (..., 2, 2)."""
    p = np.real(G[..., 0, 0]); s = np.real(G[..., 1, 1]); q = G[..., 0, 1]
    half = 0.5 * (p + s); disc = np.sqrt(np.maximum((0.5 * (p - s)) ** 2 + np.abs(q) ** 2, 0.0))
    l1 = np.maximum(half + disc, 0.0); l2 = np.maximum(half - disc, 0.0)
    return np.sqrt(l1) + np.sqrt(l2)


def kernel_K0(R, ell, rgrid):
    """K_0(r, R_n) of the engine (engine_fields_v12.kernels = engine_fields_v4.kernels, when importable), (len(rgrid), n); else None.
    rho_ell(r) = sum_n K_0(r, R_n) eps_n is the row's energy density (int 4 pi r^2 K_0(r, R_n) dr = 1)."""
    try:
        import engine_fields_v12 as F12
    except Exception as ex:  # noqa: BLE001
        print("engine kernels not importable:", repr(ex), flush=True)
        return None
    K0, _K1 = F12.kernels(np.asarray(rgrid, float), np.asarray(R, float), float(ell))
    return np.asarray(K0, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True); ap.add_argument("--json", required=True); ap.add_argument("--spectra", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--engine-code", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    Z = np.load(a.fields, allow_pickle=True)
    R = np.asarray(Z["R"], float); T = np.asarray(Z["T"], float); Napp = np.asarray(Z["N_applied"], float); vapp = np.asarray(Z["v_applied"], float)
    Nst = np.asarray(Z["N"], float); vst = np.asarray(Z["v"], float)
    eps = np.asarray(Z["eps_tot"], float); Jt = np.asarray(Z["J_tot"], float); ell = float(Z["ell"]); n = len(R)
    J = json.load(open(a.json)); m = float(J["args"]["m"]); g = float(J["args"]["g"]); rs = float(J["args"]["rs"]); kappa_sea = int(J["args"]["kappa_sea"])
    recs = J["records"]
    meta = {"fields": a.fields, "fields_md5": PC.md5(a.fields), "json": a.json, "json_md5": PC.md5(a.json), "spectra": a.spectra, "spectra_md5": (PC.md5(a.spectra) if a.spectra else None),
            "script_md5": PC.md5(os.path.abspath(__file__)), "common_md5": PC.md5(os.path.join(os.path.dirname(os.path.abspath(__file__)), "packet_v1_common.py")),
            "m": m, "g": g, "rs": rs, "ell": ell, "nsites": n, "nrecords": len(T), "kappa_sea": kappa_sea}
    if a.engine_code:
        sys.path.insert(0, a.engine_code)
    kappas = list(range(1, kappa_sea + 1))
    meta["engine_check"] = PC.check_against_engine(a.engine_code, R, m, kappas)
    print("engine check:", json.dumps(meta["engine_check"]), flush=True)
    vol = 4.0 * np.pi * R ** 2
    # (1) the per-site source bound
    rho = eps / vol[None, :]; Jd = Jt / vol[None, :]
    W = vapp
    cubic = 4.0 * np.sqrt(m * m + 1.5 + W ** 2 / 8.0)
    lhs_plus = np.abs(rho + W * Jd); lhs_minus = np.abs(rho - W * Jd)
    # the chain's own constant: Gram blocks G(W) = G_hh + W (G_hq + G_hq^dag) + W^2 G_qq per channel and site
    Ghh = []; Ghq = []; Gqq = []; norm_h0 = {}; norm_Q = None; block_check = 0.0
    for k in kappas:
        h0, Q = PC.chain(R, m, k)
        Ghh.append(site_blocks(h0, h0, n)); Ghq.append(site_blocks(h0, Q, n)); Gqq.append(site_blocks(Q, Q, n))
        for (A_, B_, G_) in ((h0, h0, Ghh[-1]), (h0, Q, Ghq[-1]), (Q, Q, Gqq[-1])):
            idx, Gd = site_blocks_direct(A_, B_, n)
            block_check = max(block_check, float(np.max(np.abs(G_[idx] - Gd))))
        norm_h0[str(k)] = float(abs(eigsh(h0, k=1, which="LM", return_eigenvectors=False)[0]))
        if norm_Q is None:
            norm_Q = float(abs(eigsh(Q, k=1, which="LM", return_eigenvectors=False)[0]))
    meta["site_blocks_check_max_abs_diff"] = block_check
    print("chain blocks built (%.0f s; diagonal route against direct extraction %.1e); ||h0|| per channel %s; ||Q|| %.6f" % (time.time() - t0, block_check, json.dumps(norm_h0), norm_Q), flush=True)
    # the chain constant at W = 0 and its record-wise value at W = v_applied (per record: n sites); the 2 x 2 algebra vectorized over records
    Bchain = np.zeros_like(rho)
    for ik, k in enumerate(kappas):
        G0 = Ghh[ik][None, :, :, :]; G1 = (Ghq[ik] + np.conj(np.transpose(Ghq[ik], (0, 2, 1))))[None, :, :, :]; G2 = Gqq[ik][None, :, :, :]
        Wf = W[:, :, None, None]
        Bchain += 16.0 * k * trace_norm_2x2(G0 + Wf * G1 + Wf ** 2 * G2)
    Bchain /= vol[None, :]
    Bchain0 = np.zeros(n)
    for ik, k in enumerate(kappas):
        Bchain0 += 16.0 * k * trace_norm_2x2(Ghh[ik])
    Bchain0 /= vol
    # where the cubic bound is the tighter one (the chain carries only kappa_sea channels; beyond some radius its own constant falls below the cubic one)
    ratio_cubic_plus = lhs_plus / cubic; ratio_cubic_minus = lhs_minus / cubic
    ratio_chain_plus = lhs_plus / Bchain; ratio_chain_minus = lhs_minus / Bchain
    worst = lambda X: (float(np.max(X)), [int(i) for i in np.unravel_index(int(np.argmax(X)), X.shape)])
    res1 = {"cubic_constant_at_W0": float(4.0 * np.sqrt(m * m + 1.5)), "note": "ratios are max over sites and records; (record, site index) at the max; W = v_applied",
            "max_ratio_cubic_plus": worst(ratio_cubic_plus), "max_ratio_cubic_minus": worst(ratio_cubic_minus),
            "max_ratio_chain_plus": worst(ratio_chain_plus), "max_ratio_chain_minus": worst(ratio_chain_minus),
            "max_abs_rho": worst(np.abs(rho)), "max_abs_WJ": worst(np.abs(W * Jd)), "max_abs_rho_plus_WJ": worst(lhs_plus),
            "max_ratio_cubic_plus_sites_ge_4": worst(ratio_cubic_plus[:, R >= 4.0]), "max_ratio_cubic_plus_sites_ge_12": worst(ratio_cubic_plus[:, R >= 12.0]),
            "chain_constant_over_cubic_at_W0": {"R=1": float(Bchain0[0] / (4.0 * np.sqrt(m * m + 1.5))), "R=4": float(Bchain0[3] / (4.0 * np.sqrt(m * m + 1.5))),
                                                "R=12": float(Bchain0[11] / (4.0 * np.sqrt(m * m + 1.5))), "R=24": float(Bchain0[23] / (4.0 * np.sqrt(m * m + 1.5))),
                                                "R=64": float(Bchain0[63] / (4.0 * np.sqrt(m * m + 1.5))), "crossing_radius": (float(R[np.argmax(Bchain0 < 4.0 * np.sqrt(m * m + 1.5))]) if np.any(Bchain0 < 4.0 * np.sqrt(m * m + 1.5)) else None)},
            "per_record_max_ratio_cubic_plus": [float(x) for x in ratio_cubic_plus.max(axis=1)],
            "per_record_max_ratio_chain_plus": [float(x) for x in ratio_chain_plus.max(axis=1)],
            "per_record_max_abs_rho_a4": [float(x) for x in np.abs(rho).max(axis=1)]}
    # the formation record and the peak record in T/rs
    for k in (97, 106, 320):
        if k < len(T):
            res1["record_%d" % k] = {"T_over_rs": float(T[k] / rs), "max_ratio_cubic_plus": worst(ratio_cubic_plus[k:k + 1]), "max_abs_rho_a4": float(np.abs(rho[k]).max()),
                                     "site_of_max_abs_rho": float(R[int(np.argmax(np.abs(rho[k])))])}
    # the kernel-averaged densities against the cubic constant
    rg = np.linspace(0.0, 256.0, 1025)
    K0 = kernel_K0(R, ell, rg)
    rl = None
    if K0 is not None:
        rl = eps @ K0.T
        res1["K0_weight_sum_check"] = [float(x) for x in (4.0 * np.pi * np.trapezoid(rg[:, None] ** 2 * K0[:, [11, 63, 255]], rg, axis=0))]
        res1["rho_ell_max_a4"] = worst(np.abs(rl)); res1["rho_ell_max_over_cubic_W0"] = float(np.abs(rl).max() / (4.0 * np.sqrt(m * m + 1.5)))
        res1["rho_ell_centre_max_a4"] = float(np.abs(rl[:, 0]).max())
        res1["rho_ell_centre_record97_a4"] = float(rl[97, 0]) if len(rl) > 97 else None
    print("source bound: max |rho + W J| / cubic %s; / chain %s; max |rho| a^4 %s" % (res1["max_ratio_cubic_plus"], res1["max_ratio_chain_plus"], res1["max_abs_rho"]), flush=True)
    # (2) ||h_m|| per record and channel against the bound and the stored spectra
    S = np.load(a.spectra, allow_pickle=True) if a.spectra else None
    Ts = np.asarray(S["T"], float) if S is not None else None
    normHm = np.zeros((len(T), len(kappas))); bound = np.zeros_like(normHm); stored = np.full_like(normHm, np.nan)
    chains = {k: PC.chain(R, m, k) for k in kappas}
    t1 = time.time()
    for k_ in range(len(T)):
        for ik, k in enumerate(kappas):
            h0, Q = chains[k]
            Hm = PC.clocked(h0, Q, Napp[k_], vapp[k_])
            normHm[k_, ik] = float(abs(eigsh(Hm, k=1, which="LM", return_eigenvectors=False, tol=1e-10)[0]))
            bound[k_, ik] = float(Napp[k_].max()) * norm_h0[str(k)] + float(np.abs(vapp[k_]).max()) * norm_Q
            if S is not None:
                j = np.nonzero(np.abs(Ts - T[k_]) < 1e-9)[0]
                if len(j):
                    stored[k_, ik] = float(np.abs(np.asarray(S["e_%d" % k])[j[0]]).max())
        if k_ % 40 == 0:
            print("  record %d: ||h_m|| %s bound %s stored %s (%.0f s)" % (k_, np.array2string(normHm[k_], precision=4), np.array2string(bound[k_], precision=4),
                                                                      np.array2string(stored[k_], precision=4), time.time() - t1), flush=True)
    res2 = {"norm_h0": norm_h0, "norm_Q": norm_Q, "band_top_free_radial": float(np.sqrt(m * m + 1.0)),
            "max_normHm_over_bound": float(np.max(normHm / bound)), "min_normHm_over_bound": float(np.min(normHm / bound)),
            "max_normHm": worst(normHm), "max_bound": worst(bound),
            "stored_spectra_records": int(np.sum(~np.isnan(stored[:, 0]))),
            "max_rel_diff_normHm_vs_stored": float(np.nanmax(np.abs(normHm - stored) / stored)),
            "max_N_applied_over_records": float(Napp.max()), "max_abs_v_applied_over_records": float(np.abs(vapp).max()),
            "per_record_max_normHm": [float(x) for x in normHm.max(axis=1)], "per_record_max_bound": [float(x) for x in bound.max(axis=1)],
            "per_record_Nmax": [float(x) for x in Napp.max(axis=1)], "per_record_absvmax": [float(x) for x in np.abs(vapp).max(axis=1)]}
    print("norm bound: ||h_m||/bound in [%.4f, %.4f]; max rel diff against the stored spectra %.2e" % (res2["min_normHm_over_bound"], res2["max_normHm_over_bound"], res2["max_rel_diff_normHm_vs_stored"]), flush=True)
    np.savez_compressed(os.path.join(a.out, "packet_v1_bounds.npz"), T=T, T_over_rs=T / rs, R=R, ratio_cubic_plus=ratio_cubic_plus, ratio_cubic_minus=ratio_cubic_minus,
                        ratio_chain_plus=ratio_chain_plus, ratio_chain_minus=ratio_chain_minus, rho=rho, Jd=Jd, W=W, Bchain=Bchain, Bchain0=Bchain0, cubic=cubic,
                        normHm=normHm, bound=bound, stored=stored, kappas=np.array(kappas), rho_ell=(rl if rl is not None else np.zeros(0)), rgrid=rg)
    out = {"meta": meta, "source_bound": res1, "norm_bound": res2, "seconds": time.time() - t0}
    with open(os.path.join(a.out, "packet_v1_bounds.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
