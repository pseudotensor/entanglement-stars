#!/usr/bin/env python
"""ec_recompute.py (busi lane, cold check of EVOLVEDCUT_route_compare_v1): the matched-partition entropies recomputed from the
stored slices with own code (torch, complex128, CPU, 8 threads), independently of both routes' modules.

Per matched clock T (interior sites n_in: 12/58, 15/51, 20/38, 28/37, 35/37, 40/36) and channel kappa 1..10, from the slice
orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T*.npz (keys R, N_applied, v_applied, sea_k, dust_k, f_k, e_k; rows 2 n + alpha,
site n = 0..2047 at R_n/a = n + 1, both spinor components per site):
  C      the pure Gaussian state of the polar (= Loewdin) orthonormalization Q = A (A^dag A)^(-1/2) of the occupied columns
         A = [sea_k, dust_k] (occupations all 1), plus one Newton-Schulz refinement Q2 = Q (3 - Q^dag Q)/2 as a control on the
         orthonormality floor;
  P_-    the held Killing vacuum 1(h_kappa < 0), h_kappa = (1/2){N, h_0} + (1/2){v, Q_r}, h_0 = m sigma_3 + sigma_1 (T - T^dag)/(2i)
         + sigma_2 kappa/R, Q_r = (T^2 - T^dag 2)/(4i), open chain, m a = 0.2, built from the slice's N_applied, v_applied;
  P_0    the flat sea 1(h_0 < 0) at N = 1, v = 0.
For each state with orthonormal columns M: the complement (interior) block M_Xc M_Xc^dag (2 n_in x 2 n_in) and the exterior Gram
M_X^dag M_X (r x r), their occupation spectra and entropies, the exterior spectrum split into the 2 n_in genuine occupations and
the r - 2 n_in cluster at 1 (ideal value exactly 1 for a projector), n_pair = Tr[(1 - P_-) C], the applied root of N - |v|, and
the eigenvalues against the slice's stored e_k. Writes OUT/recompute.json (rows as they complete) and OUT/recompute.log."""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ec_common as C  # noqa: E402

torch.set_num_threads(8)
M_A = 0.2
MATCHED = [(12, 58), (15, 51), (20, 38), (28, 37), (35, 37), (40, 36)]
LOG = C.OUT + "/recompute.log"
OUTJ = C.OUT + "/recompute.json"


def log(s):
    with open(LOG, "a") as f:
        f.write(time.strftime("%H:%M:%S ") + s + "\n")
    print(s, flush=True)


def build_h(R, N, v, kappa, m=M_A):
    """h_kappa in the site-major layout (row 2 n + alpha), open chain; numpy complex128 then torch."""
    n = len(R)
    d = 2 * n
    H = np.zeros((d, d), dtype=np.complex128)
    i2 = np.arange(n)
    # diagonal blocks N_j (m sigma_3 + (kappa / R_j) sigma_2); sigma_2 = [[0, -i], [i, 0]]
    H[2 * i2, 2 * i2] = N * m
    H[2 * i2 + 1, 2 * i2 + 1] = -N * m
    H[2 * i2, 2 * i2 + 1] = -1j * N * kappa / R
    H[2 * i2 + 1, 2 * i2] = 1j * N * kappa / R
    # (1/2){N, sigma_1 (T - T^dag)/(2i)}: (j, j+1) block (N_j + N_{j+1})/(4i) sigma_1, sigma_1 = [[0, 1], [1, 0]]
    j = np.arange(n - 1)
    c1 = (N[j] + N[j + 1]) / (4j)
    H[2 * j, 2 * (j + 1) + 1] = c1
    H[2 * j + 1, 2 * (j + 1)] = c1
    H[2 * (j + 1) + 1, 2 * j] = np.conj(c1)
    H[2 * (j + 1), 2 * j + 1] = np.conj(c1)
    # (1/2){v, (T^2 - T^dag 2)/(4i) I_2}: (j, j+2) block (v_j + v_{j+2})/(8i) I_2
    j = np.arange(n - 2)
    c2 = (v[j] + v[j + 2]) / (8j)
    H[2 * j, 2 * (j + 2)] = c2
    H[2 * j + 1, 2 * (j + 2) + 1] = c2
    H[2 * (j + 2), 2 * j] = np.conj(c2)
    H[2 * (j + 2) + 1, 2 * j + 1] = np.conj(c2)
    Ht = torch.from_numpy(H)
    assert torch.allclose(Ht, Ht.conj().T)
    return Ht


def negative_projector_columns(H):
    e, U = torch.linalg.eigh(H)
    neg = e < 0
    res = (H @ U - U * e).norm().item()
    return e, U[:, neg], {"n_neg": int(neg.sum().item()), "nearest_zero": float(e.abs().min().item()), "eig_residual_frob": res,
                          "orth_defect_frob": (U.conj().T @ U - torch.eye(U.shape[1], dtype=U.dtype)).norm().item()}


def spectra_of_state(Mc, split):
    """Mc: orthonormal columns (d x r) torch complex128; split = 2 n_in. Returns the interior-block and exterior-Gram spectra."""
    Mi, Mx = Mc[:split], Mc[split:]
    B = Mi @ Mi.conj().T
    mu = torch.linalg.eigvalsh(B).cpu().numpy()
    G = Mx.conj().T @ Mx
    lam = torch.linalg.eigvalsh(G).cpu().numpy()
    return mu, lam


def state_entropies(Mc, split, near=1e-9):
    mu, lam = spectra_of_state(Mc, split)
    r = {"S_int": C.entropy(mu), "S_ext": C.entropy(lam), "dim_int": int(len(mu)), "dim_ext": int(len(lam)),
         "int_min": float(mu.min()), "int_max": float(mu.max()), "ext_min": float(lam.min()), "ext_max": float(lam.max())}
    r["ext_minus_int"] = r["S_ext"] - r["S_int"]
    lam_s = np.sort(lam)
    gen, clu = lam_s[:split], lam_s[split:]
    r["S_ext_genuine"] = C.entropy(gen)
    r["S_ext_cluster"] = C.entropy(clu)
    r["cluster_n"] = int(len(clu))
    r["cluster_max_abs_dev"] = float(np.abs(1.0 - clu).max()) if len(clu) else 0.0
    r["cluster_rms_dev"] = float(np.sqrt(np.mean((1.0 - clu) ** 2))) if len(clu) else 0.0
    r["cluster_n_below_1"] = int((clu < 1.0).sum())
    r["S_ext_genuine_minus_int"] = r["S_ext_genuine"] - r["S_int"]
    tot, s_near, s_rest, n_near, maxdev = C.entropy_split(mu, near)
    r["int_S_near_boundary"] = s_near
    r["int_n_near_boundary"] = n_near
    r["mu_sorted_smallest3"] = [float(x) for x in np.sort(mu)[:3]]
    r["mu_sorted_largest3"] = [float(x) for x in np.sort(mu)[-3:]]
    return r


def selftest():
    """known answers: (a) a 2-mode pure state with cross-cut weight p has S_X = S_Xc = h2(p); (b) a random isometry on 40 modes,
    cut 12 | 28: S_X = S_Xc to roundoff and the exterior cluster at 1 has entropy ~ 1e-14."""
    p = 0.3
    Mc = torch.tensor([[math.sqrt(1 - p)], [math.sqrt(p)]], dtype=torch.complex128)
    r = state_entropies(Mc, 1, near=1e-12)
    a = abs(r["S_int"] - C.h2(p)) + abs(r["S_ext"] - C.h2(p))
    g = torch.Generator().manual_seed(1)
    A = torch.randn(40, 17, dtype=torch.complex128, generator=g)
    Q, _ = torch.linalg.qr(A)
    r2 = state_entropies(Q, 12, near=1e-12)
    b = abs(r2["S_int"] - r2["S_ext"])
    log("selftest: 2-mode |S - h2(p)| %.1e; random isometry |S_X - S_Xc| %.1e, cluster entropy %.1e (n %d)" % (a, b, r2["S_ext_cluster"], r2["cluster_n"]))
    assert a < 1e-12 and b < 1e-11
    return {"two_mode_abs_err": a, "isometry_ext_minus_int": b, "isometry_cluster_S": r2["S_ext_cluster"]}


def applied_root(R, N, v):
    f = N - np.abs(v)
    roots = []
    for j in range(len(R) - 1):
        if f[j] < 0 and f[j + 1] > 0:
            roots.append(float(R[j] - f[j] * (R[j + 1] - R[j]) / (f[j + 1] - f[j])))
    return roots


def main():
    os.makedirs(C.OUT, exist_ok=True)
    out = {"inputs": {}, "selftest": selftest(), "threads": torch.get_num_threads(), "torch": torch.__version__, "clocks": {}, "rows": []}
    if os.path.exists(OUTJ):
        prev = C.jload(OUTJ)
        out["rows"] = prev.get("rows", [])
        out["clocks"] = prev.get("clocks", {})
        done = {(r["T"], r["kappa"]) for r in out["rows"]}
        log("resuming: %d rows done" % len(done))
    else:
        done = set()
    flat_cache = {}
    r1 = {T: C.jload(C.r1_path(T)) for T, _ in MATCHED}
    r2 = C.r2_rows_index(C.jload(C.R2_RESULTS))
    for T, n_in in MATCHED:
        split = 2 * n_in
        path = C.SLICE_FMT % T
        out["inputs"]["slice_T%d" % T] = [path, os.path.getsize(path)]
        t0 = time.time()
        z = np.load(path, allow_pickle=False)
        R = np.asarray(z["R"], dtype=float)
        N = np.asarray(z["N_applied"], dtype=float)
        v = np.asarray(z["v_applied"], dtype=float)
        rs = float(z["rs"])
        assert np.array_equal(R, np.arange(1, 2049)) and rs == 64.0 and float(z["T"]) / rs == T
        kap = np.asarray(z["kappas"]).ravel()
        wts = np.asarray(z["weights"]).ravel()
        roots = applied_root(R, N, v)
        rec = r1[T]
        ck = {"n_in": n_in, "dtype": str(z["dtype"].item()) if hasattr(z["dtype"], "item") else str(z["dtype"]), "applied_roots_mine": roots,
              "applied_outer_mine": roots[-1] if roots else None, "r1_applied_outer": rec["root"].get("applied_outer"), "r1_R_cut": rec["root"]["R_cut"],
              "r1_record_outer": rec["root"].get("record_outer"), "r1_two_gE": rec["record"].get("two_gE"), "r2_radius": r2[(T, 1)]["radius_over_a"],
              "r2_cut_label": r2[(T, 1)]["cut_label"], "n_in_r2_primary": int(r2[(T, 1)]["interior_site_count"]),
              "n_in_from_r1_R_cut": int(np.count_nonzero(R <= rec["root"]["R_cut"])), "n_in_from_my_applied_root": (int(np.count_nonzero(R <= roots[-1])) if roots else None),
              "min_N_minus_absv": float((N - np.abs(v)).min()), "N_at_cut": float(N[n_in - 1]), "v_at_cut": float(v[n_in - 1]), "load_seconds": time.time() - t0}
        out["clocks"][str(T)] = ck
        log("T%d: n_in %d, applied roots %s (route 2 %.6f, route 1 applied_outer %s, route 1 cut %.6f %s)" % (T, n_in, ["%.6f" % x for x in roots], ck["r2_radius"], ck["r1_applied_outer"], ck["r1_R_cut"], rec["root"]["source"]))
        for k in C.KAPPAS:
            if (T, k) in done:
                continue
            t1 = time.time()
            assert float(wts[np.flatnonzero(kap == k)[0]]) == 16.0 * k
            sea = np.asarray(z["sea_%d" % k])
            dust = np.asarray(z["dust_%d" % k])
            f = np.asarray(z["f_%d" % k], dtype=float).ravel()
            e_stored = np.sort(np.asarray(z["e_%d" % k], dtype=float).ravel())
            src_dtype = str(sea.dtype)
            A = torch.from_numpy(np.ascontiguousarray(np.column_stack((sea.astype(np.complex128), dust.astype(np.complex128)))))
            n_occ = A.shape[1]
            assert A.shape[0] == 4096 and dust.shape[1] == len(f) and np.all(f == 1.0)
            # polar / Loewdin factor
            G = A.conj().T @ A
            g, U = torch.linalg.eigh(G)
            gram_defect = (G - torch.eye(n_occ, dtype=G.dtype)).norm().item()
            Q = (A @ U) @ (U.conj().T / torch.sqrt(g)[:, None])
            QQ = Q.conj().T @ Q
            orth_Q = (QQ - torch.eye(n_occ, dtype=QQ.dtype)).norm().item()
            Q2 = Q @ ((3.0 * torch.eye(n_occ, dtype=QQ.dtype) - QQ) / 2.0)
            orth_Q2 = (Q2.conj().T @ Q2 - torch.eye(n_occ, dtype=QQ.dtype)).norm().item()
            # entropy of the whole-chain state (ideal 0): from the Gram spectra
            S_global_Q = C.entropy(torch.linalg.eigvalsh(QQ).cpu().numpy())
            rC = state_entropies(Q, split)
            rC2 = state_entropies(Q2, split)
            # held vacuum
            H = build_h(R, N, v, k)
            e, Um, hinfo = negative_projector_columns(H)
            e_mine = np.sort(e.cpu().numpy())
            hinfo["max_abs_eig_vs_stored_e_k"] = float(np.abs(e_mine - e_stored).max()) if len(e_stored) == len(e_mine) else None
            rH = state_entropies(Um, split)
            # flat sea (per kappa, cut-independent columns)
            if k not in flat_cache:
                H0 = build_h(R, np.ones_like(N), np.zeros_like(v), k)
                e0, U0, finfo = negative_projector_columns(H0)
                flat_cache[k] = (U0, finfo)
            U0, finfo = flat_cache[k]
            rF = state_entropies(U0, split)
            # n_pair = Tr[(1 - P_-) C] = n_occ - ||U_-^dag Q||_F^2
            n_pair = n_occ - (Um.conj().T @ Q).norm().item() ** 2
            alt = None
            if (T, k) == (12, 1):
                # the alternative (block) layout reading of the columns, as a discriminating control on the shared site-major convention
                perm = np.empty(4096, dtype=np.int64)
                for site in range(2048):
                    for comp in range(2):
                        perm[2 * site + comp] = comp * 2048 + site
                Qalt = Q[torch.from_numpy(perm)]
                alt = n_occ - (Um.conj().T @ Qalt).norm().item() ** 2
            row = {"T": T, "kappa": k, "n_in": n_in, "split": split, "n_occ": int(n_occ), "n_dust": int(dust.shape[1]), "source_dtype": src_dtype,
                   "gram_defect_frob": gram_defect, "orth_Q_frob": orth_Q, "orth_Q2_frob": orth_Q2, "S_global_Q": S_global_Q,
                   "C": rC, "C_refined": rC2, "held": rH, "flat": rF, "held_info": hinfo, "flat_info": finfo, "n_pair": n_pair, "n_pair_alt_layout": alt,
                   "excess_ext": rC["S_ext"] - rH["S_ext"], "excess_int": rC["S_int"] - rH["S_int"], "excess_int_refined": rC2["S_int"] - rH["S_int"],
                   "seconds": time.time() - t1}
            out["rows"].append(row)
            C.jdump(out, OUTJ)
            log("T%d k%d (%s, n_occ %d): C int %.12f ext %.12f (ext-int %+.2e; cluster %.2e, genuine-int %+.2e; orth %.1e, global S %.1e) | refined int %.12f ext %.12f (ext-int %+.2e; orth %.1e) | held int %.12f ext %.12f (ext-int %+.2e; n_neg %d, gap %.4f, eig vs stored %.1e) | flat int %.12f ext %.12f (ext-int %+.2e) | excess int %.12f ext %.12f | n_pair %.6f%s | %.0f s"
                % (T, k, src_dtype, n_occ, rC["S_int"], rC["S_ext"], rC["ext_minus_int"], rC["S_ext_cluster"], rC["S_ext_genuine_minus_int"], orth_Q, S_global_Q,
                   rC2["S_int"], rC2["S_ext"], rC2["ext_minus_int"], orth_Q2, rH["S_int"], rH["S_ext"], rH["ext_minus_int"], hinfo["n_neg"], hinfo["nearest_zero"], hinfo["max_abs_eig_vs_stored_e_k"],
                   rF["S_int"], rF["S_ext"], rF["ext_minus_int"], row["excess_int"], row["excess_ext"], n_pair, (" (alt layout %.3f)" % alt) if alt is not None else "", row["seconds"]))
        del z
    C.jdump(out, OUTJ)
    log("done: %d rows" % len(out["rows"]))


if __name__ == "__main__":
    main()
