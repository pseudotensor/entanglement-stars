#!/usr/bin/env python
"""sym_test_v1.py (busi lane, the owner's code-wide symmetry audit, inbox #359/#360; reviewer A max).

Numerical checks, on the box only (nice 10, 8 threads, one free A6000), of the exact symmetries of the clocked Dirac chain
h_m[N, v] = (1/2){N, h_0} + (1/2){v, Q} that every production engine assembles (engine_vacflow_v1.chain_sparse / page_ops_v6.chain_sparse:
h_0 = on-site m sigma_z + (kappa/R) sigma_y, sine bond -(i/2) sigma_x, two-step Q = -(i/4) 1 from site n to n + 2), and the timings of the
cheaper operations those symmetries allow.

  A  (the heavy, massive chain, m > 0): h_m = D + i A with D real diagonal and A real antisymmetric, so the antiunitary
     C = S K (S = sum_n (-1)^n sigma_y, K complex conjugation; C^2 = -1) anticommutes with h_m: the spectrum is symmetric (+-E) and the
     eigenvectors at -E are S conj(u_E) up to a phase.  Checked on stored fields.  No real form exists (the shift term breaks time reversal).
  B  (the massless chain, m = 0: every light channel of twofield, Page and B1): S commutes with h_m, the two sectors S = +-1 decouple
     exactly, the sector blocks are n x n with H_- = -conj(H_+) (the "conjugate mirror"): one n x n eigendecomposition gives the whole
     2n x 2n spectrum and every eigenvector.  Checked by construction against the full eigendecomposition.
  C  the banded Cholesky certificate of the twofield light walk (K(s)^2 - d^2 positive definite) on the full 2n x 2n K against the
     sector block K_+ (n x n, half the bandwidth): the same verdicts at every d of the ladder, and the timings.
  T  timings on the box GPU: complex Hermitian eigh at 2n and n (the sector halving), real symmetric for reference.

Outputs a json and a text summary under --out.  No production file is read for writing; nothing on any card.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

sys.path.insert(0, "<internal>/engine")
sys.path.insert(0, "<internal>/code_v6")
import engine_vacflow_v1 as VF          # noqa: E402
import page_ops_v6 as PO6               # noqa: E402
import torch                            # noqa: E402


def S_csr(n):
    """S = sum_n (-1)^n sigma_y as a host CSR (engine_twofield light_S_v36's S)."""
    sgn = (1.0 - 2.0 * (np.arange(n) % 2)).astype(float)
    ev = 2 * np.arange(n)
    return sp.csr_matrix((np.concatenate([-1j * sgn, 1j * sgn]), (np.concatenate([ev, ev + 1]), np.concatenate([ev + 1, ev]))), shape=(2 * n, 2 * n))


def sector_basis(n):
    """P_+ and P_- (2n x n, site-local): column i is site i times the +1 (resp. -1) eigenvector of (-1)^i sigma_y, (1, +-(-1)^i i)/sqrt 2."""
    sgn = (1.0 - 2.0 * (np.arange(n) % 2))
    r = 1.0 / np.sqrt(2.0)
    rows = np.concatenate([2 * np.arange(n), 2 * np.arange(n) + 1]); cols = np.concatenate([np.arange(n), np.arange(n)])
    Pp = sp.csr_matrix((np.concatenate([np.full(n, r, complex), 1j * r * sgn]), (rows, cols)), shape=(2 * n, n))
    Pm = sp.csr_matrix((np.concatenate([np.full(n, r, complex), -1j * r * sgn]), (rows, cols)), shape=(2 * n, n))
    return Pp, Pm


def sector_block_direct(N, v, R, kappa):
    """H_+ built from the formulas (massless): diag (-1)^i N_i kappa/R_i, bond (i, i+1) (-1)^(i+1) (N_i + N_{i+1})/4, two-step (i, i+2)
    -(i/4)(v_i + v_{i+2})/2 and the conjugates: real symmetric tridiagonal plus an imaginary antisymmetric two-step part."""
    n = len(R); sgn = (1.0 - 2.0 * (np.arange(n) % 2))
    d0 = sgn * N * kappa / R
    b = -sgn[:-1] * 0.5 * (N[:-1] + N[1:]) * 0.5
    q = -0.25j * 0.5 * (v[:-2] + v[2:])
    i = np.arange(n)
    rows = np.concatenate([i, i[:-1], i[1:], i[:-2], i[2:]])
    cols = np.concatenate([i, i[1:], i[:-1], i[2:], i[:-2]])
    vals = np.concatenate([d0.astype(complex), b, b, q, np.conj(q)])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def band_upper(A):
    """engine_twofield band_upper_v36 (the scatter form)."""
    co = A.tocoo()
    off = co.col - co.row
    u = int(off.max()) if off.size else 0
    n = A.shape[0]
    ab = np.zeros((u + 1, n), dtype=complex)
    sel = off >= 0
    np.add.at(ab, (u - off[sel], co.col[sel]), co.data[sel])
    ab[u, :] = ab[u, :].real
    return ab, u


def pd_banded(ab, d):
    ab2 = ab.copy(); ab2[-1, :] -= d * d
    try:
        sla.cholesky_banded(ab2, lower=False, check_finite=False)
        return True
    except (np.linalg.LinAlgError, ValueError):
        return False


def dense_dev(A, dev):
    co = A.tocoo()
    M = torch.zeros(co.shape, dtype=torch.complex128, device=dev)
    M.index_put_((torch.as_tensor(co.row, dtype=torch.int64, device=dev), torch.as_tensor(co.col, dtype=torch.int64, device=dev)),
                 torch.as_tensor(np.asarray(co.data, complex), dtype=torch.complex128, device=dev), accumulate=True)
    return M


def eigh_t(M, reps=3):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize() if M.is_cuda else None
        t0 = time.time()
        e, U = torch.linalg.eigh(M)
        torch.cuda.synchronize() if M.is_cuda else None
        ts.append(time.time() - t0)
    return e, U, float(np.median(ts)), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", required=True, help="a stored collapse *_fields.npz (N, v per record)")
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--kappas", default="1,5,10")
    ap.add_argument("--rec", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--timing-sizes", default="2048,4096,8192")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda:%d" % a.gpu) if torch.cuda.is_available() else torch.device("cpu")
    torch.set_num_threads(8)
    z = np.load(a.fields, allow_pickle=True)
    N = np.asarray(z["N"][a.rec], float); v = np.asarray(z["v"][a.rec], float)
    N_prev = np.asarray(z["N"][a.rec - 1], float); v_prev = np.asarray(z["v"][a.rec - 1], float)
    n = len(N); R = np.arange(n, dtype=float) + 1.0
    out = {"fields": a.fields, "record_index": a.rec, "nsites": n, "m": a.m, "device": str(dev), "gpu_name": (torch.cuda.get_device_name(dev) if dev.type == "cuda" else None),
           "torch": torch.__version__, "A": {}, "B": {}, "C": {}, "T": {}}
    S = S_csr(n); Pp, Pm = sector_basis(n)
    kappas = [int(x) for x in a.kappas.split(",")]
    lines = []

    def say(s):
        print(s, flush=True); lines.append(s)

    say("device %s %s torch %s; nsites %d; record %d of %d; m %g; kappas %s" % (dev, out["gpu_name"], torch.__version__, n, a.rec, len(z["N"]), a.m, kappas))
    # ---------------------------------------------------------------- A: the massive chain's particle-hole symmetry
    for kappa in kappas:
        h0, Q = VF.chain_sparse(R, a.m, float(kappa))
        h = PO6.clocked_csr(h0, Q, N, v)
        # structure: diagonal real, off-diagonal purely imaginary
        co = h.tocoo(); offd = co.row != co.col
        A_struct = {"max_abs_imag_diag": float(np.max(np.abs(co.data[~offd].imag))) if (~offd).any() else 0.0,
                    "max_abs_real_offdiag": float(np.max(np.abs(co.data[offd].real))), "nnz": int(h.nnz), "bandwidth_u": int(np.max(co.col - co.row))}
        # anticommutation with C = S K: S conj(h) S = -h  <=>  S h^* S + h = 0 (S real-symmetric? S is Hermitian with S^2 = 1)
        anti = (S @ h.conj() @ S + h)
        A_struct["norm_S_conj_h_S_plus_h"] = float(abs(anti).max()) if anti.nnz else 0.0
        M = dense_dev(h, dev)
        e, U, t_eigh, ts = eigh_t(M, reps=2)
        e_np = e.real.cpu().numpy()
        sym = float(np.max(np.abs(e_np + e_np[::-1])))
        pos = e_np > 0; neg = e_np < 0
        Up = U[:, torch.as_tensor(pos, device=dev)]; Un = U[:, torch.as_tensor(neg, device=dev)]
        Sd = dense_dev(S, dev)
        V = Sd @ Up.conj()                                   # candidates for the -E eigenvectors
        res = (M @ V + V * torch.as_tensor(e_np[pos], device=dev)[None, :]).abs().max().item()     # h V = -E V
        Ov = (Un.conj().T @ V).abs()                           # should be a (near) permutation matrix in modulus
        col_max = Ov.max(dim=0).values.min().item(); row_max = Ov.max(dim=1).values.min().item()
        A_struct.update({"n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "n_zero": int((e_np == 0).sum()), "spectrum_symmetry_max_abs_e_plus_reversed_e": sym,
                         "residual_h_Sconj_u_plus_E": res, "min_over_columns_of_max_overlap_modulus": col_max, "min_over_rows_of_max_overlap_modulus": row_max,
                         "eigh_seconds_2n": t_eigh, "eigh_seconds_list": ts, "e_min": float(e_np.min()), "e_max": float(e_np.max()),
                         "gap_at_zero": float(e_np[pos].min() - e_np[neg].max())})
        out["A"][str(kappa)] = A_struct
        say("A kappa %d (m %g): diag imag %.1e, offdiag real %.1e, u %d, ||S h* S + h|| %.1e; eigh(%d) %.3f s; +-E symmetry %.2e; residual of S conj(u_+) as -E eigenvectors %.2e; overlap moduli min(col max) %.12f min(row max) %.12f; n_pos %d n_neg %d" % (
            kappa, a.m, A_struct["max_abs_imag_diag"], A_struct["max_abs_real_offdiag"], A_struct["bandwidth_u"], A_struct["norm_S_conj_h_S_plus_h"], 2 * n, t_eigh, sym, res, col_max, row_max, A_struct["n_pos"], A_struct["n_neg"]))
        del M, U, Up, Un, V, Ov, Sd
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    # ---------------------------------------------------------------- B: the massless chain's sector decoupling
    for kappa in kappas:
        h0, Q = VF.chain_sparse(R, 0.0, float(kappa))
        h = PO6.clocked_csr(h0, Q, N, v)
        comm = (S @ h - h @ S)
        B = {"norm_commutator_S_h": float(abs(comm).max()) if comm.nnz else 0.0}
        off = (Pm.conj().T @ h @ Pp)
        B["max_abs_offblock_Pm_h_Pp"] = float(abs(off).max()) if off.nnz else 0.0
        Hp = (Pp.conj().T @ h @ Pp).tocsr(); Hm = (Pm.conj().T @ h @ Pm).tocsr()
        Hp_direct = sector_block_direct(N, v, R, float(kappa))
        B["max_abs_Hp_product_minus_direct"] = float(abs(Hp - Hp_direct).max()) if (Hp - Hp_direct).nnz else 0.0
        mir = (Hm + Hp.conj())
        B["max_abs_Hm_plus_conj_Hp"] = float(abs(mir).max()) if mir.nnz else 0.0
        B["Hp_nnz"] = int(Hp_direct.nnz); B["h_nnz"] = int(h.nnz); B["Hp_bandwidth_u"] = int(np.max(Hp_direct.tocoo().col - Hp_direct.tocoo().row))
        M = dense_dev(h, dev); e, U, t_full, _ = eigh_t(M, reps=2); e_np = e.real.cpu().numpy()
        Mp = dense_dev(Hp_direct, dev); ep, Upl, t_half, _ = eigh_t(Mp, reps=2); ep_np = ep.real.cpu().numpy()
        both = np.sort(np.concatenate([ep_np, -ep_np]))
        B["max_abs_spectrum_diff_full_vs_sector_union"] = float(np.max(np.abs(both - e_np)))
        # the full eigenvectors' sector purity: sigma = <u|S|u>
        Sd = dense_dev(S, dev)
        sig = (U.conj() * (Sd @ U)).sum(dim=0).real.cpu().numpy()
        B["sigma_impurity_max_full_eigh"] = float(np.max(np.abs(np.abs(sig) - 1.0)))
        B["n_sector_plus_full_eigh"] = int((sig > 0).sum())
        # the sector eigenvectors lifted: residual of h (P_+ u_+) - e (P_+ u_+)
        Ppd = dense_dev(Pp, dev)
        Vp = Ppd @ Upl
        B["residual_lifted_sector_eigenvectors"] = float((M @ Vp - Vp * ep[None, :]).abs().max().item())
        B["eigh_seconds_full_2n"] = t_full; B["eigh_seconds_sector_n"] = t_half; B["speedup_eigh"] = t_full / t_half
        out["B"][str(kappa)] = B
        say("B kappa %d (m 0): [S, h] %.1e; off-block %.1e; H+ product vs direct %.1e; H- + conj(H+) %.1e; nnz h %d H+ %d (u %d); spectrum union vs full %.2e; full-eigh sector impurity %.1e; lifted residual %.1e; eigh %d: %.3f s, eigh %d: %.3f s (x%.1f)" % (
            kappa, B["norm_commutator_S_h"], B["max_abs_offblock_Pm_h_Pp"], B["max_abs_Hp_product_minus_direct"], B["max_abs_Hm_plus_conj_Hp"], B["h_nnz"], B["Hp_nnz"], B["Hp_bandwidth_u"],
            B["max_abs_spectrum_diff_full_vs_sector_union"], B["sigma_impurity_max_full_eigh"], B["residual_lifted_sector_eigenvectors"], 2 * n, t_full, n, t_half, B["speedup_eigh"]))
        del M, U, Mp, Upl, Sd, Ppd, Vp
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    # ---------------------------------------------------------------- C: the twofield light walk's Cholesky certificate per sector
    for kappa in kappas[:2]:
        h0, Q = VF.chain_sparse(R, 0.0, float(kappa))
        K0 = PO6.clocked_csr(h0, Q, N_prev, v_prev); K1 = PO6.clocked_csr(h0, Q, N, v)
        Hp0 = sector_block_direct(N_prev, v_prev, R, float(kappa)); Hp1 = sector_block_direct(N, v, R, float(kappa))
        # the start gap of the + sector at mu = 0 (the anchor at zero: K = h - 0 S)
        Mp = dense_dev(Hp0, dev); ep = torch.linalg.eigvalsh(Mp).cpu().numpy(); del Mp
        g0 = float(np.min(np.abs(ep)))
        C = {"gap_start_plus_sector": g0, "ladder": []}
        t_band_full = t_band_sec = t_chol_full = t_chol_sec = t_sq_full = t_sq_sec = 0.0; n_calls = 0
        for s in (0.0, 0.25, 0.5, 0.75, 1.0):
            Ks = (K0 + s * (K1 - K0)).tocsr(); Hs = (Hp0 + s * (Hp1 - Hp0)).tocsr()
            t0 = time.time(); K2 = (Ks @ Ks).tocsr(); t_sq_full += time.time() - t0
            t0 = time.time(); H2 = (Hs @ Hs).tocsr(); t_sq_sec += time.time() - t0
            t0 = time.time(); abF, uF = band_upper(K2); t_band_full += time.time() - t0
            t0 = time.time(); abS, uS = band_upper(H2); t_band_sec += time.time() - t0
            d = g0 * 1.5
            for k in range(12):
                t0 = time.time(); vF = pd_banded(abF, d); t_chol_full += time.time() - t0
                t0 = time.time(); vS = pd_banded(abS, d); t_chol_sec += time.time() - t0
                n_calls += 1
                C["ladder"].append({"s": s, "d": d, "full": bool(vF), "sector": bool(vS)})
                d *= 0.5
        agree = all(q["full"] == q["sector"] for q in C["ladder"])
        C.update({"verdicts_agree": bool(agree), "n_cholesky_calls_each": n_calls, "u_full_K2": int(uF), "u_sector_H2": int(uS),
                  "seconds_cholesky_full_total": t_chol_full, "seconds_cholesky_sector_total": t_chol_sec, "seconds_band_full": t_band_full, "seconds_band_sector": t_band_sec,
                  "seconds_square_full": t_sq_full, "seconds_square_sector": t_sq_sec})
        out["C"][str(kappa)] = C
        say("C kappa %d: + sector start gap %.3e; %d Cholesky calls each: verdicts agree %s; per call full %.4f s (u %d) vs sector %.4f s (u %d) (x%.1f); band full %.4f vs sector %.4f s; K@K full %.4f vs sector %.4f s" % (
            kappa, g0, n_calls, agree, t_chol_full / n_calls, uF, t_chol_sec / n_calls, uS, t_chol_full / max(t_chol_sec, 1e-12), t_band_full / 5, t_band_sec / 5, t_sq_full / 5, t_sq_sec / 5))
        disagreements = [q for q in C["ladder"] if q["full"] != q["sector"]]
        if disagreements:
            say("  disagreements: %s" % disagreements[:5])
    # ---------------------------------------------------------------- T: eigh timings on the device at the production sizes
    for sz in [int(x) for x in a.timing_sizes.split(",")]:
        g = torch.Generator(device="cpu").manual_seed(1)
        Ar = torch.randn((sz, sz), dtype=torch.float64, generator=g); Ai = torch.randn((sz, sz), dtype=torch.float64, generator=g)
        Mc = ((Ar + 1j * Ai) + (Ar + 1j * Ai).conj().T).to(dev)
        Mr = (Ar + Ar.T).to(dev)
        _, _, tc, tcs = eigh_t(Mc, reps=3)
        _, _, tr, trs = eigh_t(Mr, reps=3)
        out["T"][str(sz)] = {"complex128_eigh_s": tc, "float64_eigh_s": tr, "complex_list": tcs, "real_list": trs}
        say("T n %d: complex128 eigh %.3f s (%s), float64 eigh %.3f s" % (sz, tc, " ".join("%.3f" % x for x in tcs), tr))
        del Mc, Mr
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    with open(os.path.join(a.out, "sym_test_v1.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(a.out, "sym_test_v1.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    say("written %s" % a.out)


if __name__ == "__main__":
    main()
