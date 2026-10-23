#!/usr/bin/env python
"""pt3_output.py (busi, PAGE design v3 sec. 7 and sec. 3; A5, A8, F8, F9): the nested orthonormal outgoing modes, S(b|R), and the certified
unitary exterior.

Output algebra. Raw bins B_1, B_2, ... (columns: packets, both bands, declared copy weights), ordered by arrival at the collar. nested_modes
orthonormalizes them in arrival order (block Gram-Schmidt with one re-orthogonalization): bin n is made orthogonal to bins 1 .. n-1 and
orthonormal within itself, so no later arrival is ever moved into an earlier bin (the earlier W_k never change when a bin is added); the raw Gram
matrix, each bin's retained fraction after orthogonalization and the final orthonormality defect are printed. Every mode is counted once
(rank-revealing: columns below a declared retained-norm threshold are dropped and counted).
Entropies (absolute, independent of any subtraction): with C_out = W^dag C W (W the nested modes), S(R_n), S(b_n), I(b_n : R_{n-1}) and
S(b_n | R_{n-1}) = S(R_n) - S(R_{n-1}), each with a certified interval (pt3_core.s_interval at eigenvalue error eps = the state's error bound
+ the Hermiticity defect; Weyl), never clipped; an eigenvalue farther than eps outside [0, 1] invalidates the interval (inconclusive).
The pure-state floor (F9): on an exactly pure state the entropy of a complete mode set is zero; floor_pure_state returns that computed value.
The bound inventory (A8): bound_projector gives the orthogonal projector onto the bound levels of a self-adjoint held operator (eigenvectors
whose weight beyond a radius stays below a threshold), with Tr s(C_b), Tr[(C - P_-) Pi_b] and the residual weight.
Exterior certification (F8, A5): a self-adjoint parent chain is unitary by construction; certify_exterior gives the first-return time
T_ret = 2 (L - R_read)/v_max with v_max the largest |d omega/dk| over every site and k of the actual symbol (both bands, the near-root cubic
branch included, against the bound N|cos k| + |v||cos 2k| <= N + |v|), and unitarity_check the forward-backward error and the norm drift on
probes propagated by the parent's Chebyshev exponential (pt_lib_v1)."""
import math

import numpy as np

import pt3_core as K


def nested_modes(bins, drop_tol=1e-10):
    """bins: list of (d, m_n) arrays in arrival order. Returns (W list, report)."""
    Ws, rep = [], {"gram_raw": None, "retained_fraction": [], "dropped": [], "orthonormality_defect": None}
    allraw = np.concatenate(bins, axis=1)
    G = allraw.conj().T @ allraw
    rep["gram_raw"] = G
    prev = np.zeros((allraw.shape[0], 0), complex)
    for B in bins:
        X = np.array(B, complex)
        n0 = np.linalg.norm(X, axis=0)
        for _ in range(2):
            X = X - prev @ (prev.conj().T @ X)
        frac = np.linalg.norm(X, axis=0) / np.maximum(n0, 1e-300)
        # orthonormalize within the bin (eigen route: keeps the bin's span, drops near-null directions)
        S = X.conj().T @ X
        lam, V = np.linalg.eigh(0.5 * (S + S.conj().T))
        keep = lam > drop_tol * max(float(lam.max()), 1e-300)
        Wn = X @ (V[:, keep] / np.sqrt(lam[keep])[None, :])
        Wn = Wn - prev @ (prev.conj().T @ Wn)
        q, _ = np.linalg.qr(Wn)
        Ws.append(q)
        rep["retained_fraction"].append([float(f) for f in frac])
        rep["dropped"].append(int((~keep).sum()))
        prev = np.concatenate([prev, q], axis=1)
    rep["orthonormality_defect"] = float(np.max(np.abs(prev.conj().T @ prev - np.eye(prev.shape[1])))) if prev.shape[1] else 0.0
    return Ws, rep


def nested_entropies(C, Ws, eps=0.0):
    """S(R_n), S(b_n), I(b_n:R_{n-1}), S(b_n|R_{n-1}) with certified intervals. C dense (d, d) or (Phi, f) columns."""
    if isinstance(C, tuple):
        Phi, f = C
        A = [W.conj().T @ Phi for W in Ws]
        def blk(i, j):
            return (A[i] * np.asarray(f)[None, :]) @ A[j].conj().T if f is not None else A[i] @ A[j].conj().T
    else:
        def blk(i, j):
            return Ws[i].conj().T @ C @ Ws[j]
    n = len(Ws)
    full = np.block([[blk(i, j) for j in range(n)] for i in range(n)])
    sizes = np.cumsum([0] + [W.shape[1] for W in Ws])
    rows = []
    prev = {"S": 0.0, "S_lo": 0.0, "S_hi": 0.0}
    for k in range(n):
        R = K.cov_entropy(full[: sizes[k + 1], : sizes[k + 1]], eps)
        b = K.cov_entropy(full[sizes[k]: sizes[k + 1], sizes[k]: sizes[k + 1]], eps)
        cond = R["S"] - prev["S"]
        cond_lo = R["S_lo"] - prev["S_hi"]; cond_hi = R["S_hi"] - prev["S_lo"]
        rows.append({"n": k + 1, "S_R": R["S"], "S_R_lo": R["S_lo"], "S_R_hi": R["S_hi"], "S_b": b["S"], "S_b_lo": b["S_lo"], "S_b_hi": b["S_hi"],
                     "I_b_R": b["S"] + prev["S"] - R["S"], "S_b_given_R": cond, "S_b_given_R_lo": cond_lo, "S_b_given_R_hi": cond_hi,
                     "valid": bool(R["valid"] and b["valid"]), "excursion": max(R["excursion"], b["excursion"])})
        prev = R
    return rows, full


def floor_pure_state(C_pure, d):
    """the computed entropy of the complete orthonormal mode set (the identity) on an exactly pure covariance: the entropy floor (F9)."""
    return K.cov_entropy(np.asarray(C_pure), 0.0)["S"]


def bound_projector(h_dense, R_sites2, R_bound, w_tol=1e-6):
    """eigenvectors of the self-adjoint h with weight beyond R_bound below w_tol: the bound levels; returns (Pi_b columns, energies, max out-weight)."""
    e, U = np.linalg.eigh(h_dense)
    out = np.real((np.abs(U[R_sites2 > R_bound, :]) ** 2).sum(axis=0))
    b = out < w_tol
    return U[:, b], e[b], float(out[b].max()) if b.any() else None


def bound_inventory(C, Pminus, Ub):
    Cb = Ub.conj().T @ C @ Ub
    return {"Tr_s_Cb": K.cov_entropy(Cb)["S"], "Tr_C_minus_Pm_on_bound": float(np.real(np.trace(Ub.conj().T @ (C - Pminus) @ Ub))), "n_bound": Ub.shape[1]}


# ------------------------------------------------------------------------------------------------------------ the exterior
def group_velocity_max(N, v, kappa, R, nk=2049):
    """max over sites and k of |d omega/dk| for omega_+-(k) = v sin(2k)/2 +- N sqrt(sin^2 k + y^2), y = kappa/R (the chain symbol, both bands);
    also the bound max(N + |v|) and the near-root maximum (sites with N - |v| < 0.1 max N)."""
    N = np.asarray(N, float); v = np.asarray(v, float); y = kappa / np.asarray(R, float)
    k = np.linspace(-math.pi, math.pi, nk)[None, :]
    s, c = np.sin(k), np.cos(k)
    best = np.zeros(len(N))
    for j0 in range(0, len(N), 4096):
        sl = slice(j0, j0 + 4096)
        e = np.sqrt(s * s + (y[sl, None]) ** 2)
        vg_flow = v[sl, None] * np.cos(2 * k)
        vg_clock = N[sl, None] * s * c / e
        best[sl] = np.max(np.maximum(np.abs(vg_flow + vg_clock), np.abs(vg_flow - vg_clock)), axis=1)
    near = (N - np.abs(v)) < 0.1 * float(N.max())
    return {"v_max": float(best.max()), "bound_max": float(np.max(N + np.abs(v))), "near_root_v_max": float(best[near].max()) if near.any() else None,
            "n_near_root_sites": int(near.sum()), "site_of_max": int(np.argmax(best))}


def certify_exterior(N, v, kappa, R, R_read, T_needed, rs=1.0):
    gv = group_velocity_max(N, v, kappa, R)
    L = float(np.max(R))
    T_ret = 2.0 * (L - R_read) / gv["v_max"]
    T_ret_bound = 2.0 * (L - R_read) / gv["bound_max"]
    return {"L_sites": L, "L_rs": L / rs, "R_read": R_read, "v_max": gv["v_max"], "v_bound": gv["bound_max"], "near_root_v_max": gv["near_root_v_max"],
            "T_return": T_ret, "T_return_rs": T_ret / rs, "T_return_bound_rs": T_ret_bound / rs, "T_needed_rs": T_needed / rs,
            "certified": bool(T_ret_bound >= T_needed), "length_ge_1e4_rs": bool(L / rs >= 1e4)}


def unitarity_check(n_sites, kappa, N, v, T, n_probe=4, seed=3, dt=None):
    """forward exp(-i T h) then backward on random probes with pt_lib_v1's Chebyshev exponential (CPU, torch banded): max |X_back - X|, the
    norm drift and the Hermiticity defect of h (exactly 0 by construction: h = (1/2){N, h_0} + (1/2){v, Q})."""
    import torch
    import pt_lib_v1 as L1
    ch = L1.Chain(n_sites, kappa, "cpu")
    Nt = torch.as_tensor(np.asarray(N, float), dtype=L1.RD); vt = torch.as_tensor(np.asarray(v, float), dtype=L1.RD)
    D = ch.diags(Nt, vt)
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((ch.rows, n_probe)) + 1j * rng.standard_normal((ch.rows, n_probe))
    X0 /= np.linalg.norm(X0, axis=0)[None, :]
    X = torch.tensor(X0.copy(), dtype=L1.CD)              # a copy: as_tensor would share X0's memory
    P = L1.Prop(ch.rows, n_probe, "cpu")
    nsteps = 1 if dt is None else max(1, int(math.ceil(T / dt)))
    tau = T / nsteps
    tails = 0.0
    for _ in range(nsteps):
        tails += P.expm(D, X, tau, ch.rows, n_probe)
    nf = np.linalg.norm(X.numpy(), axis=0)
    for _ in range(nsteps):
        tails += P.expm(D, X, -tau, ch.rows, n_probe)
    err = float(np.max(np.abs(X.numpy() - X0)))
    h = ch.csr(np.asarray(N, float), np.asarray(v, float))
    herm = float(abs(h - h.conj().T).max())
    return {"forward_backward_max_err": err, "norm_drift_max": float(np.max(np.abs(nf - 1.0))), "hermiticity_defect": herm, "cheb_tail_sum": tails,
            "T": T, "n_probe": n_probe, "rows": ch.rows, "nsteps": nsteps}
