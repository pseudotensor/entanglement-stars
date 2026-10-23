#!/usr/bin/env python
"""pt3_prep.py (busi, PAGE design v3 sec. 1; A2, F5, F8): the two initial states and the constructed depleted arm.

(a) The old-reference history state C_0 = f_ws(h_0), f_ws(E) = erfc(E/(sqrt 2 ws))/2 (full rank, mixed). mixed_diagnostics measures, before any
    use: ||C_0 - P_-(h_0)||_1 = sum_i |f_ws(e_i) - theta(-e_i)| and S(C_0) = sum_i s(f_ws(e_i)), exactly from a dense or sector spectrum, or by a
    stochastic trace of Chebyshev moments (the KPM route for long chains, with its statistical error), and the count of C_0's eigenvalues near
    1/2. Occupations may be read from (a); entropies and purity may not (the rule is printed by the readers).
(b) The pure preparation C_0 = P_-(h_0) exactly, zero modes empty: pure_sector (one dense eigh of the + sector block, page_w1_sector_v1d's trick)
    where the dimension allows; otherwise sharp_step_projector_apply: the Chebyshev series of the smoothed step at a width ws_sharp below the
    measured smallest |e| of h_0 (min_abs_eig_banded: eigenvalues only of the banded + sector in (-delta, delta)), applied to the vectors that
    need it, with the idempotency defect ||C^2 - C||_1 = Tr f(1 - f) (stochastic trace, or exact on a spectrum) printed.
(c) The constructed depleted arm (F5): C_dep = theta(-QhQ) (+) theta(-(1-Q)h(1-Q)) at T_hold, pure by construction; checks 0 <= C_dep <= 1, its
    idempotency defect, Tr s(C_dep) and ||C_dep - C_nat||_1."""
import math

import numpy as np
import scipy.sparse as sp
from scipy.special import erfc

import pt3_core as K


def f_ws(E, ws):
    return 0.5 * erfc(np.asarray(E, float) / (math.sqrt(2.0) * ws))


def theta_minus(E):
    return (np.asarray(E, float) < 0).astype(float)


def mixed_diagnostics_exact(e, ws):
    """from the full spectrum e of h_0."""
    f = f_ws(e, ws)
    return {"trace_norm_C0_minus_P": float(np.abs(f - theta_minus(e)).sum()), "S_C0": float(K.s_bin(f).sum()),
            "n_near_half_(0.1,0.9)": int(((f > 0.1) & (f < 0.9)).sum()), "n_zero_modes_1e-12": int((np.abs(e) < 1e-12).sum()), "ws": ws}


def cheb_coeffs(fun, w, M):
    """Chebyshev coefficients of fun on [-w, w] (DCT at Mn >= 2M nodes)."""
    from scipy.fft import dct
    Mn = 1
    while Mn < 2 * M:
        Mn *= 2
    th = (np.arange(Mn) + 0.5) * math.pi / Mn
    c = dct(fun(w * np.cos(th)), type=2) / Mn
    c[0] /= 2.0
    return c[:M]


def jackson(M):
    k = np.arange(M)
    q = math.pi / (M + 1)
    return ((M - k + 1) * np.cos(k * q) + np.sin(k * q) / math.tan(q)) / (M + 1)


def stochastic_trace(h, fun, w, M, n_probe=32, seed=1):
    """Tr fun(h) by Rademacher probes and the Chebyshev recursion on the host sparse h (spectrum inside [-w, w]); (mean, standard error)."""
    rng = np.random.default_rng(seed)
    d = h.shape[0]
    c = cheb_coeffs(fun, w, M)
    vals = []
    for _ in range(n_probe):
        z = rng.choice([-1.0, 1.0], size=d) + 0j
        t0 = z.copy(); t1 = (h @ z) / w
        acc = c[0] * np.vdot(z, t0).real + c[1] * np.vdot(z, t1).real
        for k in range(2, M):
            t2 = 2.0 * (h @ t1) / w - t0
            acc += c[k] * np.vdot(z, t2).real
            t0, t1 = t1, t2
        vals.append(acc)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals)))


def mixed_diagnostics_kpm(h, ws, w=None, M=None, n_probe=32, seed=1):
    """the KPM route: ||C_0 - P_-||_1 = Tr |f_ws(h) - theta(-h)| and S(C_0) = Tr s(f_ws(h)) by stochastic traces; the integrands vanish
    away from |E| <~ 8 ws, so the expansion needs M ~ 16 w/ws terms."""
    if w is None:
        w = float(abs(h).sum(axis=1).max()) * 1.0001
    if M is None:
        M = int(24 * w / ws) + 64
    g1 = lambda E: np.abs(f_ws(E, ws) - theta_minus(E))
    g2 = lambda E: K.s_bin(f_ws(E, ws))
    a, da = stochastic_trace(h, g1, w, M, n_probe, seed)
    b, db = stochastic_trace(h, g2, w, M, n_probe, seed + 1)
    return {"trace_norm_C0_minus_P": a, "trace_norm_se": da, "S_C0": b, "S_C0_se": db, "M": M, "n_probe": n_probe, "ws": ws, "w": w}


def pure_sector(h0_csr):
    """C_0 = P_-(h_0) by one dense eigh of h_+; returns (occupied columns (2n x n_neg), e, n_zero, the idempotency defect of V V^dag computed
    as ||V^dag V - 1||_1 (the Gram's trace-norm defect, page_w1_sector_v1d.impurity_trace_norm's measure))."""
    e, U, sec = K.full_spectrum_from_sector(h0_csr)
    neg = e < 0
    V = U[:, neg]
    G = V.conj().T @ V
    ev = np.linalg.eigvalsh(0.5 * (G + G.conj().T))
    return {"V": V, "e": e, "sec": sec, "n_neg": int(neg.sum()), "n_zero_1e-12": int((np.abs(e) < 1e-12).sum()),
            "gram_defect_1": float(np.abs(ev - 1.0).sum()), "route": "sector (dense eigh of h_+)"}


def min_abs_eig_banded(h_csr, delta):
    """the smallest |e| of h from the + sector's banded eigenvalues in (-delta, delta) (eigenvalues only); None if none there."""
    from scipy.linalg import eigvals_banded
    n = h_csr.shape[0] // 2
    hp = K.sector_plus(h_csr).tocsr()
    ab = np.zeros((3, n), complex)
    for d in range(3):
        ab[2 - d, d:] = hp.diagonal(d)
    ev = eigvals_banded(ab, lower=False, select="v", select_range=(-delta, delta))
    return (float(np.min(np.abs(ev))) if len(ev) else None), len(ev)


def sharp_step_apply(h, X, ws, w=None, tol=1e-14):
    """f_ws(h) X by the Chebyshev series (host sparse, the production uses pt_lib_v1.Prop.series on the GPU with the same coefficients)."""
    if w is None:
        w = float(abs(h).sum(axis=1).max()) * 1.0001
    M = 64
    while True:
        c = cheb_coeffs(lambda E: f_ws(E, ws), w, M)
        if np.max(np.abs(c[-8:])) < tol or M > 2 ** 22:
            break
        M *= 2
    keep = np.nonzero(np.abs(c) > tol)[0]
    c = c[: keep[-1] + 1]
    t0 = np.array(X, complex); t1 = (h @ t0) / w
    Y = c[0] * t0 + c[1] * t1
    for k in range(2, len(c)):
        t2 = 2.0 * (h @ t1) / w - t0
        Y += c[k] * t2
        t0, t1 = t1, t2
    return Y, len(c)


def idempotency_defect_exact(e, ws):
    f = f_ws(e, ws)
    return float((f * (1 - f)).sum())


def depleted_projector(h, qmask, C_nat=None):
    """F5's C_dep = theta(-QhQ) (+) theta(-(1-Q)h(1-Q)) and its printed checks."""
    H = h.toarray() if sp.issparse(h) else np.asarray(h)
    qi = np.where(qmask)[0]; qo = np.where(~np.asarray(qmask, bool))[0]
    ei, Ui = np.linalg.eigh(H[np.ix_(qi, qi)]); eo, Uo = np.linalg.eigh(H[np.ix_(qo, qo)])
    d = H.shape[0]
    Cd = np.zeros((d, d), complex)
    Cd[np.ix_(qi, qi)] = (Ui[:, ei < 0]) @ Ui[:, ei < 0].conj().T
    Cd[np.ix_(qo, qo)] = (Uo[:, eo < 0]) @ Uo[:, eo < 0].conj().T
    nu = np.linalg.eigvalsh(0.5 * (Cd + Cd.conj().T))
    out = {"min_eig": float(nu.min()), "max_eig": float(nu.max()), "idempotency_defect_F": float(np.linalg.norm(Cd @ Cd - Cd)),
           "Tr_s": float(K.s_bin(np.clip(nu, 0, 1)).sum()), "n_zero_modes_inner_1e-12": int((np.abs(ei) < 1e-12).sum()),
           "n_zero_modes_outer_1e-12": int((np.abs(eo) < 1e-12).sum())}
    if C_nat is not None:
        out["trace_norm_minus_native"] = float(np.abs(np.linalg.eigvalsh(Cd - C_nat)).sum())
    return Cd, out
