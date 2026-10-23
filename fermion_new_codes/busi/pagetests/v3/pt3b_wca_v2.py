#!/usr/bin/env python
"""pt3b_wca_v2.py (busi; B' v2: reviewer B's artificial-window control (ReviewB_pc13 sec. 1.7) with its full output saved, as both checks of design v1
ask (ReviewA_bpchk_v1 sec. 1 table: "also save its full output covariance"; ReviewB_bpchk_v1 sec. 1: "the complete Hamiltonian and output
covariance"). window_control_reviewb_v2 is pt3b_chain.window_control_reviewb copied unchanged except that it also returns the full escape
(depletion) Gram at the end clock and, for 2 n_w <= 200, at every clock, with the block's levels and couplings (the complete auxiliary
Hamiltonian: the block diagonal e_s, e_m, the couplings Gc g on the port orbital, the native chain h on the profile, the absorber)."""
import math

import numpy as np
from scipy.linalg import expm

import pt3_core as K3


def window_control_reviewb_v2(n_w, L, prof, arm="mixing", slide_rate=0.0, G2_over_T0=0.1, T_obs_over_L=200.0, n_bins=20, seed=7, kappa=1,
                         box_L=2.5, onset_L=2.0, eta_max=1.0):
    """ARTIFICIAL (labelled; not realizable by h[N, v]): n_w levels e_j spread uniformly through (0, T0) (T0 = 1/(4 pi L)), each a basis state
    |j>, coupled to the native kappa = 1 chain's port orbital |p> (the upper component at the chain's first site outside the profile's root)
    by H_c = sum_j g_j (|j><p| + h.c.) with the port-coupling norm sum |g_j|^2 = G^2 fixed as n_w changes (G^2 = G2_over_T0 T0^2... set by the
    coupling strength argument). Arms: 'mixing' (nondegenerate levels, g_j = G n_w^(-1/2) e^(i phi_j), random phases: every level couples);
    'bright' (all levels degenerate at T0/2, g_j = G n_w^(-1/2): one bright combination, n_w - 1 dark ones). slide_rate: e_j(t) = e_j - rate t
    (prescribed monotone slide; 0 = stationary). The chain: the profile's kappa = 1 operator from the root to box_L L, reviewer A's absorber beyond
    onset_L L. Prepared inputs: every block level (supply), plus (i) a hole: the highest mirror level -e_j empty in a filled mirror band (its exit
    through (-T0, 0)), (ii) a correlation-paired mode: one supply level split coherently with an inert early-output mode (purification test).
    Returns the escape (depletion) matrix spectrum at n_bins clocks, the emitted number, the return kernel S(t) diagonal, the paired mode's
    nested-entropy rows, the hole's exit."""
    import pt3b_kernels as KR
    import pt3b_cube as CU
    rng = np.random.default_rng(seed)
    T0 = 1.0 / (4 * math.pi * L)
    R_t = max(prof["roots_over_L"]) * L
    i0 = int(math.ceil(R_t))
    n = int(round(box_L * L)) - i0 + 1
    ops = K3.ChainOps(n, kappa, R0=float(i0))
    x = ops.R / L
    N = prof["N"](x); v = prof["v"](x)
    h_ch = ops.h(N, v).toarray()
    eta = np.repeat(CU.cap_reviewa(ops.R, L, onset_L, box_L - onset_L, eta_max), 2)
    # the block: n_w supply levels and n_w mirror levels (the hole's band)
    if arm == "bright":
        e_s = np.full(n_w, 0.5 * T0)
        g = np.full(n_w, 1.0 / math.sqrt(n_w), complex)
    else:
        e_s = (np.arange(n_w) + 0.5) / n_w * T0
        g = np.exp(2j * math.pi * rng.random(n_w)) / math.sqrt(n_w)
    e_m = -e_s[::-1]
    gm = g[::-1].copy()
    nb = 2 * n_w
    dim = nb + 2 * n
    # the port: the chain's density of states at the port orbital near 0 (for the coupling norm): G^2 = G2_over_T0 * T0 / (2 pi rho_p)
    ec, Uc = np.linalg.eigh(h_ch)
    p_idx = 0
    w_p = np.abs(Uc[p_idx]) ** 2
    rho = float(np.sum(w_p * np.exp(-(ec / (2 * T0)) ** 2)) / (math.sqrt(math.pi) * 2 * T0))
    Gc = math.sqrt(G2_over_T0 * T0 / (2 * math.pi * max(rho, 1e-300)))
    def Hfull(t):
        H = np.zeros((dim, dim), complex)
        es = e_s - slide_rate * t; em = e_m + slide_rate * t
        H[np.arange(n_w), np.arange(n_w)] = es
        H[np.arange(n_w, nb), np.arange(n_w, nb)] = em
        H[nb:, nb:] = h_ch - 1j * np.diag(eta)
        cpl = Gc * np.concatenate([g, gm])
        H[:nb, nb + p_idx] = cpl; H[nb + p_idx, :nb] = cpl.conj()
        return H
    imask = np.zeros(dim, bool); imask[:nb] = True
    T_obs = T_obs_over_L * L
    ts = np.linspace(0, T_obs, n_bins + 1)[1:]
    F = np.zeros((dim, nb), complex); F[np.arange(nb), np.arange(nb)] = 1.0
    if slide_rate == 0.0:
        Hs = Hfull(0.0)
        wv, Vr = np.linalg.eig(Hs)
        Vinv = np.linalg.inv(Vr)
        prop = lambda t: (Vr * np.exp(-1j * wv * t)) @ (Vinv @ F)
        Psis = [prop(t) for t in ts]
    else:
        dt = T_obs / (20 * n_bins)
        Psi = F.copy(); Psis = []; t = 0.0
        for T in ts:
            while t < T - 1e-9:
                hh = min(dt, T - t)
                Psi = expm(-1j * Hfull(t + 0.5 * hh) * hh) @ Psi; t += hh
            Psis.append(Psi.copy())
    Gs = [(t, np.eye(nb) - P.conj().T @ P) for t, P in zip(ts, Psis)]
    sup = np.arange(n_w)
    p_end = np.sort(np.linalg.eigvalsh(0.5 * (Gs[-1][1][np.ix_(sup, sup)] + Gs[-1][1][np.ix_(sup, sup)].conj().T)))[::-1]
    S_diag = [[complex(P[j, j]) for j in sup[:min(5, n_w)]] for P in Psis]
    # the paired mode: input psi = (|f_0> + |inert>)/sqrt 2 occupied; the inert mode counted as collected output from t = 0
    Gaug = []
    for t, G in Gs:
        A = np.zeros((2, 2), complex); A[0, 0] = G[0, 0]; A[1, 1] = 1.0
        Gaug.append((t, A))
    Cpair = 0.5 * np.array([[1, 1], [1, 1]], complex)
    pair_rows = KR.collected_entropies([(0.0, np.diag([0.0, 1.0]).astype(complex))] + Gaug, Cpair)
    # the hole: the mirror band filled except its highest level; the hole's exit = that level's escape
    jh = nb - 1
    hole_exit = float(np.real(Gs[-1][1][jh, jh]))
    G_end = Gs[-1][1]
    G_bins = np.array([G for _, G in Gs]) if nb <= 200 else None
    full = {"G_end": G_end, "G_bins": G_bins, "t": ts, "e_s": e_s, "e_m": e_m, "g": g, "Gc": Gc}
    return full, {"n_w": n_w, "arm": arm, "slide_rate": slide_rate, "T0": T0, "G_coupling": Gc, "rho_port": rho, "T_obs_over_L": T_obs_over_L,
            "emitted_number_supply": float(np.real(np.trace(Gs[-1][1][np.ix_(sup, sup)]))), "depletion_spectrum_end": [float(x) for x in p_end],
            "largest_over_median": float(p_end[0] / max(np.median(p_end), 1e-300)), "n_p_above_1e-2": int(np.sum(p_end > 1e-2)),
            "depletion_diag_end": [float(np.real(Gs[-1][1][j, j])) for j in sup], "return_S_diag": [[abs(c) for c in row] for row in S_diag],
            "paired_mode_rows": pair_rows, "hole_exit": hole_exit, "t_over_L": [float(t / L) for t in ts]}
