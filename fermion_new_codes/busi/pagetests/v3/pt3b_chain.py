#!/usr/bin/env python
"""pt3b_chain.py (busi; B' chain comparator and reviewer A pc12 sec. 3.2's artificial-window chain control).

Comparator (reviewer A item 5, reviewer B item 3): the angular chains kappa = 1..10 (page_ops_v6.chain_sparse, m = 0) on the same radial profile as the
cube (matched profile or stored record), sites out to 2.5 r_s with the absorbing potential beyond 2 r_s, the same retained space (sites inside
the outer root), the same kernels and readouts (pt3b_kernels); per-channel results and the totals weighted by the paper's multiplicity 16 kappa.
The chain's propagation is exact: one non-Hermitian matrix exponential per dt (dense, chains are small).

Artificial-window control (reviewer A 3.2): one channel (kappa = 1); the record's exterior (rooted record at r_s/a 64) beyond R_t is held; the
interior is replaced by a cavity of L_c sites with a uniform clock N_c and no flow, joined to the record's profile at R_t by a cosine ramp of
8 sites; N_c sets the cavity level spacing (about pi N_c/L_c per valley), chosen so that n_lev cavity levels lie in the window (0, T_H)
(n_lev = 1, 10, 100, 1000; L_c grown with n_lev). Prepared state (one-body): the cavity levels with e in (T_H, T_H + slide) filled;
prescribed slow slide N_c(t) = N_c0 (1 - t/tau) carries them down through the window (and the empty mirror levels up through (-T_H, 0)).
Measured: the emitted number per transit (the collected weight 1 - ||Psi(T)||^2 summed, plus the exterior weight), the per-level escape
distribution (equal fractions: H1; one bright level: its failure), and a prepared hole (one level in (-T_H - slide, -T_H) left empty) and its
exit through the mirror window (H2). Propagation: Chebyshev on the sparse chain with the Strang absorber split."""
import math

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

import pt3_core as K3


def chain_profile_fields(R, rp, Np, up, rs_vac):
    u = np.where(R < rp[0], up[0] * R / rp[0], np.where(R > rp[-1], np.sqrt(rs_vac / R), np.interp(R, rp, up)))
    N = np.where(R < rp[0], Np[0], np.where(R > rp[-1], 1.0, np.interp(R, rp, Np)))
    return N, -u


def chain_kernel(kappa, rp, Np, up, rs_vac, rs_a, R_t, T_H, box_rs=2.5, cap_rs=2.0, eta0=0.05, T_list=None, dt=None, clock=None):
    import pt3b_kernels as KR
    n = int(round(box_rs * rs_a))
    ops = K3.ChainOps(n, kappa)
    N, v = chain_profile_fields(ops.R, rp, Np, up, rs_vac)
    if clock is not None:
        d, mode = clock
        x = ops.R / R_t
        f = np.where(x <= 0.9, 1.0, np.where(x >= 1.0, 0.0, 0.5 * (1 + np.cos(math.pi * (x - 0.9) / 0.1))))
        s = np.exp(-d * f)
        N = N * s
        if mode == "reviewb":
            v = v * s
    h = ops.h(N, v)
    eta = np.repeat(np.where(ops.R > cap_rs * rs_a, eta0 * ((ops.R - cap_rs * rs_a) / ((box_rs - cap_rs) * rs_a)) ** 2, 0.0), 2)
    imask = ops.site_mask(R_t)
    out = {"kappa": kappa, "n_sites": n}
    if T_list is None:
        T_list = KR.T_list_for(rs_a)
    dt = dt or 0.5 * rs_a
    Ueta = expm(-1j * (h.toarray() - 1j * np.diag(eta)) * dt)
    step = lambda X, hh: (Ueta @ X) if abs(hh - dt) < 1e-12 else expm(-1j * (h.toarray() - 1j * np.diag(eta)) * hh) @ X
    q = ops.h(np.zeros(ops.n), -np.ones(ops.n))          # d h / d u for v = -u (the flow coupling), Hellmann-Feynman
    for lab, (lo, hi) in {"supply": (0.0, T_H), "exit": (-T_H, 0.0)}.items():
        e, U, dI = KR.interior_window(h, imask, lo, hi)
        if len(e) == 0:
            out[lab] = {"N_w": 0, "e_over_TH": []}
            continue
        kern, _ = KR.propagate_kernel(step, U, T_list, dt, imask)
        r = KR.readouts(e, kern, T_H, rs_a)
        r["e_over_TH"] = [float(x) for x in e / T_H]
        r["q_r"] = [float(x) for x in np.real(np.einsum("ij,ij->j", U.conj(), q @ U))]
        out[lab] = r
    return out


def chains_comparator(rp, Np, up, rs_vac, rs_a, R_t, T_H, kmax=10, **kw):
    per = {k: chain_kernel(k, rp, Np, up, rs_vac, rs_a, R_t, T_H, **kw) for k in range(1, kmax + 1)}
    tot = {}
    for lab in ("supply", "exit"):
        tot[lab] = {"N_w_weighted": float(sum(16 * k * per[k][lab]["N_w"] for k in per)),
                    "N_acc_weighted": {th: float(sum(16 * k * per[k][lab].get("N_acc", {}).get(th, 0) for k in per)) for th in ("0.1", "0.01", "0.001")}}
    return per, tot


# ------------------------------------------------------------------------------------------------ the artificial-window control
def cavity_chain(L_c, N_c, ext_R, ext_N, ext_v, kappa=1, ramp=8):
    """sites 1..L_c the cavity (N = N_c, v = 0), then a cosine ramp of `ramp` sites into the record's exterior profile (ext_R relabelled to
    follow the cavity), returned as (ops, N, v, n_cav)."""
    n = L_c + ramp + len(ext_R)
    ops = K3.ChainOps(n, kappa)
    N = np.empty(n); v = np.empty(n)
    N[:L_c] = N_c; v[:L_c] = 0.0
    s = 0.5 * (1 - np.cos(math.pi * np.arange(1, ramp + 1) / (ramp + 1)))
    N[L_c:L_c + ramp] = N_c + s * (ext_N[0] - N_c); v[L_c:L_c + ramp] = s * ext_v[0]
    N[L_c + ramp:] = ext_N; v[L_c + ramp:] = ext_v
    return ops, N, v, L_c


def cheb_step(h, X, dt, w=None, tol=1e-14):
    import page_ops_v6 as OPS
    from scipy.special import jv
    w = w or float(abs(h).sum(axis=1).max()) * 1.0001
    z = w * dt
    Kc, _ = OPS.cheb_order(z, tol)
    kk = np.arange(Kc + 1)
    ck = jv(kk, z) * ((-1j) ** kk) * np.where(kk == 0, 1.0, 2.0)
    T0 = X; T1 = (h @ X) / w
    Y = ck[0] * T0 + ck[1] * T1
    for k in range(2, Kc + 1):
        T2 = 2.0 * (h @ T1) / w - T0
        Y = Y + ck[k] * T2
        T0, T1 = T1, T2
    return Y


def window_control(n_lev, ext, T_H, slide_TH=2.0, final_factor=0.05, tau=None, dt=None, Lc_per_level=8, eta0=0.05, cap_len=200):
    """one run of the control. ext: (R, N, v) of the record's exterior beyond R_t (sites), with an absorber over its last cap_len sites.
    The cavity clock N_c = 2 T_H L_c/(pi n_lev) puts about n_lev levels (two valleys) in (0, T_H). Prepared: the cavity-block eigenvectors with
    e in (T_H, (1 + slide) T_H) filled, and one hole (the highest cavity level in (-(1 + slide) T_H, -T_H) left empty). The slide: the cavity
    clock N_c(t) = N_c0 (1 - (1 - final_factor) t/tau) compresses the cavity spectrum toward zero (Killing energy scales with the clock), so
    the filled levels traverse the window from above and the hole the mirror window from below. Returns the emitted number, the per-level
    escape fractions, the hole's exit fraction."""
    import pt3b_kernels as KR
    L_c = max(32, int(Lc_per_level * n_lev))
    N_c0 = 2.0 * T_H * L_c / (math.pi * n_lev)
    eR, eN, ev = ext
    ops, N, v, nc = cavity_chain(L_c, N_c0, eR, eN, ev)
    eta = np.zeros(ops.n); eta[-cap_len:] = eta0 * (np.arange(1, cap_len + 1) / cap_len) ** 2
    eta2 = np.repeat(eta, 2)
    cav = ops.site_mask(L_c + 0.5)
    h0 = ops.h(N, v)
    ew, Uw, _ = KR.interior_window(h0, cav, 0.0, T_H)
    ef, Uf, _ = KR.interior_window(h0, cav, T_H, (1 + slide_TH) * T_H)
    eh, Uh, _ = KR.interior_window(h0, cav, -(1 + slide_TH) * T_H, -T_H)
    X = Uf.copy()
    Hh = Uh[:, [int(np.argmax(eh))]].copy() if len(eh) else None
    tau = tau or 50.0 * L_c / N_c0
    w = float(abs(h0).sum(axis=1).max()) * 1.0001
    dt = dt or 200.0
    nst = int(math.ceil(tau / dt)); dt = tau / nst
    d = np.exp(-0.5 * dt * eta2)
    for k in range(nst):
        tm = (k + 0.5) * dt
        Nt = N.copy(); Nt[:L_c] = N_c0 * (1 - (1 - final_factor) * tm / tau)
        ht = ops.h(Nt, v)
        X = d[:, None] * cheb_step(ht, d[:, None] * X, dt, w)
        if Hh is not None:
            Hh = d[:, None] * cheb_step(ht, d[:, None] * Hh, dt, w)
    kept = np.real((np.abs(X) ** 2).sum(axis=0))
    in_cav_end = np.real((np.abs(X[cav]) ** 2).sum(axis=0))
    esc = 1.0 - in_cav_end
    out = {"n_lev_target": n_lev, "n_window_measured": int(len(ew)), "L_c": L_c, "N_c0": N_c0, "n_filled": int(len(ef)), "tau": tau, "dt": dt,
           "filled_e_over_TH": [float(x) for x in ef / T_H], "emitted_number_collected": float(np.sum(1 - kept)), "escaped_from_cavity": float(np.sum(esc)),
           "per_level_escape": [float(x) for x in esc], "escape_mean": float(np.mean(esc)) if len(esc) else None,
           "escape_max_over_mean": float(np.max(esc) / max(np.mean(esc), 1e-300)) if len(esc) else None,
           "escape_gini": float(np.sum(np.abs(esc[:, None] - esc[None, :])) / (2 * len(esc) ** 2 * max(np.mean(esc), 1e-300))) if len(esc) else None}
    if Hh is not None:
        out["hole_exit"] = float(1.0 - np.real((np.abs(Hh[cav]) ** 2).sum()))
        out["hole_e_over_TH"] = float(np.max(eh) / T_H)
    return out


# ================================================================== reviewer B pc13 1.7: the finite Hermitian interior block on the native chain port
def window_control_reviewb(n_w, L, prof, arm="mixing", slide_rate=0.0, G2_over_T0=0.1, T_obs_over_L=200.0, n_bins=20, seed=7, kappa=1,
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
    return {"n_w": n_w, "arm": arm, "slide_rate": slide_rate, "T0": T0, "G_coupling": Gc, "rho_port": rho, "T_obs_over_L": T_obs_over_L,
            "emitted_number_supply": float(np.real(np.trace(Gs[-1][1][np.ix_(sup, sup)]))), "depletion_spectrum_end": [float(x) for x in p_end],
            "largest_over_median": float(p_end[0] / max(np.median(p_end), 1e-300)), "n_p_above_1e-2": int(np.sum(p_end > 1e-2)),
            "depletion_diag_end": [float(np.real(Gs[-1][1][j, j])) for j in sup], "return_S_diag": [[abs(c) for c in row] for row in S_diag],
            "paired_mode_rows": pair_rows, "hole_exit": hole_exit, "t_over_L": [float(t / L) for t in ts]}
