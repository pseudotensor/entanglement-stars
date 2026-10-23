#!/usr/bin/env python
"""pt3b_kernels.py (busi; B' one-body kernels, both specs' readouts). Works for the cube (pt3b_cube) and the chains (pt3_core.ChainOps).

Retained space I: the sites inside the outer root R_t (reviewer A: "interior norm fraction inside the root"); the exterior E the rest; the
interface is the bond set crossing R_t (hopping range 2). Retained inputs f_alpha: the eigenvectors of the interior block h_II (Dirichlet at
R_t) with e in the window (0, T_H) (particle supply) and (-T_H, 0) (the exit window, reviewer A's H2 test), orthonormal by construction.

The port kernel by the absorbing-boundary dilation (reviewer B: "an equivalent unitary propagation construction"): with h - i eta, eta >= 0 beyond
R_a = 2 r_s, the absorbed amplitude sqrt(2 eta) psi is a unitary dilation into a collecting bath, and for Psi(T) = U_eta(T) F
  G(T) = 1 - Psi(T)^dag Psi(T) = int_0^T Psi^dag 2 eta Psi dt  (exactly: d/dt Psi^dag Psi = -Psi^dag 2 eta Psi)
is K^dag K of reviewer B's collected-output kernel K_{j alpha}(T) = <b_j|U(T)|f_alpha> with b_j the collected modes. Its eigenvalues p_alpha(T) are the
whole singular spectrum (escape probabilities of the retained modes' combinations), separately in the supply and exit windows (particle and
hole channels). The return kernel: S(t) = F^dag U_eta(t) F (the retained-space amplitude, its non-exponential part the return); the
retained-to-exterior-but-not-collected weight 1 - p - ||P_I Psi||^2 (in transit, or trapped outside the root) printed beside.
reviewer A's readouts from the same propagation: N_w = #inputs in (0, T_H) (and reviewer A's quasi-mode count with P_int > 1/2 on small boxes by
shift-invert of h - i eta, quasimodes()); N_acc(theta) = #{p_alpha(T_theta) > 1/2} at T_theta = ln 2 r_s/theta (an escape width Gamma r_s >
theta), theta = 1e-1, 1e-2, 1e-3, and in the exit window; the q_r distribution <alpha|q_r|alpha>.
Densities (reviewer B's correction): the count in the fixed omega/T_H window N_w (area scaling: ratio 4 per doubling of r_s/a), the dimensionless
density r_s^-1 dN/d omega = N_w/(r_s T_H) (also ratio 4), the raw density per unit energy dN/d omega = N_w/T_H (ratio 8), all printed;
D = N_w/(4 pi (r_s/a)^2) (reviewer A's D A/a^2 per field); two sizes give a two-point exponent only, labelled so."""
import math

import numpy as np
import scipy.sparse as sp


def interior_window(H_csr, imask, lo, hi, dense_max=6000, k_guess=64, zero_tol=1e-10, info=None):
    """eigenpairs of the interior block in (lo, hi): dense eigh below dense_max dimensions, else shift-invert eigsh about the window centre
    (k grown until the window is covered). Structural zero modes (|e| < zero_tol x the block's scale: the naive lattice Dirac operator's exact
    zero modes on an open cluster, which the declarations keep empty) are excluded and counted into info["n_zero"].
    Returns (e, U_full (d, m) zero outside I, dim_I)."""
    ii = np.where(imask)[0]
    HII = H_csr[ii][:, ii]
    if len(ii) <= dense_max:
        e, V = np.linalg.eigh(HII.toarray())
        sel = (e > lo) & (e < hi)
        e, V = e[sel], V[:, sel]
    else:
        from scipy.sparse.linalg import eigsh
        c = 0.5 * (lo + hi)
        k = k_guess
        while True:
            e, V = eigsh(HII.tocsc(), k=min(k, len(ii) - 2), sigma=c, which="LM")
            if e.min() < lo and e.max() > hi or k >= len(ii) - 2:
                break
            k *= 2
        sel = (e > lo) & (e < hi)
        o = np.argsort(e[sel])
        e, V = e[sel][o], V[:, sel][:, o]
    scale = float(abs(HII).sum(axis=1).max()) if HII.nnz else 1.0
    z = np.abs(e) < zero_tol * scale
    if info is not None:
        info["n_zero"] = info.get("n_zero", 0) + int(z.sum())
    e, V = e[~z], V[:, ~z]
    U = np.zeros((H_csr.shape[0], len(e)), complex)
    U[ii] = V
    return e, U, len(ii)


def quasimodes(H_csr_eta, k, sigma=0.0):
    """small boxes: k eigenpairs of the non-Hermitian h - i eta nearest sigma (shift-invert Arnoldi)."""
    from scipy.sparse.linalg import eigs
    w, V = eigs(H_csr_eta.tocsc(), k=k, sigma=sigma, which="LM")
    o = np.argsort(w.real)
    return w[o], V[:, o]


def residues(w, V, f):
    """the biorthogonal residues of an input f on the quasi-modes: f = sum_j (l_j^dag f) V_j, l_j the left eigenvectors (rows of V^-1); returns
    |(l_j^dag f) <f|V_j>| (the weight of mode j in the return signal <f|U(t)|f>) — the mode an input belongs to, robust to non-orthogonality."""
    Wl = np.linalg.inv(V)
    return np.abs((Wl @ f) * (V.conj().T @ f))


def mode_table(w, V, imask, Jop=None, qop=None):
    """per quasi-mode: Re w, Gamma = -2 Im w, P_int, the flux Gamma_far = <J> per unit norm (J = i[h, Theta] outward at the far sphere), <q_r>."""
    nrm = np.real(np.einsum("ij,ij->j", V.conj(), V))
    P_int = np.real(np.einsum("ij,ij->j", V[imask].conj(), V[imask])) / nrm
    out = {"Re": w.real, "Gamma": -2 * w.imag, "P_int": P_int}
    if Jop is not None:
        out["Gamma_far"] = np.real(np.einsum("ij,ij->j", V.conj(), Jop @ V)) / nrm
    if qop is not None:
        out["q_r"] = np.real(np.einsum("ij,ij->j", V.conj(), qop @ V)) / nrm
    return out


def flux_operator(H_herm, mask_out):
    """J = i[h, Theta_out] (the outward energy-number current through the surface between mask_out and its complement)."""
    Th = sp.diags(mask_out.astype(float))
    return (1j * (H_herm @ Th - Th @ H_herm)).tocsr()


def propagate_kernel(step, F, T_list, dt, imask, record_S=None):
    """F (d, K) inputs; step(X, dt) -> U_eta(dt) X on (d, K) numpy (the caller wraps the cube or chain). At each T in T_list: G = 1 - Psi^dag Psi
    and its eigenvalues p, the retained weight ||P_I Psi||^2 per input, the return matrix S = F^dag Psi; record_S: also S at every step."""
    Psi = F.copy()
    t = 0.0
    out = []
    S_hist = []
    for T in sorted(T_list):
        while t < T - 1e-12:
            h = min(dt, T - t)
            Psi = step(Psi, h)
            t += h
            if record_S:
                S_hist.append((t, F.conj().T @ Psi))
        Gm = np.eye(F.shape[1]) - Psi.conj().T @ Psi
        Gm = 0.5 * (Gm + Gm.conj().T)
        p = np.linalg.eigvalsh(Gm)
        ret = np.real(np.einsum("ij,ij->j", Psi[imask].conj(), Psi[imask]))
        out.append({"T": T, "p": p, "trace_G": float(np.real(np.trace(Gm))), "retained": ret, "S": F.conj().T @ Psi,
                    "unphysical_p": int(np.sum((p < -1e-10) | (p > 1 + 1e-10)))})
    return out, S_hist


def readouts(e_in, kern, T_H, rs, thetas=(1e-1, 1e-2, 1e-3)):
    """both specs' readouts from one propagation: kern = propagate_kernel output for the supply-window inputs (e_in > 0) or the exit window."""
    res = {"N_w": int(len(e_in)), "N_w_over_TH_rs": len(e_in) / (T_H * rs), "dN_domega": len(e_in) / T_H, "D": len(e_in) / (4 * math.pi * rs * rs),
           "N_acc": {}, "spectra": {}}
    byT = {round(k["T"], 9): k for k in kern}
    for th in thetas:
        T = math.log(2.0) * rs / th
        k = byT.get(round(T, 9))
        if k is not None:
            res["N_acc"]["%g" % th] = int(np.sum(k["p"] > 0.5))
    for k in kern:
        res["spectra"]["%.6g" % k["T"]] = {"p_sorted_desc": [float(x) for x in np.sort(k["p"])[::-1]], "trace": k["trace_G"],
                                          "retained_mean": float(np.mean(k["retained"])) if len(k["retained"]) else None,
                                          "unphysical_p": k["unphysical_p"]}
    return res


def T_list_for(rs, thetas=(1e-1, 1e-2, 1e-3), extra=()):
    return sorted(set([math.log(2.0) * rs / th for th in thetas] + list(extra)))


def two_point_exponent(a1, a2, x1, x2):
    """the effective exponent of a in x between two sizes (two points only: not an asymptotic exponent)."""
    if a1 <= 0 or a2 <= 0:
        return None
    return math.log(a2 / a1) / math.log(x2 / x1)


# ================================================================================= pc13 readouts
def collected_entropies(G_list, C_in, eps=0.0):
    """the collected output's nested time-bin algebra in the absorber's dilation: bath modes in different time bins are orthogonal by support, so
    for retained inputs with one-body covariance C_in (K x K) the collected output up to bin n has the nonzero spectrum of
    C^(1/2) G(t_n) C^(1/2) (G the cumulative escape Gram = A_n^dag A_n), and bin n alone that of C^(1/2) dG_n C^(1/2). Returns rows with S(R_n),
    S(b_n), S(b_n|R_{n-1}) and certified intervals (pt3_core.cov_entropy at eigenvalue error eps)."""
    import pt3_core as K3
    w, V = np.linalg.eigh(0.5 * (C_in + C_in.conj().T))
    Ch = (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T
    rows, prev = [], None
    Gp = np.zeros_like(G_list[0][1])
    for (t, G) in G_list:
        R = K3.cov_entropy(Ch @ G @ Ch, eps)
        b = K3.cov_entropy(Ch @ (G - Gp) @ Ch, eps)
        row = {"t": t, "S_R": R["S"], "S_R_lo": R["S_lo"], "S_R_hi": R["S_hi"], "S_b": b["S"], "valid": R["valid"] and b["valid"]}
        if prev is not None:
            row.update({"S_b_given_R": R["S"] - prev["S"], "S_b_given_R_hi": R["S_hi"] - prev["S_lo"], "S_b_given_R_lo": R["S_lo"] - prev["S_hi"]})
        rows.append(row); prev = R; Gp = G
    return rows


def matrix_pencil(sig, dt, n_poles=None, tol=1e-10):
    """poles of a sampled signal s(k dt) = sum_j c_j exp(-i w_j k dt) by the matrix pencil (Hankel SVD; Hua and Sarkar): returns w (complex,
    Re the energy, -2 Im the width), amplitudes c, and the singular values (the model order's evidence)."""
    s = np.asarray(sig, complex)
    n = len(s)
    Lp = n // 2
    Y = np.array([s[i:i + Lp + 1] for i in range(n - Lp)])
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    M = n_poles or int(np.sum(S > tol * S[0]))
    Vm = Vh[:M].T                                       # the rows of Vh span the signal's shift-invariant row space (no conjugate)
    V1, V2 = Vm[:-1], Vm[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    w = 1j * np.log(z) / dt
    A = np.vander(z, n, increasing=True).T
    c = np.linalg.lstsq(A, s, rcond=None)[0]
    o = np.argsort(-np.abs(c))
    return w[o], c[o], S


def pc13_readouts(e_in, G_by_T, T0, L, q_in=None, thetas=(1e-1, 1e-2, 1e-3, 1e-4), widths=None, p_thresh=(1e-1, 1e-2, 1e-3)):
    """both specs' readouts for one window (supply (0, T0) or exit (-T0, 0)): N_w; counts in (0, T0) and (0, 3 T0) (if e_in spans it); the escape
    spectrum p(T) at every stored T (reviewer B: whole spectrum, Tr K^dag K, counts above p thresholds, largest/median); reviewer A's N_acc(theta) from
    p(T_theta) > 1/2 at T_theta = ln 2 L/theta where T_theta was propagated, and from pole widths Gamma L > theta (widths given); theta(chi) =
    <aq> chi/60.8 with <aq> the window inputs' mean |<q_r>| (chi = a/L)."""
    e_in = np.asarray(e_in, float)
    res = {"N_w": int(np.sum((e_in > 0) & (e_in < T0)) if np.all(e_in >= 0) else np.sum((e_in < 0) & (e_in > -T0))),
           "N_w_3T0": int(np.sum(np.abs(e_in) < 3 * T0)), "spectra": {}, "N_acc_time": {}, "N_acc_poles": {}}
    if q_in is not None and len(q_in):
        aq = float(np.mean(np.abs(q_in)))
        res["aq_mean_abs"] = aq; res["aq_sign_mean"] = float(np.mean(np.sign(q_in)))
        res["theta_chi"] = aq * (1.0 / L) / 60.8
    for T, G in sorted(G_by_T.items()):
        p = np.sort(np.linalg.eigvalsh(0.5 * (G + G.conj().T)))[::-1]
        med = float(np.median(p)) if len(p) else 0.0
        res["spectra"]["%.6g" % (T / L)] = {"p_desc": [float(x) for x in p], "trace_KdagK": float(np.sum(p)),
                                           "counts_above": {"%g" % th: int(np.sum(p > th)) for th in p_thresh},
                                           "largest_over_median": float(p[0] / med) if med > 0 else None}
    ths = list(thetas) + ([res["theta_chi"]] if "theta_chi" in res else [])
    for th in ths:
        T = math.log(2.0) * L / th
        hit = [k for k in G_by_T if abs(k - T) <= 1e-6 * T]
        if hit:
            p = np.linalg.eigvalsh(0.5 * (G_by_T[hit[0]] + G_by_T[hit[0]].conj().T))
            res["N_acc_time"]["%g" % th] = int(np.sum(p > 0.5))
        if widths is not None:
            res["N_acc_poles"]["%g" % th] = int(np.sum(np.asarray(widths) * L > th))
    if widths is not None:
        h, edges = np.histogram(np.log10(np.maximum(np.asarray(widths) * L, 1e-12)), bins=np.arange(-8, 1.01, 0.5))
        res["width_histogram_log10_GammaL"] = {"edges": edges.tolist(), "counts": h.tolist()}
    return res
