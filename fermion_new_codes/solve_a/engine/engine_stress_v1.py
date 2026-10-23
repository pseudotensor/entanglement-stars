#!/usr/bin/env python
"""engine_stress_v1.py (builder analyst; engine spec 20.1, the stress declared by the lane [chosen; the owner's confirmation asked]): the
lattice matter's own momentum flux in a normal-ordered state, per channel of the radial chain (engine_collapse_v5.Chan: the flat chain
h = h_site + h_bond with h_site = m sigma_z + (kappa/R_n) sigma_y on site n and h_bond the sine bonds; Q the two-step momentum, symbol
sin 2p/2). For a set of columns c_j with weights f_j (the state's part: the dust's columns with their occupations; the sea's with +1 and
R2's with -1 for the pairs), with P_<=n the projector on the shells 0..n and Q_in^n = (1/2){P_<=n, Q}:
  F_{n+1/2} = -Tr[D i[h_bond, Q_in^n]] = 2 Im sum_j f_j sum_{s<=n} g_s(c_j), g_s = (1/2)[(h_bond c)_s^dag (Q c)_s + (Q h_bond c)_s^dag c_s],
      the outward flux of radial momentum through the face between shells n and n + 1;
  f_n = Tr[D (1/2){P_n, i[h_site, Q]}], the centrifugal force on shell n;
  b_n = Tr[D P_n sigma_z] (the mass-term density, before the volume), and <sigma_y>_n for the continuum form of S_t;
  the momentum-balance identity d<Q_in^n>/dt = -F_{n+1/2} + sum_{n'<=n} f_n' at unit clock and zero flow, checked by computing
      <i[h, Q_in^n]> = -2 Im sum_j f_j sum_{s<=n} g0_s with h in place of h_bond (the largest difference printed);
the shell values (lattice units, Delta R = 1): S_r at the faces F/(4 pi R_{n+1/2}^2), at the shells the mean of their faces; S_t,n =
f_n/(8 pi R_n) (its continuum form (kappa/R_n) <sigma_y>_n/(8 pi R_n^2) beside); rho_n = eps_n/(4 pi R_n^2), b_n/(4 pi R_n^2); the virial
check S_r + 2 S_t against rho - m b (exact for the continuum Dirac field, O((p a)^2) on the sine lattice). The smoothing: the shell totals
X_n^tot = 4 pi R_n^2 X_n smoothed by the Gaussian's l-projections K_l (l = 0, 2; i_2(z) = ((1 + 3/z^2) sinh z - 3 cosh z/z)/z, series
(z^2/15)(1 + z^2/14) below 1e-2): S_r,ell = [S_t]_0 + (1/3)[S_r - S_t]_0 + (2/3)[S_r - S_t]_2, S_t,ell = [S_t]_0 + (1/3)[S_r - S_t]_0
- (1/3)[S_r - S_t]_2; the trace by K_0 alone (the identity checked). Functions: stress_parts, shell_values, smooth_stress, kernels_l."""
import math

import numpy as np


def kernels_l(r, R, ell, l):
    """K_l(r, R) for l = 0, 2 (and 1): the density at r from a unit shell total at R, the Gaussian of width ell projected on the l-th
    harmonic; exponentially scaled (no overflow); r: (n_r,), R: (n_R,) -> (n_r, n_R)."""
    r = np.asarray(r, float)[:, None]; R = np.asarray(R, float)[None, :]
    Em = np.exp(-(r - R) ** 2 / (2.0 * ell ** 2)); Ep = np.exp(-(r + R) ** 2 / (2.0 * ell ** 2)); EA = np.exp(-(r ** 2 + R ** 2) / (2.0 * ell ** 2))
    z = r * R / ell ** 2
    small = z < 1e-2
    zs = np.where(small, 1.0, z)
    sh = 0.5 * (Em - Ep); ch = 0.5 * (Em + Ep)                       # sinh z and cosh z times exp(-(r^2 + R^2)/(2 ell^2))
    if l == 0:
        il = np.where(small, EA * (1.0 + z ** 2 / 6.0), sh / zs)
    elif l == 1:
        il = np.where(small, EA * (z / 3.0) * (1.0 + z ** 2 / 10.0), (ch - sh / zs) / zs)
    elif l == 2:
        il = np.where(small, EA * (z ** 2 / 15.0) * (1.0 + z ** 2 / 14.0), ((1.0 + 3.0 / zs ** 2) * sh - 3.0 * ch / zs) / zs)
    else:
        raise ValueError("l in (0, 1, 2)")
    return il / ((2.0 * math.pi) ** 1.5 * ell ** 3)


def split_chain(hs):
    """h = h_site + h_bond on the host (scipy CSR, 2 components per site): h_site the block-diagonal part, h_bond the rest."""
    import scipy.sparse as sp
    h = hs.tocoo()
    same = (h.row // 2) == (h.col // 2)
    hsite = sp.csr_matrix((h.data[same], (h.row[same], h.col[same])), shape=h.shape)
    hbond = sp.csr_matrix((h.data[~same], (h.row[~same], h.col[~same])), shape=h.shape)
    return hsite, hbond


class ChanStress:
    """the device operators of one channel for the stress: h, h_site, h_bond, Q (torch CSR), sigma_z and sigma_y signs."""

    def __init__(self, ch, torch, dev):
        self.torch, self.dev = torch, dev
        hsite, hbond = split_chain(ch.hs)
        csr = lambda A: torch.sparse_csr_tensor(torch.as_tensor(A.indptr, dtype=torch.int64), torch.as_tensor(A.indices, dtype=torch.int64),
                                                torch.as_tensor(A.data, dtype=torch.complex128), size=A.shape, device=dev)
        self.h = csr(ch.hs.tocsr()); self.hsite = csr(hsite); self.hbond = csr(hbond); self.Q = csr(ch.Qs.tocsr())
        n2 = ch.hs.shape[0]
        self.sz = torch.as_tensor(np.tile([1.0, -1.0], n2 // 2), device=dev, dtype=torch.float64)
        self.kappa = ch.kappa; self.w = ch.w; self.R = np.asarray(ch.R, float)


def stress_parts(cs, cols, f):
    """per site (host arrays): G (complex; the flux's cumulant summand), G0 (the same with h), fc (the centrifugal force), b (sigma_z), sy
    (sigma_y), eps (h), J (Q), all summed over the columns with the weights f (not yet the channel weight)."""
    torch = cs.torch
    if cols.shape[1] == 0:
        z = np.zeros(cols.shape[0] // 2)
        return {k: z.copy() for k in ("Gr", "Gi", "G0i", "fc", "b", "sy", "eps", "J")}
    fw = torch.as_tensor(np.asarray(f, float), device=cs.dev, dtype=torch.float64)
    A = cs.hbond @ cols; Qc = cs.Q @ cols; QA = cs.Q @ A
    H = cs.h @ cols; QH = cs.Q @ H
    site = lambda X: X.reshape(-1, 2, X.shape[1]).sum(dim=1)            # the two components of each site summed
    g = 0.5 * site(A.conj() * Qc + QA.conj() * cols)
    g0 = 0.5 * site(H.conj() * Qc + QH.conj() * cols)
    Sc = cs.hsite @ cols
    Xc = 1j * (cs.hsite @ Qc - cs.Q @ Sc)                                 # i[h_site, Q] c
    fcn = site((cols.conj() * Xc).real)
    b = site((cols.conj() * cols).real * cs.sz[:, None])
    # sigma_y on site: [[0, -i], [i, 0]]: <c|sigma_y|c> = 2 Im(conj(c_0) c_1)
    c0 = cols[0::2, :]; c1 = cols[1::2, :]
    sy = 2.0 * (c0.conj() * c1).imag
    eps = site((cols.conj() * H).real)
    Jd = site((cols.conj() * Qc).real)
    red = lambda X: (X * fw[None, :]).sum(dim=1).cpu().numpy()
    gw = (g * fw[None, :].to(g.dtype)).sum(dim=1).cpu().numpy(); g0w = (g0 * fw[None, :].to(g0.dtype)).sum(dim=1).cpu().numpy()
    return {"Gr": gw.real, "Gi": gw.imag, "G0i": g0w.imag, "fc": red(fcn), "b": red(b), "sy": red(sy), "eps": red(eps), "J": red(Jd)}


def combine(parts_list, weights):
    """the channel-weighted sum of stress_parts dicts."""
    out = None
    for p, w in zip(parts_list, weights):
        if out is None:
            out = {k: w * np.asarray(v, float) for k, v in p.items()}
        else:
            for k in out:
                out[k] = out[k] + w * np.asarray(p[k], float)
    return out


def shell_values(P, R, m, kappa_sum=None):
    """from the weighted per-site sums: the face flux F, the force f, S_r (faces and shells), S_t, its continuum form (needs the
    kappa-weighted sigma_y sum, passed as P['sy_k']), rho, b, the virial residual, and the momentum identity's largest difference."""
    R = np.asarray(R, float)
    F = 2.0 * np.cumsum(P["Gi"])                                         # F_{n+1/2}, n = 0 .. N-1 (the last face at the chain's end)
    ident = float(np.max(np.abs(-2.0 * np.cumsum(P["G0i"]) - (-F + np.cumsum(P["fc"])))))
    Rf = np.concatenate([R[:-1] + 0.5 * np.diff(R), [R[-1] + 0.5]])
    Sr_face = F / (4.0 * math.pi * Rf ** 2)
    Sr = 0.5 * (np.concatenate([[Sr_face[0]], Sr_face[:-1]]) + Sr_face)
    St = P["fc"] / (8.0 * math.pi * R)
    rho = P["eps"] / (4.0 * math.pi * R ** 2)
    bden = P["b"] / (4.0 * math.pi * R ** 2)
    out = {"F_face": F, "R_face": Rf, "S_r_face": Sr_face, "S_r": Sr, "S_t": St, "rho": rho, "b": bden, "virial_lhs": Sr + 2.0 * St, "virial_rhs": rho - m * bden,
           "identity_max_abs": ident}
    if "sy_k" in P:
        out["S_t_continuum"] = P["sy_k"] / R / (8.0 * math.pi * R ** 2)
    return out


def smooth_stress(Sr, St, R, ell, r_eval):
    """S_r,ell and S_t,ell at r_eval from the shell values by the K_0 and K_2 projections of the shell totals; the trace identity's error."""
    R = np.asarray(R, float)
    V = 4.0 * math.pi * R ** 2
    K0 = kernels_l(r_eval, R, ell, 0); K2 = kernels_l(r_eval, R, ell, 2)
    St0 = K0 @ (V * St); D0 = K0 @ (V * (Sr - St)); D2 = K2 @ (V * (Sr - St))
    Sr_l = St0 + D0 / 3.0 + 2.0 * D2 / 3.0
    St_l = St0 + D0 / 3.0 - D2 / 3.0
    tr0 = K0 @ (V * (Sr + 2.0 * St))
    scale = max(float(np.max(np.abs(tr0))), 1e-300)
    return Sr_l, St_l, float(np.max(np.abs(Sr_l + 2.0 * St_l - tr0))) / scale
