#!/usr/bin/env python
"""engine_ops_v1.py (lead's first draft; SPEC_final4b_engine.md secs. 0 to 2): the lattice operators of the declared model and the
normal-ordered sources.  Conventions: D_i = (T_i^dag - T_i)/(2i), <x|D_i|x+e_i> = -i/2 (symbol sin p); Q_i = (1/2i)[X_i, D^2],
<x|Q_i|x+2e_i> = -i/4 (symbol sin(2p)/2); h_m[N, v] = (1/2){N, h} + (1/2){V, Q}; the site density of a one-body operator A in orbitals Phi
(columns) with occupations f is sum_k f_k Re(Phi_{k,x}^dag (A Phi_k)_x) = <(1/2){P_x, A}>.
Radial chain (channel kappa = j + 1/2 = 1, 2, ...): two components per site, R_n = n + 1, on-site m SZ + (kappa/R_n) SY, sine bond
-(i/2) SX, two-step bond -(i/4) 1; the weight of every chain mode is w_kappa = 16 kappa (8 valleys times 2j + 1; the chain's own two
valleys are the continuum channels +kappa and -kappa, spec 2.1).  Lattice units."""
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)
GPU = {"dev": None}


def spectrum(h):
    """dense Hermitian eigh; torch on GPU["dev"] when set and the matrix is large (flow_chart_v4's pattern)."""
    if GPU["dev"] is not None and h.shape[0] > 1500:
        import torch
        a = torch.as_tensor(h, device="cuda:%d" % GPU["dev"])
        e, Q = torch.linalg.eigh(a)
        out = e.cpu().numpy(), Q.cpu().numpy()
        del a, e, Q
        return out
    return eigh(h, driver="evr", check_finite=False)


# ------------------------------------------------------------------ radial chain
def chain_parts(nsites, m, kappa, half_offset=False):
    """h_chain (unit clock, no flow), Q_chain (the two-step current), R (site radii).  half_offset: the variant R_n = n + 1/2 (spec 2.3)."""
    R = np.arange(nsites, dtype=float) + (0.5 if half_offset else 1.0)
    n2 = 2 * nsites
    h = np.zeros((n2, n2), complex)
    for n in range(nsites):
        h[2 * n:2 * n + 2, 2 * n:2 * n + 2] = m * SZ + (kappa / R[n]) * SY
    for n in range(nsites - 1):
        h[2 * n:2 * n + 2, 2 * n + 2:2 * n + 4] = -0.5j * SX
    h = np.triu(h) + np.triu(h, 1).conj().T
    Q = np.zeros((n2, n2), complex)
    for n in range(nsites - 2):
        Q[2 * n:2 * n + 2, 2 * n + 4:2 * n + 6] = -0.25j * I2
    Q = Q + Q.conj().T
    return h, Q, R


def clocked(h, Q, N, v, ncomp=2):
    """h_m[N, v] = (1/2){N, h} + (1/2){V, Q}: bond (x, y) carries (N_x + N_y)/2 times h_xy and (v_x + v_y)/2 times Q_xy; on-site N_x h_xx."""
    Nd = np.repeat(np.asarray(N, float), ncomp)
    Vd = np.repeat(np.asarray(v, float), ncomp)
    return 0.5 * (Nd[:, None] * h + h * Nd[None, :]) + 0.5 * (Vd[:, None] * Q + Q * Vd[None, :])


def site_density(A, Phi, f=None, ncomp=2):
    """sum_k f_k Re(Phi_k^dag P_x A Phi_k) per site: the site-symmetrized density of A (eps_n for A = h_chain, J_n for A = Q_chain)."""
    AP = A @ Phi
    if f is not None:
        AP = AP * np.asarray(f)[None, :]
    d = np.real(np.sum(np.conj(Phi) * AP, axis=1))
    return d.reshape(-1, ncomp).sum(axis=1)


def fermi(e, EF, Ts):
    x = np.clip((np.asarray(e, float) - EF) / Ts, -700, 700)
    return 1.0 / (np.exp(x) + 1.0)


def sources_static(h, Q, Phi_pos, f, w=1.0):
    """form (i) of spec 1.2: the normal-ordered sources of the ground state plus occupied positive modes (the sea cancels identically)."""
    return w * site_density(h, Phi_pos, f), w * site_density(Q, Phi_pos, f)


def sources_dynamic(h, Q, Phi, f, Qneg, w=1.0):
    """form (ii): Tr[(C - C_0) hcal_n] with C from the propagated occupied orbitals Phi (f) and C_0 the current reference's negative
    eigenvectors Qneg (columns); never a hand-summed sea."""
    return (w * (site_density(h, Phi, f) - site_density(h, Qneg)),
            w * (site_density(Q, Phi, f) - site_density(Q, Qneg)))


def cut_current(hm, Phi, f, cut_site, ncomp=2):
    """the number current through the cut between sites cut-1 and cut: dN_in/dt = +2 Im Phi_in^H h_{in,out} Phi_out (the checked sign)."""
    c = ncomp * cut_site
    hio = hm[:c, c:]
    val = np.sum(np.conj(Phi[:c, :]) * (hio @ Phi[c:, :]), axis=0)
    return float(2.0 * np.imag(np.sum((np.asarray(f) if f is not None else 1.0) * val)))


def lowdin(Phi):
    S = Phi.conj().T @ Phi
    w_, U = eigh(S)
    return Phi @ (U * w_ ** -0.5) @ U.conj().T


# ------------------------------------------------------------------ three-dimensional lattice (the small-size validation; spec 2.4)
ALPHA = [np.block([[np.zeros((2, 2)), s], [s, np.zeros((2, 2))]]) for s in (SX, SY, SZ)]
BETA = np.diag([1.0, 1.0, -1.0, -1.0]).astype(complex)


def lattice3d_parts(L, m, periodic=False):
    """sparse h_D and Q_i (i = 0, 1, 2) on an L^3 box, 4 components per site (site index x = (i*L + j)*L + k, component index 4x + a).
    periodic only for the dispersion control."""
    nsite = L ** 3
    idx = lambda i, j, k: (i * L + j) * L + k
    rows, cols, vals = [], [], []
    rq = [[] for _ in range(3)]
    cq = [[] for _ in range(3)]
    vq = [[] for _ in range(3)]
    for i in range(L):
        for j in range(L):
            for k in range(L):
                x = idx(i, j, k)
                for a in range(4):
                    for b in range(4):
                        if BETA[a, b] != 0:
                            rows.append(4 * x + a); cols.append(4 * x + b); vals.append(m * BETA[a, b])
                for d, (di, dj, dk) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                    for step, blk, store in ((1, -0.5j * ALPHA[d], (rows, cols, vals)), (2, -0.25j * np.eye(4), (rq[d], cq[d], vq[d]))):
                        ii, jj, kk = i + step * di, j + step * dj, k + step * dk
                        if periodic:
                            ii, jj, kk = ii % L, jj % L, kk % L
                        elif not (ii < L and jj < L and kk < L):
                            continue
                        y = idx(ii, jj, kk)
                        for a in range(4):
                            for b in range(4):
                                if blk[a, b] != 0:
                                    store[0].extend([4 * x + a, 4 * y + b]); store[1].extend([4 * y + b, 4 * x + a])
                                    store[2].extend([blk[a, b], np.conj(blk[a, b])])   # BUILDER: extend (the draft's += on a tuple item raised)
    n = 4 * nsite
    h = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    Q = [sp.csr_matrix((vq[d], (rq[d], cq[d])), shape=(n, n)) for d in range(3)]
    return h, Q


def clocked3d(h, Q, N, v):
    """N per site (length L^3), v per site and direction (shape (L^3, 3)); returns the dense h_m (the validation sizes are dense-able)."""
    Nd = np.repeat(np.asarray(N, float), 4)
    hm = 0.5 * (sp.diags(Nd) @ h + h @ sp.diags(Nd))
    for d in range(3):
        Vd = np.repeat(np.asarray(v)[:, d], 4)
        hm = hm + 0.5 * (sp.diags(Vd) @ Q[d] + Q[d] @ sp.diags(Vd))
    return hm.toarray()


def shell_bin(L, dens, rmax=None):
    """bin a per-site density about the centre site into shells [n + 1/2, n + 3/2) around R_n = n + 1 (the centre site in shell 0)."""
    c = (L - 1) / 2
    g = np.arange(L) - c
    r = np.sqrt(g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2).ravel()
    n = np.maximum(np.round(r - 1.0), 0).astype(int)
    out = np.bincount(n, weights=dens, minlength=(rmax or int(n.max()) + 1))
    return out


if __name__ == "__main__":
    # BUILDER: J0 checks: Hermiticity; chain_parts at N = 1, v = 0 against flow_chart_v4.chain_hamiltonian(m, kappa/R, 0) (identical);
    # periodic 3D dispersion against +-sqrt(m^2 + sum sin^2) + v . sin(2p)/2 at uniform N, v; vacuum sources zero; form (i) = form (ii).
    h, Q, R = chain_parts(64, 0.2, 1.0)
    N = 1 - 0.3 * np.exp(-R ** 2 / 50.0)
    v = -0.4 * (R / 5) * np.exp(-R ** 2 / 50.0)
    hm = clocked(h, Q, N, v)
    print("herm", np.max(np.abs(hm - hm.conj().T)))
    e, U = spectrum(hm)
    pos = e > 0
    EF, Ts = 0.19, 0.002
    f = fermi(e[pos], EF, Ts)
    e1, J1 = sources_static(h, Q, U[:, pos], f)
    Phi = np.hstack([U[:, ~pos], U[:, pos]])
    ff = np.concatenate([np.ones((~pos).sum()), f])
    e2, J2 = sources_dynamic(h, Q, Phi, ff, U[:, ~pos])
    print("form(i) - form(ii)", np.max(np.abs(e1 - e2)), np.max(np.abs(J1 - J2)), "vacuum", np.max(np.abs(site_density(h, U[:, ~pos]) - site_density(h, U[:, ~pos]))))
