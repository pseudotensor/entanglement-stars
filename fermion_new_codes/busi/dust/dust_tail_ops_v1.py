#!/usr/bin/env python
"""dust_tail_ops_v1.py (busi lane, reviewer A; the shed dust's second route, inbox #48 item 2). The lane's own construction of the
radial-chain operators and readouts, written from the definitions and not from the solve lane's code; no file I/O here.

Conventions (lattice units a = 1; a site i = 0 .. n-1 at radius R[i]; 2 spinor components per site, component index 2 i + a, a in {0, 1}):
- h_0 (the flat chain of one radial channel kappa): on-site block m sigma_z + (kappa / R_i) sigma_y; the hop from site i to i+1 carries the
  block -(i/2) sigma_x (element [2i+a, 2(i+1)+b] = -(i/2) (sigma_x)_ab) and the hop back its Hermitian conjugate.
- Q (the flow operator): the two-step hop from site i to i+2 with the block -(i/4) 1 and its conjugate back.
- h_m (the Killing Hamiltonian at the fields N, v): (1/2){N, h_0} + (1/2){v, Q}, N and v diagonal per site (repeated on both components).
  The bond blocks of h_m are therefore (N_i + N_{i+1})/2 (-(i/2) sigma_x) and (v_i + v_{i+2})/2 (-(i/4) 1); the on-site block N_i h_0[i, i].
- A state is a set of columns c_j with occupations f_j: C = sum_j f_j c_j c_j^dagger (the covariance <c_y^dagger c_x> = C_xy).
- Site density rho_x = sum_j f_j |c_j(x, .)|^2; the density of an operator A at site x: Re sum_j f_j c_j^dagger P_x A c_j (summed over the 2
  components); the symmetrized Killing density k_x uses A = h_m; the bare energy and flow densities eps_x and J_x use A = h_0 and A = Q, so that
  sum_x (N_x eps_x + v_x J_x) = sum_x k_x = Re Tr(h_m C) exactly (they differ site by site at the bonds only).
- The number current through the cut between sites p-1 and p (X = sites < p, Y = sites >= p): from i dC/dt = [h_m, C],
  dN_Y/dt = -2 Im Tr(h_XY C_YX) = -2 Im sum_j f_j c_jX^dagger h_XY c_jY (outward positive); only the pairs within the stencil's reach
  (1 site for h_0, 2 for Q) contribute.
- The exterior block's Killing spectrum: h_YY = Pi_Y h_m Pi_Y, eigenpairs (omega_l, phi_l); the occupation of level l by the columns is
  n_l = sum_j f_j |phi_l^dagger Pi_Y c_j|^2, so that sum_l n_l = N(Y) and sum_l omega_l n_l = Re Tr(h_YY C_YY) (the block energy).
- The structural fact used as a check: the antiunitary A = (-1)^i sigma_y K anticommutes with h_m for every N, v, so the spectrum is symmetric
  and the negative-energy projector has site density exactly 1; a sea propagated under such h_m keeps site density 1 and carries no number
  current through any cut; the number beyond a cut in C - P_- is then the dust's alone.
"""
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def chain_operators(R, m, kappa):
    """(h_0, Q) as sparse CSR matrices of size 2n x 2n for one channel kappa on the sites R (lattice units)."""
    R = np.asarray(R, float)
    n = len(R)
    one = np.ones(n)
    i = np.arange(n)
    rows = [2 * i, 2 * i + 1, 2 * i, 2 * i + 1]
    cols = [2 * i, 2 * i + 1, 2 * i + 1, 2 * i]
    vals = [m * one + 0j, -m * one + 0j, -1j * kappa / R, 1j * kappa / R]
    ib = np.arange(n - 1)
    # the hop i -> i+1: -(i/2) sigma_x has the elements (a, b) = (0, 1) and (1, 0); the hop back is the conjugate transpose
    rows += [2 * ib, 2 * ib + 1, 2 * ib + 3, 2 * ib + 2]
    cols += [2 * ib + 3, 2 * ib + 2, 2 * ib, 2 * ib + 1]
    vals += [-0.5j * np.ones(n - 1), -0.5j * np.ones(n - 1), 0.5j * np.ones(n - 1), 0.5j * np.ones(n - 1)]
    h0 = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n)).tocsr()
    iq = np.arange(n - 2)
    rq = np.concatenate([2 * iq, 2 * iq + 1, 2 * iq + 4, 2 * iq + 5])
    cq = np.concatenate([2 * iq + 4, 2 * iq + 5, 2 * iq, 2 * iq + 1])
    vq = np.concatenate([-0.25j * np.ones(n - 2), -0.25j * np.ones(n - 2), 0.25j * np.ones(n - 2), 0.25j * np.ones(n - 2)])
    Q = sp.coo_matrix((vq, (rq, cq)), shape=(2 * n, 2 * n)).tocsr()
    return h0, Q


def killing_h(R, m, kappa, N, v):
    """h_m = (1/2){N, h_0} + (1/2){v, Q} at the fields N, v (per site), sparse CSR."""
    h0, Q = chain_operators(R, m, kappa)
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2))
    vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ h0 + h0 @ Nd) + 0.5 * (vd @ Q + Q @ vd)).tocsr()


def site_density(cols, f=None):
    """rho_x = sum_j f_j |c_j(x, .)|^2 per site (an array of n)."""
    cols = np.asarray(cols)
    if cols.shape[1] == 0:
        return np.zeros(cols.shape[0] // 2)
    w = np.abs(cols) ** 2
    if f is not None:
        w = w * np.asarray(f, float)[None, :]
    return w.sum(axis=1).reshape(-1, 2).sum(axis=1)


def op_density(A, cols, f=None):
    """the density of the operator A: Re sum_j f_j c_j^dagger P_x A c_j per site."""
    cols = np.asarray(cols)
    if cols.shape[1] == 0:
        return np.zeros(cols.shape[0] // 2)
    AP = A @ cols
    w = np.real(np.conj(cols) * AP)
    if f is not None:
        w = w * np.asarray(f, float)[None, :]
    return w.sum(axis=1).reshape(-1, 2).sum(axis=1)


def cut_currents_all(N, v, cols, f=None):
    """the number current through every cut p = 1 .. n-1 (between sites p-1 and p; outward positive) from the bond blocks of h_m:
    F[p] = -2 Im (g1[p-1] + g2[p-2] + g2[p-1]) with g1[x] = sum_j f_j c_j(x)^dagger B1_x c_j(x+1), B1_x = (N_x + N_{x+1})/2 (-(i/2) sigma_x),
    and g2[x] = sum_j f_j c_j(x)^dagger B2_x c_j(x+2), B2_x = (v_x + v_{x+2})/2 (-(i/4) 1). Returns an array of n-1 (index p-1)."""
    cols = np.asarray(cols)
    n = cols.shape[0] // 2
    N = np.asarray(N, float); v = np.asarray(v, float)
    fw = np.ones(cols.shape[1]) if f is None else np.asarray(f, float)
    c0 = cols[0::2, :]; c1 = cols[1::2, :]                                    # the two components per site, (n, ncol)
    if cols.shape[1] == 0:
        return np.zeros(n - 1)
    s1 = ((np.conj(c0[:-1]) * c1[1:] + np.conj(c1[:-1]) * c0[1:]) * fw[None, :]).sum(axis=1)     # x = 0 .. n-2
    g1 = -0.5j * 0.5 * (N[:-1] + N[1:]) * s1
    s2 = ((np.conj(c0[:-2]) * c0[2:] + np.conj(c1[:-2]) * c1[2:]) * fw[None, :]).sum(axis=1)     # x = 0 .. n-3
    g2 = -0.25j * 0.5 * (v[:-2] + v[2:]) * s2
    F = np.zeros(n - 1, dtype=complex)
    F += g1                                      # cut p uses g1[p-1]
    F[1:] += g2                                  # cut p uses g2[p-2] for p >= 2
    F[:-1] += g2                                 # cut p uses g2[p-1] for p <= n-2
    return -2.0 * np.imag(F)


def cut_current_at(hm, cols, f, p, reach=2):
    """the same current through the cut p from the sparse h_m directly: -2 Im sum_j f_j c_jX^dagger h_XY c_jY, restricted to the rows and
    columns within `reach` sites of the cut (exact when reach covers the stencil)."""
    cols = np.asarray(cols)
    n2 = cols.shape[0]
    lo = max(2 * p - 2 * reach, 0); hi = min(2 * p + 2 * reach, n2)
    hxy = hm[lo:2 * p, 2 * p:hi].toarray()
    fw = np.ones(cols.shape[1]) if f is None else np.asarray(f, float)
    t = np.sum(fw * np.sum(np.conj(cols[lo:2 * p]) * (hxy @ cols[2 * p:hi]), axis=0))
    return float(-2.0 * np.imag(t))


def exterior_spectrum(hm, p):
    """the eigenpairs (omega, phi) of the exterior block h_YY, Y = sites >= p (dense Hermitian eigenproblem of size 2 (n - p))."""
    hY = hm[2 * p:, 2 * p:].toarray()
    hY = 0.5 * (hY + hY.conj().T)
    return sla.eigh(hY)


def occupations_in_basis(phi, colsY, f=None):
    """n_l = sum_j f_j |phi_l^dagger c_jY|^2 for the columns restricted to Y."""
    colsY = np.asarray(colsY)
    if colsY.shape[1] == 0:
        return np.zeros(phi.shape[1])
    A = phi.conj().T @ colsY
    w = np.abs(A) ** 2
    if f is not None:
        w = w * np.asarray(f, float)[None, :]
    return w.sum(axis=1)


def block_energy(hm, colsY, p, f=None):
    """Re sum_j f_j c_jY^dagger h_YY c_jY (the block energy, no cross-cut bonds)."""
    colsY = np.asarray(colsY)
    if colsY.shape[1] == 0:
        return 0.0
    hY = hm[2 * p:, 2 * p:]
    w = np.real(np.sum(np.conj(colsY) * (hY @ colsY), axis=0))
    if f is not None:
        w = w * np.asarray(f, float)
    return float(w.sum())


def full_spectrum(hm, vectors=False):
    """the eigenvalues (and eigenvectors) of the whole chain's h_m (dense)."""
    H = hm.toarray()
    H = 0.5 * (H + H.conj().T)
    return sla.eigh(H) if vectors else sla.eigvalsh(H)


def antiunitary_conjugate(hm):
    """U conj(h_m) U^dagger for U = diag((-1)^i sigma_y): the structural fact says this equals -h_m."""
    H = hm.toarray()
    n = H.shape[0] // 2
    U = sp.block_diag([((-1) ** i) * SY for i in range(n)]).toarray()
    return U @ np.conj(H) @ U.conj().T


def cut_index(R, Rc):
    """p = the number of sites with R <= Rc, so that Y = sites with R > Rc starts at index p (the first route's convention)."""
    return int(np.searchsorted(np.asarray(R, float), Rc, side="right"))
