#!/usr/bin/env python
"""engine_hf_cluster_v1.py (lead's draft; SPEC_final4b_engine.md sec. 11.6(1)): task 8's control, the small spherical cluster.
One radial chain (channel kappa, n_s shells, 2 n_s modes; the real form of the chain operator: on-site m SZ - (kappa/R_n) SX, the sine hop
as the real antisymmetric block [[0, -1/2], [1/2, 0]] from n to n + 1, unitarily equivalent to engine_ops_v1.chain_parts at v = 0) with the
gravitational pair interaction at first order in G in its Newtonian monopole form between the site-symmetrized energy densities
Hcal_n = c^dag hcal_n c, hcal_n = (1/2){P_n, h}:
    H_cl = c^dag h c + (1/2) sum_{n,n'} V_{nn'} :Hcal_n: :Hcal_n':,   V_{nn'} = -G/max(R_n, R_n'),   :Hcal_n: = Hcal_n - <Hcal_n>_ref,
the reference the sea of the Hartree operator h_H = h + sum_n Phi_n hcal_n (Phi_n = sum_n' V_nn' rho_n'), the symmetric ordering (the
double sum with the symmetric kernel).  Levels: Hartree (the engine's), Hartree-Fock (first order: E_x = (1/2) sum V F,
F_nn' = Tr[hcal_n C hcal_n' (1 - C)]; self-consistent with Sigma = (1/2) sum V hcal_n (1 - 2C) hcal_n' as the option), exact (Lanczos on the
sector with n_sea + N_f fermions; the vacuum sector likewise).  The mass-function fluctuation Var m(R_c) = sum_{n,n' <= c} F_nn' (Hartree)
and <m^2> - <m>^2 (exact).  The ordering variant (the Wick-normal-ordered product) is the option --ordering wick.
Usage: python engine_hf_cluster_v1.py --ns 7 --m 0.5 --kappa 1 --G 0.02 0.06 0.2 --Nf 1 2 3 4 --out <dir> --tag <tag>"""
import argparse
import json
import os
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

SX = np.array([[0.0, 1.0], [1.0, 0.0]]); SZ = np.diag([1.0, -1.0])


def chain_real(ns, m, kappa):
    R = np.arange(ns, dtype=float) + 1.0
    h = np.zeros((2 * ns, 2 * ns))
    for n in range(ns):
        h[2 * n:2 * n + 2, 2 * n:2 * n + 2] = m * SZ - (kappa / R[n]) * SX
    for n in range(ns - 1):
        h[2 * n:2 * n + 2, 2 * n + 2:2 * n + 4] = np.array([[0.0, -0.5], [0.5, 0.0]])
        h[2 * n + 2:2 * n + 4, 2 * n:2 * n + 2] = h[2 * n:2 * n + 2, 2 * n + 2:2 * n + 4].T
    hcal = []
    for n in range(ns):
        P = np.zeros((2 * ns, 2 * ns)); P[2 * n, 2 * n] = P[2 * n + 1, 2 * n + 1] = 1.0
        hcal.append(0.5 * (P @ h + h @ P))
    return h, hcal, R


def kernel_monopole(R, G):
    return -G / np.maximum(R[:, None], R[None, :])


# ------------------------------------------------------------------ one-body levels
def sea_and_filled(hH, Nf):
    e, U = eigh(hH)
    neg = e < 0
    C0 = U[:, neg] @ U[:, neg].T
    pos_idx = np.nonzero(~neg)[0][:Nf]
    Cm = U[:, pos_idx] @ U[:, pos_idx].T
    return C0, Cm, e


def hartree(h, hcal, V, Nf, alpha=0.5, tol=1e-12, maxit=500):
    """the engine's Hartree level: h_H = h + sum_n Phi_n hcal_n, rho_n = Tr[(C - C_0) hcal_n], C_0 the sea of h_H, C = C_0 + N_f lowest positive;
    E_H = Tr[h (C - C_0)] + (1/2) rho V rho (the sea's own energy not counted: the declared subtraction)."""
    ns = len(hcal)
    Phi = np.zeros(ns)
    for it in range(maxit):
        hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
        C0, Cm, e = sea_and_filled(hH, Nf)
        rho = np.array([np.trace(Cm @ hcal[n]) for n in range(ns)])
        Phi_new = V @ rho
        d = np.max(np.abs(Phi_new - Phi)); Phi = Phi + alpha * (Phi_new - Phi)
        if d < tol:
            break
    hH = h + sum(Phi[n] * hcal[n] for n in range(ns))
    C0, Cm, e = sea_and_filled(hH, Nf)
    rho = np.array([np.trace(Cm @ hcal[n]) for n in range(ns)])
    C0_free = sea_and_filled(h, 0)[0]
    return {"Phi": Phi, "rho": rho, "C0": C0, "Cm": Cm, "C": C0 + Cm, "E_kin": float(np.trace(h @ Cm)), "E_pot": float(0.5 * rho @ V @ rho),
            "E_H": float(np.trace(h @ Cm) + 0.5 * rho @ V @ rho), "sea_distortion": float(np.trace(h @ (C0 - C0_free))), "it": it + 1,
            "e_ref": np.array([np.trace(C0 @ hcal[n]) for n in range(ns)])}


def connected(hcal, C, Cm=None):
    """F_nn' = Tr[hcal_n C hcal_n' (1 - C)]; with Cm given, the matter's part uses C -> Cm on the left (k in the positive band)."""
    ns = len(hcal); F = np.zeros((ns, ns)); left = C if Cm is None else Cm
    for n in range(ns):
        A = hcal[n] @ left
        for n2 in range(ns):
            F[n, n2] = np.trace(A @ hcal[n2] @ (np.eye(len(C)) - C))
    return F


def fock_energy(hcal, V, C, Cm):
    F_all = connected(hcal, C); F_mat = connected(hcal, C, Cm)
    return {"E_x": float(0.5 * np.sum(V * F_all)), "E_x_matter": float(0.5 * np.sum(V * F_mat)), "E_x_sea": float(0.5 * np.sum(V * (F_all - F_mat))),
            "Var_m_profile": np.cumsum(np.cumsum(F_all, axis=0), axis=1).diagonal().tolist(),
            "Var_m_matter_profile": np.cumsum(np.cumsum(F_mat, axis=0), axis=1).diagonal().tolist()}


def hartree_fock_sc(h, hcal, V, Nf, start, alpha=0.3, tol=1e-11, maxit=800):
    """BUILDER (option): the self-consistent HF with the Fock operator Sigma = (1/2) sum V_nn' hcal_n (1 - 2C) hcal_n'; returns the HF state and
    E_HF = Tr[h (C - C_0)] + (1/2) rho V rho + E_x[C] with C_0 the HF sea.  Convergence may need Anderson mixing; report the iteration count."""
    raise NotImplementedError


# ------------------------------------------------------------------ the sector's Fock space
def sector_basis(nmodes, K):
    return np.array(sorted(sum(1 << i for i in c) for c in combinations(range(nmodes), K)), dtype=np.int64)


def popcount(x):
    """builder: the bit count by shifts on uint64 (the draft's SWAR constants overflow numpy's int64)."""
    x = np.asarray(x).astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.int64)
    one = np.uint64(1)
    while np.any(x):
        c += (x & one).astype(np.int64)
        x = x >> one
    return c


def hop_matrix(states, a, b):
    """the sector matrix of c_a^dag c_b with the Jordan-Wigner sign (-1)^(occupied modes strictly between a and b)."""
    idx = np.arange(len(states)); dim = len(states)
    if a == b:
        occ = ((states >> a) & 1).astype(float)
        return sp.csr_matrix((occ, (idx, idx)), shape=(dim, dim))
    ok = (((states >> b) & 1) == 1) & (((states >> a) & 1) == 0)
    s = states[ok]
    lo, hi = min(a, b), max(a, b)
    mask = np.int64(((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
    sign = 1.0 - 2.0 * (popcount(s & mask) % 2)
    t = s ^ (1 << b) ^ (1 << a)
    j = np.searchsorted(states, t)
    return sp.csr_matrix((sign, (j, idx[ok])), shape=(dim, dim))


def one_body_sector(states, x, hops):
    """sum_ab x_ab c_a^dag c_b on the sector; hops the cache of hop matrices."""
    nm = x.shape[0]; out = None
    for a in range(nm):
        for b in range(nm):
            if abs(x[a, b]) > 0:
                key = (a, b)
                if key not in hops:
                    hops[key] = hop_matrix(states, a, b)
                out = x[a, b] * hops[key] if out is None else out + x[a, b] * hops[key]
    return out.tocsr()


def exact_sector(h, hcal, V, e_ref, K, ordering="symmetric", C0=None):
    """the lowest state of H_cl on the sector with K fermions by Lanczos; the matvec x -> h1 x + (1/2) sum_n B_n (D_n x), B_n = A_n - e_ref_n,
    D_n = sum_n' V_nn' B_n'.  ordering 'wick' (BUILDER): subtract the one-body contraction term Sigma[C0] = (1/2) sum V hcal_n (1 - 2 C0) hcal_n'
    and the c-number (1/2) sum V Tr[hcal_n C0 hcal_n' (1 - C0)] so that the product is Wick-normal-ordered against the reference."""
    nm = h.shape[0]; states = sector_basis(nm, K); dim = len(states); hops = {}
    h1 = one_body_sector(states, h, hops)
    if ordering == "wick":
        if C0 is None:
            raise ValueError("the Wick variant needs the reference C0")
        Sig = 0.5 * sum(V[n, n2] * hcal[n] @ (np.eye(nm) - 2.0 * C0) @ hcal[n2] for n in range(len(hcal)) for n2 in range(len(hcal)))
        h1 = h1 - one_body_sector(states, Sig, hops)
    B = [one_body_sector(states, hcal[n], hops) - e_ref[n] * sp.identity(dim, format="csr") for n in range(len(hcal))]
    D = [sum(V[n, n2] * B[n2] for n2 in range(len(hcal))).tocsr() for n in range(len(hcal))]

    def mv(x):
        y = h1 @ x
        for n in range(len(hcal)):
            y = y + 0.5 * (B[n] @ (D[n] @ x))
        return y
    op = LinearOperator((dim, dim), matvec=mv, dtype=float)
    if dim <= 2000:                                              # dense check at small dimension
        M = np.column_stack([mv(np.eye(dim)[:, i]) for i in range(dim)])
        w, U = eigh(0.5 * (M + M.T)); E0, psi = float(w[0]), U[:, 0]
    else:
        w, U = eigsh(op, k=1, which="SA", tol=1e-12); E0, psi = float(w[0]), U[:, 0]
    # the mass function's moments in the exact state: m(R_c) = sum_{n <= c} A_n (the un-subtracted operator; <m> and Var m)
    A = [B[n] + e_ref[n] * sp.identity(dim, format="csr") for n in range(len(hcal))]
    mean, var = [], []
    acc = None
    for n in range(len(hcal)):
        acc = A[n] if acc is None else acc + A[n]
        mx = acc @ psi
        mean.append(float(psi @ mx)); var.append(float(mx @ mx - (psi @ mx) ** 2))
    return {"E0": E0, "dim": dim, "m_mean_profile": mean, "Var_m_profile": var}


def run(a):
    os.makedirs(a.out, exist_ok=True)
    h, hcal, R = chain_real(a.ns, a.m, a.kappa)
    nm = 2 * a.ns; n_sea = int(np.sum(eigh(h)[0] < 0))
    results = []
    for G in a.G:
        V = kernel_monopole(R, G)
        H0 = hartree(h, hcal, V, 0)                              # the vacuum sector at the Hartree level: the free sea
        ex0 = exact_sector(h, hcal, V, H0["e_ref"], n_sea, a.ordering, H0["C0"])
        f0 = fock_energy(hcal, V, H0["C"], H0["Cm"])
        row = {"G": G, "n_sea": n_sea, "vacuum": {"E_exact": ex0["E0"], "E_kin_sea": float(np.trace(h @ H0["C0"])), "E_x_sea": f0["E_x"],
                                                 "Var_m_sea_profile": f0["Var_m_profile"], "Var_m_exact_profile": ex0["Var_m_profile"]}, "Nf": []}
        for Nf in a.Nf:
            H = hartree(h, hcal, V, Nf)
            fx = fock_energy(hcal, V, H["C"], H["Cm"])
            ex = exact_sector(h, hcal, V, H["e_ref"], n_sea + Nf, a.ordering, H["C0"])
            E_full_H = float(np.trace(h @ H["C"])) + H["E_pot"]                      # the full variational Hartree energy of the sector
            E_full_HF = E_full_H + fx["E_x"]
            E_full_H0 = float(np.trace(h @ H0["C0"])); E_full_HF0 = E_full_H0 + f0["E_x"]
            grav_H = H["E_pot"]; grav_HF = H["E_pot"] + fx["E_x_matter"]
            dE_ex = ex["E0"] - ex0["E0"]; dE_HF = E_full_HF - E_full_HF0; dE_H = E_full_H - E_full_H0
            row["Nf"].append({"Nf": Nf, "E_H_engine": H["E_H"], "E_kin": H["E_kin"], "E_pot": H["E_pot"], "E_x_matter": fx["E_x_matter"],
                              "E_x_sea": fx["E_x_sea"], "sea_distortion": H["sea_distortion"], "E_exact_minus_vacuum": dE_ex,
                              "E_HF_full_minus_vacuum": dE_HF, "E_H_full_minus_vacuum": dE_H,
                              "ratio_HF_over_H_gravitational": grav_HF / grav_H if grav_H else None,
                              "ratio_HF_over_H_full": dE_HF / dE_H if dE_H else None, "ratio_exact_over_HF_full": dE_ex / dE_HF if dE_HF else None,
                              "fluct_H_matter_at_collar": float(np.sqrt(max(fx["Var_m_matter_profile"][-1], 0.0)) / (H["rho"].sum() or 1.0)),
                              "fluct_exact_at_collar": float(np.sqrt(max(ex["Var_m_profile"][-1] - ex0["Var_m_profile"][-1], 0.0)) /
                                                             ((ex["m_mean_profile"][-1] - ex0["m_mean_profile"][-1]) or 1.0)),
                              "Var_m_H_profile": fx["Var_m_profile"], "Var_m_exact_profile": ex["Var_m_profile"], "hartree_it": H["it"], "dim": ex["dim"]})
            print("G %.3g Nf %d  E_pot %.6f  E_x(mat) %.6f  HF/H %.5f  exact/HF %.5f  fluct_H %.4f fluct_ex %.4f  dim %d" % (
                G, Nf, H["E_pot"], fx["E_x_matter"], row["Nf"][-1]["ratio_HF_over_H_gravitational"] or 0, row["Nf"][-1]["ratio_exact_over_HF_full"] or 0,
                row["Nf"][-1]["fluct_H_matter_at_collar"], row["Nf"][-1]["fluct_exact_at_collar"], ex["dim"]), flush=True)
        results.append(row)
    with open(os.path.join(a.out, "hf_cluster_%s.json" % a.tag), "w") as fh:
        json.dump({"script": "engine_hf_cluster_v1.py", "args": vars(a), "results": results}, fh, default=float)
    # BUILDER: the fits of E_x_matter/E_pot against N_f^(-1/2) and N_f^(-1) over a.Nf at each G; the --ordering wick rerun; the kappa = 2 cluster;
    # the self-consistent HF option; the Lanczos sizes (ns 10, Nf 6, 8) timed to one hour.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, default=7)
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--G", type=float, nargs="+", default=[0.02, 0.06, 0.2])
    ap.add_argument("--Nf", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--ordering", default="symmetric", choices=["symmetric", "wick"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
