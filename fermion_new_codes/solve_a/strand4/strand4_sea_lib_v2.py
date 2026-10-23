"""Strand 4 Part A library (spec solve/specs/SPEC_strand4_controls.md; builder analyst): the paper's two-quartic half-filled sea
(03_collective eq. new-H, in full_numerics/hf_closure_3d.py's conventions, whose constants TI, G0 and uniform solver are imported
unchanged) under a clock N and a ruler s that vary along the first lattice axis only, at the direct-saddle (HS) and Hartree-Fock
(HF) levels, zero temperature, half filling, periodic L^3 box.  [paper, control]

Conventions (hf_closure_3d): 4 components per site, TI_i = -i/2 AL_i, G0 = sz (x) 1; h[x, x + e_i] = Q TI_i (+ h.c.);
b_{x,i} = <psi_{x+i}^dag TI_i psi_x>, sigma_x = tr(G0 D_xx), D_xx[a, b] = <c_{x,b}^dag c_{x,a}>;
X_{x,i} = tr(D_{x+i} TI_i D_x TI_i^dag), Y_x = tr(G0 D_x G0 D_x);
E_HS = sum_bonds (-2 t0 Re b - U_B |b|^2) + sum_x (m0 sigma - U_M sigma^2), E_Wick = E_HS + sum_bonds U_B X + sum_x U_M Y.
Clock and ruler (the spec's dictionary): every local term carries its N (sites N_x, bonds the midpoint (N_x + N_y)/2); the ruler
enters the hopping as (s_x s_y)^-1 and the quartics by the chosen weights:
  dictionary "spec": on-site quartic s_x^-6, bond quartic (s_x s_y)^-3;
  dictionary "p2"  : on-site quartic s_x^-2, bond quartic (s_x s_y)^-1 (the executed star's p = -2, read at the bond's midpoint N);
the mass term m0 sigma carries N_x only.  So, with weights, E = sum_bonds Nb [-2 t0 r Re b - U_B wB (|b|^2 - [X])]
+ sum_x N_x [m0 sigma - U_M wM (sigma^2 - [Y])] ([.] at the HF level), and the Fock operator (derivative of E) is
  hopping Q_{x,i} = Nb (t0 r + U_B wB b),  on-site N_x (m0 - 2 U_M wM sigma) G0
  + [HF] 2 N_x U_M wM G0 D_x G0 + sum_i (Nb wB U_B)_{x,i} TI_i^dag D_{x+i} TI_i + sum_i (Nb wB U_B)_{x-i,i} TI_i D_{x-i} TI_i^dag
(the paper's onsite() with weights).  The HS saddle is the same without the exchange blocks.
Layered solver: with N, s functions of x_1 only, the state keeps the translations along x_2, x_3; for each (k_2, k_3) the one-body
operator is a 4L x 4L matrix on the layers x_1 (direction-1 bonds between layers, directions 2, 3 as e^{i k} phases inside the
layer); the global aufbau fills the lowest 2 L^3 levels.  Checked against hf_closure_3d.uniform_solve at the flat point.
Version 2: dictionary "p2mid" (the lead's ruling after Part A): the quartics as "p2" and the hopping ruler weight of the executed code,
Q_b = Nbar_b/psibar_b^2 with psibar_b = (s_x + s_y)/2, in place of (s_x s_y)^-1 (identical on the in-layer bonds, where s_x = s_y).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FN = os.path.abspath(os.path.join(HERE, "..", "..", "..", "full_numerics"))
sys.path.insert(0, os.environ.get("FULL_NUMERICS", FN))
import hf_closure_3d as HC  # noqa: E402

TI, G0 = HC.TI, HC.G0


class LayeredSea:
    def __init__(self, L, t0, m0, UB, UM, dictionary="spec", tw=(0.0, 0.0, 0.0)):
        self.L, self.t0, self.m0, self.UB, self.UM = L, t0, m0, UB, UM
        self.dict = dictionary
        self.tw = tw
        k2 = (2 * np.pi * np.arange(L) + tw[1]) / L
        k3 = (2 * np.pi * np.arange(L) + tw[2]) / L
        self.K = np.array([(a, b) for a in k2 for b in k3])
        self.nk = len(self.K)
        self.nocc = 2 * L ** 3

    # ---------------------------------------------------------------- weights per layer
    def weights(self, N, s):
        L = self.L
        N, s = np.asarray(N, float), np.asarray(s, float)
        Np, sp = np.roll(N, -1), np.roll(s, -1)
        pq = -6.0 if self.dict == "spec" else -2.0
        Nb = np.stack([0.5 * (N + Np), N, N])                 # (3, L): direction 0 between layers, 1 and 2 inside
        if self.dict == "p2mid":
            r = np.stack([(0.5 * (s + sp)) ** -2, s ** -2, s ** -2])
        else:
            r = np.stack([1.0 / (s * sp), s ** -2, s ** -2])
        wB = np.stack([(s * sp) ** (pq / 2), s ** pq, s ** pq])
        wM = s ** pq
        return {"N": N, "Nb": Nb, "r": r, "wB": wB, "wM": wM, "L": L}

    # ---------------------------------------------------------------- one-body operator and state
    def H_k(self, Q, onsite, k):
        L = self.L
        H = np.zeros((L, 4, L, 4), dtype=complex)
        for x in range(L):
            blk = onsite[x].copy()
            for i, kk in ((1, k[0]), (2, k[1])):
                A = Q[i, x] * np.exp(1j * kk) * TI[i]
                blk = blk + A + A.conj().T
            H[x, :, x, :] += blk
            y = (x + 1) % L
            ph = np.exp(1j * self.tw[0]) if y == 0 else 1.0
            A = Q[0, x] * ph * TI[0]
            H[x, :, y, :] += A
            H[y, :, x, :] += A.conj().T
        return H.reshape(4 * L, 4 * L)

    def state(self, Q, onsite, degtol=1e-9):
        L = self.L
        ws, us = [], []
        for k in self.K:
            w, u = np.linalg.eigh(self.H_k(Q, onsite, k))
            ws.append(w)
            us.append(u)
        ws, us = np.array(ws), np.array(us)          # (nk, 4L), (nk, 4L, 4L)
        allw = np.sort(ws.reshape(-1))
        ef_lo, ef_hi = allw[self.nocc - 1], allw[self.nocc]
        occ = (ws <= ef_lo + degtol).astype(float)
        nfill = occ.sum()
        if nfill != self.nocc:                       # degenerate Fermi level: share the remaining charge equally
            edge = np.abs(ws - ef_lo) < degtol
            below = ((ws < ef_lo - degtol)).sum()
            occ = (ws < ef_lo - degtol).astype(float)
            occ[edge] = (self.nocc - below) / edge.sum()
        return {"w": ws, "u": us, "occ": occ, "gap": float(ef_hi - ef_lo), "nfill": float(occ.sum())}

    def fields(self, st):
        """b[i, x] and D[x] (4 x 4) from the occupied layer orbitals."""
        L, nk = self.L, self.nk
        u = st["u"].reshape(nk, L, 4, 4 * L)
        occ = st["occ"]
        D = np.einsum("kxan,kn,kxbn->xab", u, occ, u.conj()) / nk
        b = np.zeros((3, L), dtype=complex)
        for x in range(L):
            y = (x + 1) % L
            ph = np.exp(1j * self.tw[0]) if y == 0 else 1.0
            b[0, x] = np.conj(ph) * np.einsum("kan,ab,kbn,kn->", u[:, y].conj(), TI[0], u[:, x], occ) / nk
            for i, col in ((1, 0), (2, 1)):
                e = np.exp(-1j * self.K[:, col])
                b[i, x] = np.einsum("k,kan,ab,kbn,kn->", e, u[:, x].conj(), TI[i], u[:, x], occ) / nk
        return b, D

    # ---------------------------------------------------------------- functional and Fock operator
    def energy(self, b, D, W):
        """E_HS and E_Wick per box (sums over the L^3 sites: each layer quantity times L^2)."""
        L = self.L
        t0, m0, UB, UM = self.t0, self.m0, self.UB, self.UM
        sig = np.einsum("ab,xba->x", G0, D).real
        X = np.zeros((3, L))
        for i in range(3):
            Dn = np.roll(D, -1, axis=0) if i == 0 else D
            X[i] = np.einsum("xab,bc,xcd,da->x", Dn, TI[i], D, TI[i].conj().T).real
        Y = np.einsum("ab,xbc,cd,xda->x", G0, D, G0, D).real
        bond_hs = W["Nb"] * (-2 * t0 * W["r"] * b.real - UB * W["wB"] * np.abs(b) ** 2)
        site_hs = W["N"] * (m0 * sig - UM * W["wM"] * sig ** 2)
        bond_x = W["Nb"] * UB * W["wB"] * X
        site_x = W["N"] * UM * W["wM"] * Y
        nsl = L * L
        E_hs = nsl * float(bond_hs.sum() + site_hs.sum())
        E_x = nsl * float(bond_x.sum() + site_x.sum())
        return {"E_hs": E_hs, "E_x": E_x, "E_wick": E_hs + E_x, "sig": sig, "X": X, "Y": Y}

    def fock(self, b, D, W, exchange):
        L = self.L
        t0, m0, UB, UM = self.t0, self.m0, self.UB, self.UM
        sig = np.einsum("ab,xba->x", G0, D).real
        Q = W["Nb"] * (t0 * W["r"] + UB * W["wB"] * b)
        on = np.array([W["N"][x] * (m0 - 2 * UM * W["wM"][x] * sig[x]) * G0 for x in range(L)])
        if exchange:
            c = W["Nb"] * W["wB"] * UB
            for x in range(L):
                on[x] = on[x] + 2 * W["N"][x] * UM * W["wM"][x] * G0 @ D[x] @ G0
                for i in range(3):
                    xp = (x + 1) % L if i == 0 else x
                    xm = (x - 1) % L if i == 0 else x
                    on[x] = on[x] + c[i, x] * TI[i].conj().T @ D[xp] @ TI[i] + c[i, xm] * TI[i] @ D[xm] @ TI[i].conj().T
        return Q, on

    # ---------------------------------------------------------------- self-consistency
    def solve(self, N, s, exchange, x0=None, tol=1e-12, maxit=500, mmax=7, beta=0.5):
        L = self.L
        W = self.weights(N, s)

        def pack(b, D):
            return np.concatenate([b.real.ravel(), b.imag.ravel(), D.real.ravel(), D.imag.ravel()])

        def unpack(x):
            n3 = 3 * L
            b = (x[:n3] + 1j * x[n3:2 * n3]).reshape(3, L)
            D = (x[2 * n3:2 * n3 + 16 * L] + 1j * x[2 * n3 + 16 * L:]).reshape(L, 4, 4)
            return b, 0.5 * (D + D.conj().transpose(0, 2, 1))

        def F(x):
            b, D = unpack(x)
            Q, on = self.fock(b, D, W, exchange)
            st = self.state(Q, on)
            bn, Dn = self.fields(st)
            return pack(bn, Dn), st

        if x0 is None:
            Q, on = self.fock(np.zeros((3, L), complex), np.zeros((L, 4, 4), complex), W, False)
            st = self.state(Q, on)
            x = pack(*self.fields(st))
        else:
            x = x0.copy()
        Xh, Gh = [], []
        r = np.inf
        for it in range(maxit):
            fx, st = F(x)
            g = fx - x
            r = float(np.abs(g).max())
            if r < tol:
                break
            Xh.append(x.copy())
            Gh.append(g.copy())
            if len(Xh) > mmax + 1:
                Xh.pop(0)
                Gh.pop(0)
            if len(Xh) >= 2:
                dX = np.stack([Xh[j + 1] - Xh[j] for j in range(len(Xh) - 1)], 1)
                dG = np.stack([Gh[j + 1] - Gh[j] for j in range(len(Gh) - 1)], 1)
                gam = np.linalg.lstsq(dG, g, rcond=None)[0]
                x = x + beta * g - (dX + beta * dG) @ gam
            else:
                x = x + beta * g
        b, D = unpack(x)
        Q, on = self.fock(b, D, W, exchange)
        st = self.state(Q, on)
        b, D = self.fields(st)
        en = self.energy(b, D, W)
        return {"x": pack(b, D), "b": b, "D": D, "st": st, "it": it + 1, "resid": r, "gap": st["gap"], "W": W, **en}

    def prep(self, sol0, N, s):
        """E_HS and E_Wick functionals of the fixed flat-point state sol0 (its b and D) at the weights of (N, s)."""
        return self.energy(sol0["b"], sol0["D"], self.weights(N, s))


def quadratic_check(sea, sol, V_layer, lam=1e-4):
    """Slater quadratic control (planreviewB): for the free part (U = 0) under a one-body perturbation V (one 4 x 4 on-site matrix
    per layer), d^2 <H0 + lam V>_gs/d lam^2 = -2 sum_{i occ, a empty} |V_ai|^2/(eps_a - eps_i) per (k_2, k_3) block, against the
    central difference of the ground energy."""
    L = sea.L
    W = sea.weights(np.ones(L), np.ones(L))
    Q, on = sea.fock(np.zeros((3, L), complex), np.zeros((L, 4, 4), complex), W, False)

    def Egs(l):
        st = sea.state(Q, on + l * V_layer)
        return float(np.sum(st["w"] * st["occ"]))
    fd = (Egs(lam) - 2 * Egs(0.0) + Egs(-lam)) / lam ** 2
    st = sea.state(Q, on)
    an = 0.0
    Vb = np.zeros((4 * L, 4 * L), dtype=complex)
    for x in range(L):
        Vb[4 * x:4 * x + 4, 4 * x:4 * x + 4] = V_layer[x]
    for kk in range(sea.nk):
        u, w, o = st["u"][kk], st["w"][kk], st["occ"][kk]
        Vm = u.conj().T @ Vb @ u
        occ_i = np.nonzero(o > 0.5)[0]
        emp_a = np.nonzero(o < 0.5)[0]
        an += -2 * float(np.sum(np.abs(Vm[np.ix_(emp_a, occ_i)]) ** 2 / (w[emp_a][:, None] - w[occ_i][None, :])))
    return {"fd": fd, "analytic": an}
