"""Strand 2d library (spec solve/specs/SPEC_strand2d_jet.md; builder analyst).

Lattices: C8 (strand2a_lib) and B4, the periodic 4^3 box with the 2-component Weyl operator in the native cube's convention
(D[x, x + e_j] += sigma_j/(2i), D[x, x - e_j] -= sigma_j/(2i); site x = x_0 + 4 x_1 + 16 x_2; Weyl index w = 2 x + a), the
ruler distance of V_B the minimum-image Euclidean distance on the torus (builder reading), dirs 0, 1, 2.
The vacuum W_{0,a} is strand2a_lib.W0 (exact one-body traces).  The matter is the lowest multiplet of
H_exc[N, psi]|_(2,0) = e^S H_0^pp(sum_x N_x X_x) e^S on the (2,0) block of the light packet (2a's reduction, as in
strand2b_c8_lib), built in torch (complex128) on a chosen device.  The seed's (2,2) projection on the pp block uses its
Kronecker structure: with m_t = vp^dag M_t vp (particle orbitals vp), T'_{ijkl} = sum_t w_t m1_t[i,k] m2_t[j,l], so
K[(i,k),(j,l)] = sum_t w_t m1_t[i,k] m2_t[j,l] is one matrix product over the term index, and
v_IJ = T'[i1,i2,j1,j2] - T'[i2,i1,j1,j2] - T'[i1,i2,j2,j1] + T'[i2,i1,j2,j1] (= 4 W_A of strand2a_lib.two_two on pp);
the V_B double sum over sites is grouped per direction as sum_x J_x (x) (sum_y w_xy J_y), which strand2a_lib's term list
(one term per site pair) would store 4096 times per direction on B4.  The overlaps of the moved block with a fixed Fock
vector use the common-sea Schur complement (strand2b_c8_lib_v3).
"""
import os
import sys

import numpy as np
import scipy.linalg as sla
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import native_core as nc  # noqa: E402

SX, SY, SZ, I2 = nc.SX, nc.SY, nc.SZ, nc.I2
SIG = (SX, SY, SZ)
CT = torch.complex128


def herm(A):
    return 0.5 * (A + A.conj().T)


def lattice_B4(L=4):
    n = L ** 3
    pos = np.array([[x % L, (x // L) % L, x // (L * L)] for x in range(n)], dtype=float)
    D = np.zeros((2 * n, 2 * n), dtype=complex)
    for x in range(n):
        for j in range(3):
            for sgn in (1, -1):
                c = pos[x].astype(int).copy()
                c[j] = (c[j] + sgn) % L
                y = int(c[0] + L * c[1] + L * L * c[2])
                D[2 * x:2 * x + 2, 2 * y:2 * y + 2] += sgn * SIG[j] / 2j
    dv = np.abs(pos[:, None, :] - pos[None, :, :])
    dv = np.minimum(dv, L - dv)
    dist = np.sqrt((dv ** 2).sum(-1))
    return {"name": "B4", "nsite": n, "ncomp": 2, "DW": D, "positions": pos, "dist": dist, "dirs": [0, 1, 2],
            "coord": pos.astype(int), "L": L}


def lattice(name):
    return lattice_B4() if name == "B4" else S2.lattice(name)


def site_terms(lat, psi, gM, gB, Lam):
    """The seed grouped by Kronecker factor: list of (weight, M1, M2) with V's pp part = sum w (c^dag M1 c)(c^dag M2 c)
    restricted to pp; V_B grouped per direction as sum_x J_x (x) N_x, N_x = sum_y w_xy J_y (same terms as
    strand2a_lib.seed_bilinears, summed over y first)."""
    R = S2.regulator(lat, psi, Lam)
    psi = np.asarray(psi, float)
    ncp = lat["ncomp"]
    out = []
    for x in range(lat["nsite"]):
        Mx = R @ S2.site_op(lat, x, np.eye(ncp), SX) @ R
        out.append((-gM * psi[x] ** -6, Mx, Mx))
    if gB != 0:
        if not np.isfinite(Lam):
            raise ValueError("V_B needs finite Lambda")
        ell = 1.0 / Lam
        for i in lat["dirs"]:
            sig = SIG[i] if ncp == 2 else np.eye(1)
            J = np.array([R @ S2.site_op(lat, x, sig, SY) @ R for x in range(lat["nsite"])])
            d = lat["dist"] * (psi[:, None] ** 2 + psi[None, :] ** 2) / 2.0
            w = -(gB / 2.0) * ell ** -3 * np.exp(-d * d / (ell * ell))
            Nx = np.einsum("xy,yab->xab", w, J)
            for x in range(lat["nsite"]):
                out.append((1.0, J[x], Nx[x]))
    return out


class PairBlock:
    """(2,0) block of the light packet: pairs I = (i1 < i2) of particle orbitals; torch tensors on `device`."""

    def __init__(self, lat, m, gM, gB, Lam, device="cpu"):
        self.lat, self.m, self.gM, self.gB, self.Lam = lat, m, gM, gB, Lam
        self.n = lat["nsite"]
        self.nW = lat["nsite"] * lat["ncomp"]
        self.dev = torch.device(device)
        nW = self.nW
        P = np.array([(a, b) for a in range(nW) for b in range(a + 1, nW)])
        self.pairs = P
        self.i1 = torch.tensor(P[:, 0], device=self.dev)
        self.i2 = torch.tensor(P[:, 1], device=self.dev)
        self.dim = len(P)
        self.cache = {}

    def t(self, a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=CT, device=self.dev)

    def v_pairs(self, light, psi):
        """v_IJ on the pp block from the Kronecker-grouped seed (see the module docstring)."""
        nW = self.nW
        vp = self.t(light.vp)
        terms = site_terms(self.lat, psi, self.gM, self.gB, self.Lam)
        A, Bm = [], []
        for w, M1, M2 in terms:
            m1 = vp.conj().T @ self.t(M1) @ vp
            m2 = vp.conj().T @ self.t(M2) @ vp
            A.append((w * m1).reshape(-1))
            Bm.append(m2.reshape(-1))
        A = torch.stack(A)   # (T, nW*nW), index (i, k)
        Bm = torch.stack(Bm)  # (T, nW*nW), index (j, l)
        K = A.T @ Bm         # K[(i,k), (j,l)]
        T4 = K.reshape(nW, nW, nW, nW).permute(0, 2, 1, 3)  # T'[i, j, k, l]
        a1, a2 = self.i1[:, None], self.i2[:, None]
        b1, b2 = self.i1[None, :], self.i2[None, :]
        v = T4[a1, a2, b1, b2] - T4[a2, a1, b1, b2] - T4[a1, a2, b2, b1] + T4[a2, a1, b2, b1]
        del K, T4, A, Bm
        return v

    def build(self, psi):
        key = tuple(np.round(np.asarray(psi, float), 15))
        if key in self.cache:
            return self.cache[key]
        lat, nW, n = self.lat, self.nW, self.n
        psi = np.asarray(psi, float)
        light = S2.Light(lat, self.m, psi, np.ones(n))
        ncp = lat["ncomp"]
        X = []
        for x in range(n):
            Px = np.zeros((nW, nW))
            Px[x * ncp:(x + 1) * ncp, x * ncp:(x + 1) * ncp] = np.eye(ncp)
            X.append(light.rbasis(herm(psi[x] * light.pk.E0h @ Px @ light.pk.E0h)))
        Xt = self.t(np.array(X))
        eps = torch.as_tensor(light.eps, dtype=torch.float64, device=self.dev)
        if self.gM != 0 or self.gB != 0:
            v = self.v_pairs(light, psi)
            eI = eps[self.i1] + eps[self.i2]
            S = v / (eI[:, None] + eI[None, :]).to(CT)
            eS = torch.linalg.matrix_exp(S)
            vnorm = float(torch.linalg.norm(v))
            del v, S
        else:
            eS, vnorm = None, 0.0
        out = {"light": light, "X": Xt, "eS": eS, "eps": light.eps, "v_norm": vnorm}
        if len(self.cache) > 6:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out

    def H0pp(self, X):
        """<I| sum_ab X_ab d_a^dag d_b |J> on the pair basis (dense)."""
        a1, a2 = self.i1[:, None], self.i2[:, None]
        b1, b2 = self.i1[None, :], self.i2[None, :]
        return (X[a1, b1] * (a2 == b2) - X[a1, b2] * (a2 == b1) - X[a2, b1] * (a1 == b2) + X[a2, b2] * (a1 == b1))

    def H(self, b, N):
        Nt = torch.as_tensor(np.asarray(N, float), dtype=CT, device=self.dev)
        Xn = torch.einsum("x,xab->ab", Nt, b["X"])
        H0 = self.H0pp(Xn)
        Hm = H0 if b["eS"] is None else b["eS"] @ H0 @ b["eS"]
        return 0.5 * (Hm + Hm.conj().T)

    def overlap(self, b_new, b_old):
        """O_{I'J} = <I'(psi_new)|J(psi_old)> by the common-sea Schur complement (strand2b_c8_lib_v3)."""
        ln, lo = b_new["light"], b_old["light"]
        A = ln.vp.conj().T @ lo.vp
        B = ln.vp.conj().T @ lo.vm
        C = ln.vm.conj().T @ lo.vp
        D = ln.vm.conj().T @ lo.vm
        M = self.t(A - B @ np.linalg.solve(D, C))
        dD = complex(np.linalg.det(D))
        a1, a2 = self.i1[:, None], self.i2[:, None]
        b1, b2 = self.i1[None, :], self.i2[None, :]
        return dD * (M[a1, b1] * M[a2, b2] - M[a1, b2] * M[a2, b1])


def eigh(Hm):
    w, V = torch.linalg.eigh(Hm)
    return w.cpu().numpy(), V
