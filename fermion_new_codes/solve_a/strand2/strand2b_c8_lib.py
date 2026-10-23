"""Strand 2b on C8 (spec SPEC_strand2b_row.md secs. 1-4 and the ruling of sec. 10; builder analyst).

Blocks (1,0) and (2,0) of the light packet on the open cube, in the quasiparticle basis of the current psi:
  (1,0): |r> = d_r^dag |Omega>, rho^P_x block = X_x (|r> basis), protected (no dressing);
  (2,0): |I> = d_i1^dag d_i2^dag |Omega> (i1 < i2), S_* block = C_IJ = v_IJ/(eps_I + eps_J) (a pure (2,2) operator's matrix on
         two-particle states is its coefficient matrix), rho^P_x block = e^{S} H_0^pp(X_x) e^{S} (the (2,0) reduction).
"At fixed state" (sec. 10): the fixed Fock vector Phi = sum_J c_J |J(psi)> is projected on the moved block,
c'_I' = sum_J det(V_I'(psi')^dag V_J(psi)) c_J, with V_I = [particle orbitals in tuple order, the nW sea orbitals in index order]
(the Slater determinant d_i1^dag d_i2^dag e_0^dag ... e_{nW-1}^dag |0>, the strand 0 convention for Omega).
"""
import os
import sys

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402


def herm(A):
    return 0.5 * (A + A.conj().T)


class BlockModel:
    def __init__(self, lat, m, gM, gB, Lam, blk):
        self.lat, self.m, self.gM, self.gB, self.Lam, self.blk = lat, m, gM, gB, Lam, tuple(blk)
        self.n = lat["nsite"]
        self.nW = lat["nsite"] * lat["ncomp"]
        nW = self.nW
        if self.blk == (1, 0):
            self.basis = [(r,) for r in range(nW)]
        elif self.blk == (2, 0):
            self.basis = [(a, b) for a in range(nW) for b in range(a + 1, nW)]
        else:
            raise ValueError("C8 2b blocks: (1,0) and (2,0)")
        self.pairs = np.array(self.basis)
        self.cache = {}

    def H0pp(self, X):
        """<I| sum_ab X_ab d_a^dag d_b |J> on the (2,0) basis."""
        P = self.pairs
        i1, i2 = P[:, 0][:, None], P[:, 1][:, None]
        j1, j2 = P[:, 0][None, :], P[:, 1][None, :]
        return (X[i1, j1] * (i2 == j2) - X[i1, j2] * (i2 == j1) - X[i2, j1] * (i1 == j2) + X[i2, j2] * (i1 == j1))

    def build(self, psi):
        key = tuple(np.round(np.asarray(psi, float), 15))
        if key in self.cache:
            return self.cache[key]
        lat, nW, n = self.lat, self.nW, self.n
        psi = np.asarray(psi, float)
        light = S2.Light(lat, self.m, psi, np.ones(n))
        ncp = lat["ncomp"]
        Xr = []
        for x in range(n):
            Px = np.zeros((nW, nW))
            Px[x * ncp:(x + 1) * ncp, x * ncp:(x + 1) * ncp] = np.eye(ncp)
            Xr.append(light.rbasis(herm(psi[x] * light.pk.E0h @ Px @ light.pk.E0h)))
        if self.blk == (1, 0):
            M = np.array(Xr)
            S = np.zeros((nW, nW))
            orb = np.array([np.column_stack([light.vp[:, r]] + [light.vm[:, s] for s in range(nW)]) for r in range(nW)])
        else:
            P = self.pairs
            if self.gM != 0 or self.gB != 0:
                terms = S2.seed_bilinears(lat, psi, self.gM, self.gB, self.Lam)
                T = S2.seed_tensor(terms, 2 * nW)
                WA = S2.two_two(S2.transform(T, light.Q), nW)
                v = 4.0 * WA[P[:, 0][:, None], P[:, 1][:, None], P[:, 0][None, :], P[:, 1][None, :]]
                eI = light.eps[P[:, 0]] + light.eps[P[:, 1]]
                S = v / (eI[:, None] + eI[None, :])
            else:
                v = np.zeros((len(P), len(P)), dtype=complex)
                S = np.zeros_like(v)
            eS = sla.expm(S)
            M = np.array([eS @ self.H0pp(X) @ eS for X in Xr])
            sea = light.vm
            orb = np.array([np.column_stack([light.vp[:, a], light.vp[:, b], sea]) for a, b in P])
        out = {"M": np.array([herm(Mx) for Mx in M]), "orb": orb, "S": S, "light": light}
        if len(self.cache) > 2000:
            self.cache.clear()
        self.cache[key] = out
        return out

    @staticmethod
    def overlap(orb_new, orb_old):
        """O_{I'J} = det(V_I'^dag V_J)."""
        G = np.einsum("iab,jac->ijbc", orb_new.conj(), orb_old, optimize=True)
        return np.linalg.det(G)


class SystemC8:
    """Reduced system in z = (psi, N) on a C8 block; the state = lowest multiplet of H = sum N_y M_y in the moving basis;
    partial derivatives at fixed Fock vector by Slater overlaps (sec. 10)."""

    def __init__(self, bm, K, k, h=1e-4, tol_deg=1e-9):
        self.bm, self.K, self.k, self.h, self.tol_deg = bm, np.asarray(K, float), float(k), h, tol_deg
        self.n = bm.n
        self.one = np.ones(self.n)

    def state(self, psi, N):
        b = self.bm.build(psi)
        H = np.einsum("y,yab->ab", N, b["M"])
        e, V = np.linalg.eigh(herm(H))
        d0 = int(np.sum(e - e[0] < self.tol_deg))
        return {"b": b, "H": H, "e": e, "V": V, "E": float(e[0]), "d0": d0, "C": V[:, :d0],
                "gap": float(e[d0] - e[0]) if len(e) > d0 else np.inf}

    def fixed_value(self, Cstate, b_old, psi_new, x_list=None):
        """Tr(rho' M_x(psi_new)) with rho' the projection of the fixed Fock vectors (columns of Cstate, at b_old's basis)."""
        bn = self.bm.build(psi_new)
        O = self.bm.overlap(bn["orb"], b_old["orb"])
        Cp = O @ Cstate
        xs = range(self.n) if x_list is None else x_list
        d0 = Cstate.shape[1]
        return np.array([np.real(np.trace(Cp.conj().T @ bn["M"][x] @ Cp)) / d0 for x in xs])

    def d1(self, psi, st, h=None):
        """t[y, x] = d/dpsi_y Tr(rho M_x) at fixed Fock vector."""
        h = self.h if h is None else h
        n = self.n
        t = np.zeros((n, n))
        for y in range(n):
            p1, p2 = np.array(psi, float), np.array(psi, float)
            p1[y] += h
            p2[y] -= h
            t[y] = (self.fixed_value(st["C"], st["b"], p1) - self.fixed_value(st["C"], st["b"], p2)) / (2 * h)
        return t

    def d1_fixed_coeff(self, psi, st, h=None):
        """Diagnostic (option (a)): fixed coefficients in the moving basis aligned by the polar part of the overlap."""
        h = self.h if h is None else h
        n = self.n
        t = np.zeros((n, n))
        for y in range(n):
            vals = []
            for s in (+1, -1):
                ps = np.array(psi, float)
                ps[y] += s * h
                bn = self.bm.build(ps)
                O = self.bm.overlap(bn["orb"], st["b"]["orb"])
                U, _, Vh = np.linalg.svd(O)
                Ca = (U @ Vh) @ st["C"]
                vals.append(np.array([np.real(np.trace(Ca.conj().T @ bn["M"][x] @ Ca)) for x in range(n)]) / st["d0"])
            t[y] = (vals[0] - vals[1]) / (2 * h)
        return t

    def F(self, z, lam=1.0, b=1.0):
        n, K, k = self.n, self.K, self.k
        psi, N = z[:n], z[n:]
        st = self.state(psi, N)
        src = np.array([np.real(np.trace(st["C"].conj().T @ st["b"]["M"][x] @ st["C"])) / st["d0"] for x in range(n)])
        t = self.d1(psi, st)
        Kpb = K @ (psi - b * self.one)
        C = lam * src - k * psi * Kpb
        G = k * (K @ self.one) + np.array([sum(N[x] * (lam * t[y, x] - k * ((x == y) * Kpb[x] + psi[x] * K[x, y]))
                                               for x in range(n)) for y in range(n)])
        return np.concatenate([C, G]), st, src, t


def proj_response_block(st, dH):
    """First-order change of rho = P_low/d0 under H -> H + dH within the (moving) block basis at psi."""
    e, V, d0 = st["e"], st["V"], st["d0"]
    low, oth = range(d0), range(d0, len(e))
    dP = np.zeros((len(e), len(e)), dtype=complex)
    for i in low:
        vi = V[:, i]
        for mm in oth:
            vm = V[:, mm]
            c = (vm.conj() @ dH @ vi) / (e[0] - e[mm])
            dP += c * np.outer(vm, vi.conj()) + np.conj(c) * np.outer(vi, vm.conj())
    return dP / d0


def jacobian_c8(sys_, z, lam=1.0, b=1.0):
    """Jacobian of the reduced map on C8: K terms analytic; the state's response by perturbation theory in the block
    (N-directions: dH = M_z; psi-directions: the change of H at fixed Fock vector, built from the overlap-projected block
    matrices); first and second psi-derivatives of Tr(rho M_x) at fixed Fock vector by central differences of overlaps."""
    n, K, k, h = sys_.n, sys_.K, sys_.k, sys_.h
    psi, N = z[:n], z[n:]
    st = sys_.state(psi, N)
    t = sys_.d1(psi, st)
    Kpb = K @ (psi - b * sys_.one)
    # the fixed-Fock-vector derivative of the block matrices themselves: dM_y[x] = d/dpsi_y (O^dag M_x(psi') O) at psi' = psi
    bo = st["b"]
    dMy = []
    for y in range(n):
        mats = []
        for s in (+1, -1):
            ps = np.array(psi, float)
            ps[y] += s * h
            bn = sys_.bm.build(ps)
            O = sys_.bm.overlap(bn["orb"], bo["orb"])
            mats.append(np.array([O.conj().T @ bn["M"][x] @ O for x in range(n)]))
        dMy.append((mats[0] - mats[1]) / (2 * h))
    dMy = np.array(dMy)  # (y, x, d, d) in the basis at psi
    # second derivatives t2[z, y, x] = d/dpsi_z d/dpsi_y Tr(rho M_x) at fixed Fock vector (mixed central differences)
    t2 = np.zeros((n, n, n))
    for zz in range(n):
        for y in range(zz, n):
            vals = []
            for sz, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                ps = np.array(psi, float)
                ps[zz] += sz * h
                ps[y] += sy * h
                vals.append(sys_.fixed_value(st["C"], bo, ps))
            t2[zz, y] = t2[y, zz] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h * h)
    Jm = np.zeros((2 * n, 2 * n))
    for col in range(2 * n):
        if col < n:
            dH = np.einsum("y,yab->ab", N, dMy[col])
        else:
            dH = bo["M"][col - n]
        dP = proj_response_block(st, dH)
        for x in range(n):
            v = lam * np.real(np.trace(dP @ bo["M"][x]))
            if col < n:
                v += lam * t[col, x] - k * ((x == col) * Kpb[x] + psi[x] * K[x, col])
            Jm[x, col] = v
        for y in range(n):
            v = 0.0
            for x in range(n):
                tr_dP = np.real(np.trace(dP @ dMy[y, x]))
                if col < n:
                    v += N[x] * (lam * (t2[col, y, x] + tr_dP) - k * ((x == y) * K[x, col] + (x == col) * K[x, y]))
                else:
                    v += N[x] * lam * tr_dP
            if col >= n:
                xx = col - n
                v += lam * t[y, xx] - k * ((xx == y) * Kpb[xx] + psi[xx] * K[xx, y])
            Jm[n + y, col] = v
    return Jm, st


def newton_c8(sys_, z0, lam=1.0, b=1.0, tol=1e-12, maxit=12, broyden=True):
    """Newton with an exact Jacobian at the first step and Broyden rank-one updates afterwards (exact Jacobian again
    when the residual stalls); positivity line search."""
    z = np.array(z0, float)
    res = []
    Fz = sys_.F(z, lam, b)[0]
    Jm = None
    nj = 0
    for it in range(maxit):
        r = float(np.linalg.norm(Fz))
        res.append(r)
        if r < tol:
            return z, {"ok": True, "residuals": res, "jacobians": nj}
        if Jm is None:
            Jm, _ = jacobian_c8(sys_, z, lam, b)
            nj += 1
        dz = np.linalg.solve(Jm, -Fz)
        step = 1.0
        while step > 1e-4:
            zn = z + step * dz
            if np.all(zn > 0):
                Fn = sys_.F(zn, lam, b)[0]
                if np.linalg.norm(Fn) < r or r < 1e-9:
                    break
            step /= 2
        s = zn - z
        if broyden and np.linalg.norm(Fn) < 0.5 * r:
            Jm = Jm + np.outer(Fn - Fz - Jm @ s, s) / (s @ s)
        else:
            Jm = None
        z, Fz = zn, Fn
    res.append(float(np.linalg.norm(Fz)))
    return z, {"ok": res[-1] < max(tol, 1e-11), "residuals": res, "jacobians": nj}


def continuation_c8(sys_, b=1.0, dlam0=0.25, min_dlam=1e-4):
    n = sys_.n
    z = np.concatenate([b * sys_.one, sys_.one])
    lam, dlam = 0.0, dlam0
    path = [{"lam": 0.0}]
    while lam < 1.0 - 1e-15:
        lt = min(1.0, lam + dlam)
        zn, info = newton_c8(sys_, z, lt, b)
        if info["ok"] and np.all(zn > 0):
            z, lam = zn, lt
            path.append({"lam": lam, "min_psi": float(z[:n].min()), "min_N": float(z[n:].min()),
                         "residual": info["residuals"][-1], "newton": len(info["residuals"]) - 1, "jacobians": info["jacobians"]})
            if len(info["residuals"]) <= 5:
                dlam = min(2 * dlam, 0.5)
        else:
            dlam /= 2
            if dlam < min_dlam:
                return z, {"ok": False, "lost_at_lam": lam, "path": path}
    return z, {"ok": True, "path": path}
