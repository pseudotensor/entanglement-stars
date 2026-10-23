"""Strand 2b library (spec solve/specs/SPEC_strand2b_row.md; builder analyst): the law's clock derivative rho^P_x[psi] in
the fixed c-basis Fock sector of R4 (8 modes), built lean from strand 2a's algebraic (2,2) projection, and the reduced
static system in z = (psi, N) with the state eliminated through its block spectral projector.

rho^P_x[psi] = sum_rr' (X_x)_rr' F_dr^dag F_dr' + sum_rr' (X_x)_r'r F_fr^dag F_fr', F = e^{-S_*} a e^{S_*} on the sector
N_c, X_x = psi_x E0^(1/2) (P_x (x) 1) E0^(1/2) in the |r> basis of B0 at N = 1 (strand 0 sec. 1.4-1.5; 2a's seed with the
regulator and V_B).  H[N, psi] = sum_y N_y rho^P_y[psi] (= H_exc[N] by the Euler identity).
Version 2: the state's projector rank d0 is fixed per solve at the degeneracy of the block's lowest multiplet at the flat
start (lambda = 0), so rho = (projector on the lowest d0 levels)/d0 is a smooth function of z; version 1 recomputed d0 at
tolerance 1e-9 at every z, which makes the map discontinuous where symmetry-breaking steps split a degenerate multiplet.
Version 3: d rho^P/d psi by the fourth-order central stencil (-f(2h) + 8 f(h) - 8 f(-h) + f(-2h))/(12 h) at h = 1e-3 (truncation
about h^4, rounding about eps/h: the residual's noise floor drops from about 2e-11 to about 1e-13); the spec's second-order
check at h = 1e-4 against 1e-5 is kept as drho2 and reported by the driver.
Version 4 (speed, same objects): BLAS tensordot/matmul in place of the general einsums; the adjoint row's second derivatives
d^2 rho^P/d psi_z d psi_y by the direct second-order mixed stencil at h = 1e-3 (32 builds per Jacobian instead of 128).
Version 5: Newton returns its best iterate and stops at the residual's noise floor (no decrease by a factor 2 over two
iterations once below 1e-10); accepted when the best residual is below 1e-11 (the spec's bound).  The floor (about 5e-12
with the fourth-order stencil) made version 4 run to the iteration cap and occasionally end on a fluctuation above 1e-11.
"""
import os
import sys

import numpy as np
import scipy.linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import native_sources as ns  # noqa: E402


def herm(A):
    return 0.5 * (A + A.conj().T)


class FockOps:
    """Annihilator stacks ann[N] (nmodes, d_{N-1}, d_N) of the 2 nW c-modes (native conventions)."""

    def __init__(self, nmodes):
        self.nm = nmodes
        self.fock = ns.Fock(nmodes)
        self.ann = {}
        for N in range(1, nmodes + 1):
            self.ann[N] = np.array([self.fock.ann(N, j) for j in range(nmodes)])

    def dim(self, N):
        return self.fock.sector(N).d


class RhoP:
    """rho^P_x[psi] on sector Nc for one (lattice, m, couplings, Lambda); evaluations cached by psi."""

    def __init__(self, lat, m, gM, gB, Lam, Nc, fops):
        self.lat, self.m, self.gM, self.gB, self.Lam, self.Nc, self.f = lat, m, gM, gB, Lam, Nc, fops
        self.nW = lat["nsite"] * lat["ncomp"]
        self.n = lat["nsite"]
        self.cache = {}

    def qp_ops(self, light, N):
        """a_i on sector N: particles i < nW map N -> N-1, holes map N -> N+1; returns (stack_p, stack_h) or None parts."""
        nW, f = self.nW, self.f
        P = None
        nm = f.nm
        if N >= 1:
            a = f.ann[N]
            P = (light.vp.conj().T @ a.reshape(nm, -1)).reshape(nW, a.shape[1], a.shape[2])
        Hh = None
        if N + 1 <= f.nm:
            # f_r = sum_j vm_jr c_j^dag : N -> N+1 ; c_j^dag on N = ann[N+1][j]^T
            a = f.ann[N + 1]
            Hh = (light.vm.T @ a.reshape(nm, -1)).reshape(nW, a.shape[1], a.shape[2]).transpose(0, 2, 1)
        return P, Hh

    def pair_ops(self, light, N):
        """For each class, the stack over pairs J = (j1 < j2) of q_J = a_j2 a_j1 acting on sector N, and the pair list."""
        nW = self.nW
        P1, H1 = self.qp_ops(light, N)
        out = {}
        # pp: a_j2 a_j1 with both particles: N -> N-1 -> N-2
        if P1 is not None and N >= 2:
            P2, _ = self.qp_ops(light, N - 1)
            pairs = [(a, b) for a in range(nW) for b in range(a + 1, nW)]
            ia, ib = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            out["pp"] = (pairs, np.matmul(P2[ib], P1[ia]))
        # hh: both holes: N -> N+1 -> N+2
        if H1 is not None and N + 2 <= self.f.nm:
            _, H2 = self.qp_ops(light, N + 1)
            pairs = [(nW + a, nW + b) for a in range(nW) for b in range(a + 1, nW)]
            ia, ib = np.array([p[0] - nW for p in pairs]), np.array([p[1] - nW for p in pairs])
            out["hh"] = (pairs, np.matmul(H2[ib], H1[ia]))
        # ph: j1 particle (< nW), j2 hole: a_j2 a_j1: N -> N-1 -> N
        if P1 is not None:
            _, H2 = self.qp_ops(light, N - 1)
            if H2 is not None:
                pairs = [(a, nW + b) for a in range(nW) for b in range(nW)]
                ia, ib = np.array([p[0] for p in pairs]), np.array([p[1] - nW for p in pairs])
                out["ph"] = (pairs, np.matmul(H2[ib], P1[ia]))
        return out

    def build(self, psi):
        key = tuple(np.round(np.asarray(psi, float), 15))
        if key in self.cache:
            return self.cache[key]
        lat, nW = self.lat, self.nW
        psi = np.asarray(psi, float)
        light = S2.Light(lat, self.m, psi, np.ones(self.n))
        eq = light.eps_q
        Cls = None
        if self.gM != 0 or self.gB != 0:
            terms = S2.seed_bilinears(lat, psi, self.gM, self.gB, self.Lam)
            T = S2.seed_tensor(terms, 2 * nW)
            WA = S2.two_two(S2.transform(T, light.Q), nW)
            Cls = WA
        Nc = self.Nc
        S = {}
        for N in (Nc - 1, Nc, Nc + 1):
            d = self.f.dim(N) if 0 <= N <= self.f.nm else 0
            Sm = np.zeros((d, d), dtype=complex)
            if Cls is not None and d > 0:
                for cl, (pairs, A) in self.pair_ops(light, N).items():
                    idx = np.array(pairs)
                    V = 4.0 * Cls[idx[:, 0][:, None], idx[:, 1][:, None], idx[:, 0][None, :], idx[:, 1][None, :]]
                    den = (eq[idx[:, 0]] + eq[idx[:, 1]])[:, None] + (eq[idx[:, 0]] + eq[idx[:, 1]])[None, :]
                    Cc = V / den
                    # S += sum_IJ C_IJ A_I^dag A_J
                    Y = np.tensordot(Cc, A, axes=([1], [0]))
                    Sm += np.tensordot(A.conj(), Y, axes=([0, 1], [0, 1]))
            S[N] = Sm
        eS = {N: sla.expm(S[N]) for N in S}
        emS = {N: sla.expm(-S[N]) for N in S}
        P, Hh = self.qp_ops(light, Nc)
        FD = np.array([emS[Nc - 1] @ P[r] @ eS[Nc] for r in range(nW)]) if P is not None else None
        FF = np.array([emS[Nc + 1] @ Hh[r] @ eS[Nc] for r in range(nW)]) if Hh is not None else None
        rho = []
        ncp = lat["ncomp"]
        for x in range(self.n):
            Px = np.zeros((nW, nW))
            Px[x * ncp:(x + 1) * ncp, x * ncp:(x + 1) * ncp] = np.eye(ncp)
            Xr = light.rbasis(herm(psi[x] * light.pk.E0h @ Px @ light.pk.E0h))
            Rx = np.zeros((self.f.dim(Nc), self.f.dim(Nc)), dtype=complex)
            if FD is not None:
                Rx += np.tensordot(FD.conj(), np.tensordot(Xr, FD, axes=([1], [0])), axes=([0, 1], [0, 1]))
            if FF is not None:
                Rx += np.tensordot(FF.conj(), np.tensordot(Xr.T, FF, axes=([1], [0])), axes=([0, 1], [0, 1]))
            rho.append(herm(Rx))
        # n_exc on Nc with the bare quasiparticle operators at psi
        nex = np.zeros((self.f.dim(Nc), self.f.dim(Nc)), dtype=complex)
        if P is not None:
            nex += np.tensordot(P.conj(), P, axes=([0, 1], [0, 1]))
        if Hh is not None:
            nex += np.tensordot(Hh.conj(), Hh, axes=([0, 1], [0, 1]))
        out = {"rho": np.array(rho), "nexc": herm(nex), "light": light, "S": S[Nc]}
        if len(self.cache) > 4000:
            self.cache.clear()
        self.cache[key] = out
        return out


def block_states(H, nexc, k, tol=1e-9):
    """Eigenpairs of H on the sector, labelled by n_exc; returns all eigvals/vecs of H (sorted) restricted per block via n_exc."""
    w, V = np.linalg.eigh(nexc)
    sel = np.where(np.abs(w - k) < 1e-6)[0]
    W = V[:, sel]
    Hb = herm(W.conj().T @ H @ W)
    e, U = np.linalg.eigh(Hb)
    return e, W @ U


class System:
    """The reduced static system of SPEC_strand2b secs. 1-2 in z = (psi, N)."""

    def __init__(self, rp, K, k, kexc, hs=(1e-3,), tol_deg=1e-9):
        self.rp, self.K, self.k, self.kexc = rp, np.asarray(K, float), float(k), kexc
        self.d0_fixed = None
        self.n = rp.n
        self.h = hs[0]
        self.tol_deg = tol_deg
        self.one = np.ones(self.n)

    def state(self, psi, N):
        b = self.rp.build(psi)
        H = np.einsum("y,yab->ab", N, b["rho"])
        e, Vb = block_states(H, b["nexc"], self.kexc)
        d0_tol = int(np.sum(e - e[0] < self.tol_deg))
        d0 = self.d0_fixed if self.d0_fixed is not None else d0_tol
        Pl = Vb[:, :d0]
        rho = Pl @ Pl.conj().T / d0
        gap = float(e[d0] - e[0]) if len(e) > d0 else np.inf
        return {"b": b, "H": H, "E": float(e[0]), "d0": d0, "d0_at_tolerance": d0_tol, "rho": rho, "Pl": Pl, "gap_block": gap,
                "multiplet_splitting": float(e[d0 - 1] - e[0]), "E_multiplet_mean": float(np.mean(e[:d0]))}

    def drho(self, psi, h=None):
        """d rho^P_x / d psi_y by the fourth-order central stencil: array (y, x, D, D)."""
        h = self.h if h is None else h
        out = []
        for y in range(self.n):
            f = {}
            for j in (-2, -1, 1, 2):
                p = np.array(psi, float)
                p[y] += j * h
                f[j] = self.rp.build(p)["rho"]
            out.append((-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * h))
        return np.array(out)

    def drho2(self, psi, h):
        """The spec's second-order central difference at step h (for the agreement check)."""
        out = []
        for y in range(self.n):
            p1, p2 = np.array(psi, float), np.array(psi, float)
            p1[y] += h
            p2[y] -= h
            out.append((self.rp.build(p1)["rho"] - self.rp.build(p2)["rho"]) / (2 * h))
        return np.array(out)

    def F(self, z, lam=1.0, b=1.0, st=None, dr=None):
        n = self.n
        psi, N = z[:n], z[n:]
        st = self.state(psi, N) if st is None else st
        dr = self.drho(psi) if dr is None else dr
        K, k = self.K, self.k
        Kpb = K @ (psi - b * self.one)
        src = np.array([np.real(np.trace(st["rho"] @ st["b"]["rho"][x])) for x in range(n)])
        C = lam * src - k * psi * Kpb
        t = np.array([[np.real(np.trace(st["rho"] @ dr[y, x])) for x in range(n)] for y in range(n)])  # t[y, x]
        G = k * (K @ self.one) + np.array([sum(N[x] * (lam * t[y, x] - k * ((x == y) * Kpb[x] + psi[x] * K[x, y]))
                                               for x in range(n)) for y in range(n)])
        return np.concatenate([C, G]), st, src, t

    def proj_response(self, st, dH):
        """First-order change of rho = P_low/d0 under H -> H + dH, over the whole sector (all other levels)."""
        w, V = np.linalg.eigh(herm(st["H"]))
        # identify the low multiplet among the sector eigenvectors by overlap with Pl
        ov = np.sum(np.abs(st["Pl"].conj().T @ V) ** 2, axis=0)
        low = np.where(ov > 0.5)[0]
        oth = np.where(ov <= 0.5)[0]
        E0 = st["E"]
        dP = np.zeros_like(st["rho"])
        for i in low:
            vi = V[:, i]
            for mm in oth:
                vm = V[:, mm]
                c = (vm.conj() @ dH @ vi) / (E0 - w[mm])
                dP += c * np.outer(vm, vi.conj()) + np.conj(c) * np.outer(vi, vm.conj())
        return dP / st["d0"], float(np.min(np.abs(E0 - w[oth]))) if len(oth) else np.inf

    def J(self, z, lam=1.0, b=1.0, h2=None):
        """Jacobian of F: K terms analytic; projector response by perturbation theory; rho^P derivatives by central
        differences (first derivatives at h, the adjoint row's second derivatives by differences of first derivatives)."""
        n = self.n
        psi, N = z[:n], z[n:]
        st = self.state(psi, N)
        dr = self.drho(psi)
        K, k = self.K, self.k
        Kpb = K @ (psi - b * self.one)
        h = self.h if h2 is None else h2
        # second derivatives d_z d_y rho^P_x by the direct mixed second-order stencil
        base = st["b"]["rho"]
        d2 = np.zeros((n, n) + base.shape, dtype=complex)
        for zz in range(n):
            for y in range(zz, n):
                if y == zz:
                    p1, p2 = np.array(psi, float), np.array(psi, float)
                    p1[y] += h
                    p2[y] -= h
                    v = (self.rp.build(p1)["rho"] - 2 * base + self.rp.build(p2)["rho"]) / (h * h)
                else:
                    vals = []
                    for sz, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        p = np.array(psi, float)
                        p[zz] += sz * h
                        p[y] += sy * h
                        vals.append(self.rp.build(p)["rho"])
                    v = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h * h)
                d2[zz, y] = v
                d2[y, zz] = v
        Jm = np.zeros((2 * n, 2 * n))
        mins = []
        for col in range(2 * n):
            if col < n:
                dH = np.einsum("y,yab->ab", N, dr[col])
            else:
                dH = st["b"]["rho"][col - n]
            dP, mg = self.proj_response(st, dH)
            mins.append(mg)
            for x in range(n):
                v = lam * np.real(np.trace(dP @ st["b"]["rho"][x]))
                if col < n:
                    v += lam * np.real(np.trace(st["rho"] @ dr[col, x]))
                    v -= k * ((x == col) * Kpb[x] + psi[x] * K[x, col])
                Jm[x, col] = v
            for y in range(n):
                v = 0.0
                for x in range(n):
                    tr_dP = np.real(np.trace(dP @ dr[y, x]))
                    if col < n:
                        v += N[x] * (lam * (np.real(np.trace(st["rho"] @ d2[col, y, x])) + tr_dP)
                                     - k * ((x == y) * K[x, col] + (x == col) * K[x, y]))
                    else:
                        v += N[x] * lam * tr_dP
                if col >= n:
                    xx = col - n
                    v += lam * np.real(np.trace(st["rho"] @ dr[y, xx])) - k * ((xx == y) * Kpb[xx] + psi[xx] * K[xx, y])
                Jm[n + y, col] = v
        return Jm, st, float(min(mins))

    def newton(self, z0, lam=1.0, b=1.0, tol=1e-12, maxit=12, accept=1e-11):
        z = np.array(z0, float)
        res = []
        best_z, best_r = z.copy(), np.inf
        for it in range(maxit):
            Fz, st, _, _ = self.F(z, lam, b)
            r = float(np.linalg.norm(Fz))
            res.append(r)
            if r < best_r:
                best_z, best_r = z.copy(), r
            if r < tol:
                return z, {"ok": True, "residuals": res, "it": it}
            if len(res) >= 3 and best_r < 1e-10 and res[-1] > 0.5 * res[-3]:
                return best_z, {"ok": best_r < accept, "residuals": res, "it": it, "stopped": "noise floor", "best": best_r}
            Jm, _, _ = self.J(z, lam, b)
            try:
                dz = np.linalg.solve(Jm, -Fz)
            except np.linalg.LinAlgError:
                return z, {"ok": False, "residuals": res, "why": "singular"}
            step = 1.0
            while step > 1e-4:
                zn = z + step * dz
                if np.all(zn > 0):
                    rn = float(np.linalg.norm(self.F(zn, lam, b)[0]))
                    if rn < r or r < 1e-9:
                        break
                step /= 2
            z = zn
        Fz = self.F(z, lam, b)[0]
        r = float(np.linalg.norm(Fz))
        res.append(r)
        if r < best_r:
            best_z, best_r = z.copy(), r
        return best_z, {"ok": best_r < accept, "residuals": res, "it": maxit, "best": best_r}

    def continuation(self, b=1.0, dlam0=0.25, min_dlam=1e-4):
        n = self.n
        z = np.concatenate([b * self.one, self.one])
        if self.d0_fixed is None:
            self.d0_fixed = self.state(z[:n], z[n:])["d0_at_tolerance"]
        lam, dlam = 0.0, dlam0
        path = [{"lam": 0.0, "min_psi": float(z[:n].min()), "min_N": float(z[n:].min())}]
        while lam < 1.0 - 1e-15:
            lt = min(1.0, lam + dlam)
            zn, info = self.newton(z, lt, b)
            if info["ok"] and np.all(zn > 0):
                z, lam = zn, lt
                path.append({"lam": lam, "min_psi": float(z[:n].min()), "min_N": float(z[n:].min()),
                             "residual": info["residuals"][-1], "newton": len(info["residuals"]) - 1})
                if len(info["residuals"]) <= 5:
                    dlam = min(2 * dlam, 0.5)
            else:
                dlam /= 2
                if dlam < min_dlam:
                    return z, {"ok": False, "lost_at_lam": lam, "path": path}
        return z, {"ok": True, "path": path}
