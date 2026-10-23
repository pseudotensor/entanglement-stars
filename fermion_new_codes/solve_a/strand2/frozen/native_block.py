"""Block-operator bridge for the native action (integ69_nativedefs2 promotions), generalizing native_core.Bridge.

Rulers S_y live on blocks y = (site x, sector s); sector s has dimension d_s.  A single-sector problem has one
sector; the field-interposed hopping c_i^dag A_i D_ij A_j c_j places rulers of sector N - 1 inside the source
of sector N, so that variant carries the chain of sectors 0..N (the full Fock space below the target, which is
all the target equations see).  Constraint, per block y = (x, s):

    C_y(S) = lam e_y(S) - k L_{S_y}(w_y),   w_y = sum_x' K_xx' (S_(x', s) - b I_s),   L_A(X) = (AX + XA)/2,

with lam the continuation parameter (lam = 1 is the bridge), Q_s = k sum_xx' K_xx' (S_(x', s) - b I_s), and the
target Hamiltonian H = Q_top.  Sources are BExpr sums of products of ruler powers and fixed (possibly
rectangular) matrices, with Daleckii-Krein derivatives and adjoints under the real trace pairing.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, eigsh

import native_core as nc


class BlockMatFun:
    """Spectral data of all ruler blocks; powers and divided differences of t -> t^p."""

    def __init__(self, S):
        self.S = S
        self.eig = [np.linalg.eigh(s) for s in S]
        self.min_eig = min(float(w.min()) for w, _ in self.eig)
        if self.min_eig <= 0:
            raise ValueError("ruler not positive")
        self._pow, self._dd = {}, {}

    def power(self, y, p):
        key = (y, p)
        if key not in self._pow:
            w, V = self.eig[y]
            self._pow[key] = (V * w ** p) @ V.conj().T
        return self._pow[key]

    def dd(self, y, p):
        key = (y, p)
        if key not in self._dd:
            w, _ = self.eig[y]
            Wi, Wj = np.meshgrid(w, w, indexing='ij')
            diff = Wi - Wj
            close = np.abs(diff) < 1e-9 * np.maximum(1.0, np.abs(Wi))
            der = p * (0.5 * (Wi + Wj)) ** (p - 1)
            self._dd[key] = np.where(close, der, (Wi ** p - Wj ** p) / np.where(close, 1.0, diff))
        return self._dd[key]

    def dpower(self, y, p, X):
        w, V = self.eig[y]
        return V @ (self.dd(y, p) * (V.conj().T @ X @ V)) @ V.conj().T


class BExpr:
    """sum of coef * F_1 F_2 ... ; F = ('S', y, p) (ruler power of block y) or ('M', matrix, possibly rectangular)."""

    def __init__(self, dim):
        self.dim = dim
        self.terms = []

    def add(self, coef, *factors):
        if coef != 0:
            self.terms.append((coef, list(factors)))
        return self

    def ruler_blocks(self):
        return sorted({f[1] for _, fs in self.terms for f in fs if f[0] == 'S'})

    @staticmethod
    def _mat(mf, f):
        return mf.power(f[1], f[2]) if f[0] == 'S' else f[1]

    @staticmethod
    def _chain(mats):
        P = mats[0]
        for A in mats[1:]:
            P = P @ A
        return P

    def value(self, mf):
        out = np.zeros((self.dim, self.dim), dtype=complex)
        for coef, fs in self.terms:
            out += coef * self._chain([self._mat(mf, f) for f in fs])
        return out

    def prepare(self, mf):
        """Cache, per ruler-factor occurrence, the left and right products (fixed while S is fixed)."""
        occ = []
        for coef, fs in self.terms:
            mats = [self._mat(mf, f) for f in fs]
            for i, f in enumerate(fs):
                if f[0] != 'S':
                    continue
                Lm = self._chain(mats[:i]) if i > 0 else None
                Rm = self._chain(mats[i + 1:]) if i + 1 < len(fs) else None
                occ.append((coef, f[1], f[2], Lm, Rm))
        return occ

    def deriv(self, mf, X, occ=None, cache=None):
        occ = self.prepare(mf) if occ is None else occ
        cache = {} if cache is None else cache
        out = np.zeros((self.dim, self.dim), dtype=complex)
        for coef, y, p, Lm, Rm in occ:
            if (y, p) not in cache:
                cache[(y, p)] = mf.dpower(y, p, X[y])
            M = cache[(y, p)]
            if Lm is not None:
                M = Lm @ M
            if Rm is not None:
                M = M @ Rm
            out += coef * M
        return out

    def adjoint(self, mf, Lam, out, occ=None):
        """Accumulate De^*[Lam] into out[y] (list of block matrices) under Re Tr pairing (herm taken by caller)."""
        occ = self.prepare(mf) if occ is None else occ
        for coef, y, p, Lm, Rm in occ:
            Z = Lam
            if Lm is not None:
                Z = Z @ Lm
            if Rm is not None:
                Z = Rm @ Z
            out[y] += coef * mf.dpower(y, p, Z)


class BlockBridge:
    """Blocks y = (site, sector); see module docstring."""

    def __init__(self, exprs, blocks, dims, K, k, top):
        self.e, self.blocks, self.dims = exprs, blocks, dims
        self.K, self.k = np.asarray(K, float), float(k)
        self.nsite = self.K.shape[0]
        self.top = top
        self.index = {b: i for i, b in enumerate(blocks)}
        self.sectors = sorted({s for _, s in blocks})
        self.nb = len(blocks)
        self.sizes = [dims[s] ** 2 for _, s in blocks]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.N = int(self.offsets[-1])
        self.bvec = self.K @ np.ones(self.nsite)

    # -- packing
    def pack(self, Xs):
        return np.concatenate([nc.herm_to_vec(nc.herm(X)) for X in Xs])

    def unpack(self, v):
        return [nc.vec_to_herm(v[self.offsets[i]:self.offsets[i + 1]], self.dims[s]) for i, (_, s) in enumerate(self.blocks)]

    def eye(self, y):
        return np.eye(self.dims[self.blocks[y][1]], dtype=complex)

    def siteblocks(self, s):
        return [self.index[(x, s)] for x in range(self.nsite)]

    # -- constraint pieces
    def w(self, S, b):
        out = [None] * self.nb
        for s in self.sectors:
            ys = self.siteblocks(s)
            I = self.eye(ys[0])
            for x, y in enumerate(ys):
                out[y] = sum(self.K[x, xp] * (S[yp] - b * I) for xp, yp in enumerate(ys))
        return out

    def Kmix(self, X):
        out = [None] * self.nb
        for s in self.sectors:
            ys = self.siteblocks(s)
            for x, y in enumerate(ys):
                out[y] = sum(self.K[x, xp] * X[yp] for xp, yp in enumerate(ys))
        return out

    def C(self, S, b=1.0, lam=1.0, mf=None):
        mf = mf or BlockMatFun(S)
        w = self.w(S, b)
        return [lam * self.e[y].value(mf) - self.k * nc.jordan(S[y], w[y]) for y in range(self.nb)]

    def source_values(self, S):
        mf = BlockMatFun(S)
        return [self.e[y].value(mf) for y in range(self.nb)]

    def linearization(self, S, b=1.0, lam=1.0):
        mf = BlockMatFun(S)
        w = self.w(S, b)
        occ = [self.e[y].prepare(mf) for y in range(self.nb)]

        def J(X):
            cache = {}
            KX = self.Kmix(X)
            return [lam * self.e[y].deriv(mf, X, occ[y], cache) - self.k * nc.jordan(w[y], X[y])
                    - self.k * nc.jordan(S[y], KX[y]) for y in range(self.nb)]

        def Jadj(Lam):
            out = [np.zeros_like(L) for L in Lam]
            for y in range(self.nb):
                self.e[y].adjoint(mf, Lam[y], out, occ[y])
            out = [lam * nc.herm(o) for o in out]
            Y = [nc.jordan(S[y], Lam[y]) for y in range(self.nb)]
            KY = self.Kmix(Y)
            return [out[y] - self.k * nc.jordan(w[y], Lam[y]) - self.k * KY[y] for y in range(self.nb)]

        return J, Jadj, mf

    def precond(self, v):
        """Exact inverse of the Jacobian at S = I, lam = 0: X = -(1/k) G V per sector, G = K^-1."""
        G = getattr(self, "_G", None)
        if G is None:
            G = self._G = np.linalg.inv(self.K)
        out = np.empty_like(v)
        for s in self.sectors:
            ys = self.siteblocks(s)
            V = np.stack([v[self.offsets[y]:self.offsets[y + 1]] for y in ys])
            Xs = -(G @ V) / self.k
            for x, y in enumerate(ys):
                out[self.offsets[y]:self.offsets[y + 1]] = Xs[x]
        return out

    def _solve_linear(self, op_fun, rhs, tol):
        A = LinearOperator((self.N, self.N), dtype=float, matvec=lambda v: self.pack(op_fun(self.unpack(v))))
        Mi = LinearOperator((self.N, self.N), dtype=float, matvec=self.precond)
        # capped: a stagnating GMRES (near a fold) fails fast and the continuation halves its step
        x, code = gmres(A, rhs, M=Mi, rtol=tol, atol=0.0, restart=150, maxiter=3)
        return x, int(code)

    # -- Newton
    def newton(self, S0, b=1.0, lam=1.0, tol=1e-12, maxit=30, gtol=1e-10):
        S = [s.copy() for s in S0]
        hist, codes = [], []
        for _ in range(maxit):
            try:
                Cv = self.pack(self.C(S, b, lam))
            except ValueError:
                return S, {"ok": False, "why": "not positive", "residuals": hist}
            hist.append(float(np.linalg.norm(Cv)))
            if hist[-1] < tol:
                return S, {"ok": True, "residuals": hist, "gmres_codes": codes}
            # rounding floor of k K(S - b) grows with k: accept a stalled iteration below 1e-9 max(1, k)
            if len(hist) > 2 and hist[-1] > 0.3 * hist[-2] and hist[-1] < 1e-9 * max(1.0, self.k):
                return S, {"ok": True, "residuals": hist, "gmres_codes": codes, "stalled": True}
            if len(hist) >= 4 and hist[-1] > 0.9 * hist[-4]:
                return S, {"ok": False, "why": "no decrease in 3 steps", "residuals": hist, "gmres_codes": codes}
            J, _, _ = self.linearization(S, b, lam)
            dv, code = self._solve_linear(J, -Cv, gtol)
            codes.append(code)
            dX = self.unpack(dv)
            step = 1.0
            while True:
                Sn = [nc.herm(s + step * d) for s, d in zip(S, dX)]
                if min(np.linalg.eigvalsh(s).min() for s in Sn) > 0:
                    break
                step *= 0.5
                if step < 1e-4:
                    return S, {"ok": False, "why": "positivity in line search", "residuals": hist}
            S = Sn
        return S, {"ok": hist[-1] < 1e-9 * max(1.0, self.k), "why": "maxit", "residuals": hist, "gmres_codes": codes}

    def identity_start(self, b=1.0):
        return [b * self.eye(y) for y in range(self.nb)]

    def continuation(self, b=1.0, S_start=None, dlam0=0.25, min_dlam=1e-4, tol=1e-12):
        """Regular-branch continuation lam: 0 -> 1 from S = bI; monitors positivity, residual, Newton/GMRES."""
        S = self.identity_start(b) if S_start is None else [s.copy() for s in S_start]
        S, info0 = self.newton(S, b, 0.0, tol)
        path = [{"lam": 0.0, "residual": info0["residuals"][-1] if info0["residuals"] else 0.0,
                 "min_eig": float(min(np.linalg.eigvalsh(s).min() for s in S))}]
        lam, dlam, prev = 0.0, dlam0, None
        while lam < 1.0:
            lt = min(1.0, lam + dlam)
            if prev is not None:
                Sp = [nc.herm(s + (lt - lam) / (lam - prev[0]) * (s - p)) for s, p in zip(S, prev[1])]
                try:
                    BlockMatFun(Sp)
                except ValueError:
                    Sp = S
            else:
                Sp = S
            Sn, info = self.newton(Sp, b, lt, tol, maxit=12)
            if info["ok"] and len(info["residuals"]) <= 12:
                prev = (lam, S)
                S, lam = Sn, lt
                path.append({"lam": lam, "residual": info["residuals"][-1], "newton": len(info["residuals"]) - 1,
                             "min_eig": float(min(np.linalg.eigvalsh(s).min() for s in S)),
                             "gmres_codes": info.get("gmres_codes", [])})
                if len(info["residuals"]) <= 5:
                    dlam = min(2 * dlam, 0.5)
            else:
                dlam *= 0.5
                if dlam < min_dlam:
                    return S, {"ok": False, "lost_at_lam": lam, "why": info.get("why", "newton"), "path": path}
        return S, {"ok": True, "path": path, "final_residual": path[-1]["residual"]}

    # -- outputs and responses
    def Q(self, S, b=1.0, s=None):
        s = self.top if s is None else s
        ys = self.siteblocks(s)
        I = self.eye(ys[0])
        return self.k * sum(self.K[x, xp] * (S[yp] - b * I) for x in range(self.nsite) for xp, yp in enumerate(ys))

    def dS_db(self, S, b=1.0, tol=1e-12):
        """Linear response to b (identical to the sigma response at fixed S): J dS = -k b_x S_y."""
        J, _, _ = self.linearization(S, b)
        rhs = self.pack([-self.k * self.bvec[x] * S[y] for y, (x, s) in enumerate(self.blocks)])
        v, code = self._solve_linear(J, rhs, tol)
        res = float(np.linalg.norm(self.pack(J(self.unpack(v))) - rhs))
        return self.unpack(v), res, code

    def dH_db(self, S, b=1.0):
        dS, res, code = self.dS_db(S, b)
        ys = self.siteblocks(self.top)
        I = self.eye(ys[0])
        dH = -self.k * self.bvec.sum() * I + self.k * sum(self.K[x, xp] * dS[yp] for x in range(self.nsite)
                                                            for xp, yp in enumerate(ys))
        return nc.herm(dH), res, code

    def adjoint_multiplier(self, S, rho, b=1.0, tol=1e-13):
        """J^* Lam = -k b_x rho on the target-sector blocks (zero right side on lower sectors)."""
        _, Jadj, _ = self.linearization(S, b)
        rhs_blocks = []
        for y, (x, s) in enumerate(self.blocks):
            rhs_blocks.append(-self.k * self.bvec[x] * rho if s == self.top else np.zeros_like(S[y]))
        rhs = self.pack(rhs_blocks)
        v, code = self._solve_linear(Jadj, rhs, tol)
        Lam = self.unpack(v)
        res = float(np.linalg.norm(self.pack(Jadj(Lam)) - rhs))
        return Lam, res, code

    def ward(self, S, Lam, Dgen):
        """sum_y Tr Lam_y (i[D_s, e_y(S)] - De_y(S)[i[D, S]]) and the same with the feedback term dropped.
        Dgen maps sector -> number-preserving generator matrix on that sector."""
        mf = BlockMatFun(S)
        X = [1j * (Dgen[s] @ S[y] - S[y] @ Dgen[s]) for y, (x, s) in enumerate(self.blocks)]
        full, nofb = 0.0, 0.0
        cache = {}
        for y, (x, s) in enumerate(self.blocks):
            ey = self.e[y].value(mf)
            comm = 1j * (Dgen[s] @ ey - ey @ Dgen[s])
            fb = self.e[y].deriv(mf, X, None, cache)
            full += np.trace(Lam[y] @ (comm - fb))
            nofb += np.trace(Lam[y] @ comm)
        return complex(full), complex(nofb)

    def sigma_min(self, S, b=1.0, seed=0, tol=1e-8, lam=1.0):
        """Smallest singular value of J by Lanczos on (J^T J)^-1 with preconditioned GMRES inner solves."""
        J, Jadj, _ = self.linearization(S, b, lam)
        rng = np.random.default_rng(seed)

        def inv_jtj(v):
            y, _ = self._solve_linear(Jadj, v, 1e-11)
            x, _ = self._solve_linear(J, y, 1e-11)
            return x

        A = LinearOperator((self.N, self.N), dtype=float, matvec=inv_jtj)
        lam = eigsh(A, k=1, which='LA', v0=rng.normal(size=self.N), tol=tol, maxiter=400, return_eigenvectors=False)
        return float(1.0 / np.sqrt(lam[0]))
