#!/usr/bin/env python
"""pt3_core.py (busi lane, analyst builder; PAGE tests design v3, stage 1). Shared pieces of the v3 build, no physics claim.

Engine code reused unchanged (copies, md5 checked at import by engine_md5s()):
  ../code_v6/page_ops_v6.py (31a85c00): chain_sparse, clocked_csr, binary entropy, eigh helpers;
  engine_copy/engine_ops_v2.py (cdabf12b, = solve/code/engine/engine_ops_v2.py): the paper's R2 continuation (R-code);
  engine_copy/page_w1_sector_v1d.py (0eaed28a, = busi/code/w1/page_w1_sector_v1d.py): the staggered-sector transform.
What is here:
  * the chain h[N, v] = (1/2){N, h_0} + (1/2){v, Q} (m = 0) as host CSR, its + sector block h_+ = W_+^dag h W_+ (n x n, half-bandwidth 2) and
    the full spectrum from h_+ alone (spec h = spec h_+ U -spec h_+, the - sector's vectors conj of the + sector's);
  * the local energy operators h_R = h[N 1_{R' <= R}, v 1_{R' <= R}] = sum_{x <= R} eps_x (eps_x = N_x e_x + v_x p_x, e_x = (1/2){Pi_x, h_0},
    p_x = (1/2){Pi_x, Q}; sum_x eps_x = h exactly), the current operator i[h, h_R], and the engine's per-site densities (engine_collapse_v5.dens);
  * entropies with certified intervals (no clipping): s(x) = -x ln x - (1-x) ln(1-x); for eigenvalues nu_i known to +-eps (Weyl), the interval of
    sum_i s over [nu_i - eps, nu_i + eps] cut to [0, 1]; an eigenvalue farther than eps outside [0, 1] makes the interval invalid (flagged);
  * a JSON writer and md5s."""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.dirname(HERE)
for p in (os.path.join(HERE, "engine_copy"), os.path.join(PT, "code_v6"), PT):
    if p not in sys.path:
        sys.path.insert(0, p)
import page_ops_v6 as OPS  # noqa: E402

ENGINE_MD5 = {"engine_copy/engine_ops_v2.py": "cdabf12b56be61d1d26879b57f073650",
              "engine_copy/page_w1_sector_v1d.py": "0eaed28a4caaa809ea03cc92e2baf4f2",
              "../code_v6/page_ops_v6.py": "31a85c00"}
_T0 = time.time()


def log(msg):
    print("[%8.1f s] %s" % (time.time() - _T0, msg), flush=True)


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def engine_md5s(check=True):
    out = {}
    for rel, want in ENGINE_MD5.items():
        got = md5_file(os.path.normpath(os.path.join(HERE, rel)))
        out[rel] = got
        if check and not got.startswith(want):
            raise RuntimeError("engine copy %s md5 %s != %s" % (rel, got, want))
    return out


def own_md5s():
    return {f: md5_file(os.path.join(HERE, f))[:8] for f in sorted(os.listdir(HERE)) if f.endswith(".py")}


def dump_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return [o.real, o.imag]
        return str(o)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)


# ------------------------------------------------------------------------------------------------------------------ the chain
class ChainOps:
    """the massless kappa-channel chain on sites R = R0 .. R0 + n - 1 (page_ops_v6.chain_sparse; R0 = 1 the engine's sites)."""

    def __init__(self, n, kappa, R0=1.0):
        self.n, self.kappa = int(n), float(kappa)
        self.R = np.arange(self.n, dtype=float) + float(R0)
        self.h0, self.Q = OPS.chain_sparse(self.R, 0.0, self.kappa)
        self.dim = 2 * self.n

    def h(self, N, v):
        return OPS.clocked_csr(self.h0, self.Q, np.asarray(N, float), np.asarray(v, float))

    def h_R(self, N, v, R_cut):
        """sum_{x: R_x <= R_cut} eps_x as a CSR."""
        m = (self.R <= R_cut).astype(float)
        return OPS.clocked_csr(self.h0, self.Q, np.asarray(N, float) * m, np.asarray(v, float) * m)

    def site_mask(self, R_cut, inside=True):
        m = (self.R <= R_cut) if inside else (self.R > R_cut)
        return np.repeat(m, 2)

    def dens(self, A, X, f=None):
        """engine_collapse_v5.dens: per site sum_cols f Re(X^dag Pi_x A X) (A host sparse, X host columns)."""
        AX = A @ X
        if f is not None:
            AX = AX * np.asarray(f)[None, :]
        d = np.real(np.conj(X) * AX).sum(axis=1)
        return d.reshape(-1, 2).sum(axis=1)


def commutator_i(A, B):
    """i[A, B] for sparse A, B."""
    return (1j * (A @ B - B @ A)).tocsr()


def sector_plus(h_csr):
    import page_w1_sector_v1d as W1S
    return W1S.sector_plus_csr(h_csr)


def W_plus(n):
    import page_w1_sector_v1d as W1S
    return W1S.W_plus_sparse(n)


def sector_offblock(h_csr):
    import page_w1_sector_v1d as W1S
    return W1S.sector_blocks_csr(h_csr)[2]


def full_spectrum_from_sector(h_csr):
    """(e (2n,) ascending, U (2n, 2n), sector (2n,) +-1) from one dense eigh of h_+ (the - sector: -e_+, W_- conj(V_+))."""
    n = h_csr.shape[0] // 2
    hp = sector_plus(h_csr).toarray()
    ep, Vp = np.linalg.eigh(hp)
    Wp = W_plus(n)
    Up = Wp @ Vp
    Um = Wp.conj() @ Vp.conj()
    e = np.concatenate([ep, -ep])
    U = np.concatenate([Up, Um], axis=1)
    sec = np.concatenate([np.ones(n, int), -np.ones(n, int)])
    o = np.argsort(e, kind="stable")
    return e[o], U[:, o], sec[o]


def dense_spectrum(h, sector_op=None):
    """(e, U, sector) by a dense eigh of the whole h; sector labels from <u|S|u> rounded when sector_op is given (degenerate blocks rotated)."""
    H = h.toarray() if sp.issparse(h) else np.asarray(h)
    e, U = np.linalg.eigh(H)
    if sector_op is None:
        return e, U, np.zeros(len(e), int)
    S = sector_op.toarray() if sp.issparse(sector_op) else sector_op
    # diagonalize S inside each (near-)degenerate block so the labels are sharp
    i = 0
    while i < len(e):
        j = i + 1
        while j < len(e) and abs(e[j] - e[i]) < 1e-10 * max(1.0, abs(e[i])):
            j += 1
        if j - i > 1:
            M = U[:, i:j].conj().T @ S @ U[:, i:j]
            lam, W = np.linalg.eigh(0.5 * (M + M.conj().T))
            U[:, i:j] = U[:, i:j] @ W
        i = j
    sec = np.rint(np.real(np.einsum("ij,ik,kj->j", U.conj(), S, U))).astype(int)
    return e, U, sec


def window_spectrum_banded(h_csr, lo, hi):
    """(e, U, sector) of the eigenpairs of h with lo < e < hi from the + sector's banded matrix (scipy eig_banded, select='v'), the - sector's
    from the + sector's at (-hi, -lo); the production route on long chains (the + sector's dimension n, bandwidth 2)."""
    from scipy.linalg import eig_banded
    n = h_csr.shape[0] // 2
    hp = sector_plus(h_csr).tocsr()
    ab = np.zeros((3, n), complex)                      # upper form: ab[2 + i - j, j] = a_ij for j >= i
    for d in range(3):
        diag = hp.diagonal(d)
        ab[2 - d, d:] = diag
    e1, V1 = eig_banded(ab, lower=False, select="v", select_range=(lo, hi))
    e2, V2 = eig_banded(ab, lower=False, select="v", select_range=(-hi, -lo))
    Wp = W_plus(n)
    U = np.concatenate([Wp @ V1, Wp.conj() @ V2.conj()], axis=1)
    e = np.concatenate([e1, -e2])
    sec = np.concatenate([np.ones(len(e1), int), -np.ones(len(e2), int)])
    o = np.argsort(e, kind="stable")
    return e[o], U[:, o], sec[o]


def count_below_banded(h_csr, x):
    """the number of eigenvalues of h_+ below x and of h_- = -conj(h_+) below x (eigenvalues only, select range)."""
    from scipy.linalg import eigvals_banded
    n = h_csr.shape[0] // 2
    hp = sector_plus(h_csr).tocsr()
    ab = np.zeros((3, n), complex)
    for d in range(3):
        ab[2 - d, d:] = hp.diagonal(d)
    lo = -10.0 * (abs(hp).sum(axis=1).max() + 1.0)
    cp = len(eigvals_banded(ab, lower=False, select="v", select_range=(lo, x)))
    cm = len(eigvals_banded(ab, lower=False, select="v", select_range=(-x, -lo)))
    return cp, cm


# ------------------------------------------------------------------------------------------------------------- the entropies
def s_bin(x):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > 0) & (x < 1)
    xm = x[m]
    out[m] = -xm * np.log(xm) - (1 - xm) * np.log1p(-xm)
    return out


def s_interval(nu, eps):
    """certified [lo, hi] of sum_i s(nu_i') over |nu_i' - nu_i| <= eps, nu_i' in [0, 1]; (lo, hi, valid, n_outside) with valid False when some
    nu_i lies farther than eps outside [0, 1] (an unphysical spectrum the bound cannot cover; never clipped)."""
    nu = np.asarray(nu, float)
    eps = float(max(eps, 0.0))
    out = (nu < -eps) | (nu > 1 + eps)
    a = np.clip(nu - eps, 0.0, 1.0)
    b = np.clip(nu + eps, 0.0, 1.0)
    sa, sb = s_bin(a), s_bin(b)
    lo = np.minimum(sa, sb)                              # concave: the minimum on [a, b] is at an end
    hi = np.where((a <= 0.5) & (b >= 0.5), math.log(2.0), np.maximum(sa, sb))
    return float(lo.sum()), float(hi.sum()), bool(not out.any()), int(out.sum())


def cov_entropy(Cblk, eps=0.0):
    """S of a Gaussian (number-conserving) state's restriction with covariance block Cblk: the point value sum s(nu), the certified interval at
    eigenvalue error eps (plus the Hermiticity defect), and the spectrum's excursion outside [0, 1]."""
    C = np.asarray(Cblk)
    herm = float(np.max(np.abs(C - C.conj().T))) if C.size else 0.0
    nu = np.linalg.eigvalsh(0.5 * (C + C.conj().T)) if C.size else np.zeros(0)
    # Weyl: the error bound plus the Hermiticity defect (times the dimension) plus a rounding allowance of 64 machine epsilons per unit of
    # the block's spectral scale (eigvalsh's backward error), never a clip
    e_tot = float(eps) + herm * max(1, C.shape[0]) + 64 * 2.2e-16 * max(1.0, float(np.max(np.abs(nu))) if len(nu) else 1.0) * max(1, C.shape[0]) ** 0.5
    lo, hi, valid, nout = s_interval(nu, e_tot)
    exc = float(max(0.0, -nu.min(), nu.max() - 1.0)) if len(nu) else 0.0
    return {"S": float(s_bin(np.clip(nu, 0, 1)).sum()), "S_lo": lo, "S_hi": hi, "valid": valid, "n_outside": nout, "excursion": exc,
            "eps_used": e_tot, "herm_defect": herm, "dim": int(C.shape[0])}
