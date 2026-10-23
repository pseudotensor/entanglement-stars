#!/usr/bin/env python3
"""obs_ball_cut_v1.py (builder analyst, .lane-alt solve lane; task S6 = J4 of REVIEWA_REVIEW_web40e, the lead's ruling in
SPEC_final4b_engine.md sec. 28.5): the genuine 3-dimensional ball cut of the flat massless naive lattice Dirac sea, by the review's exact
reductions, validated against a dense 4-component computation on the same ball.

Model (a = 1): h = sum_i alpha_i sin(p_i) on an L^3 box (L even), antiperiodic on all three axes, k_i = 2 pi (j_i + 1/2)/L; the sea
P(k) = (1/2)[I_4 - sum_i alpha_i sin k_i / E(k)], E = (sum_i sin^2 k_i)^(1/2); the ball A = {x in Z^3 : |x| <= R} centred on a site;
S(A) = Tr s(C_A), s(x) = -x ln x - (1 - x) ln(1 - x), natural logarithms.

Kernel.  C(x, y) = (1/2) delta_xy I_4 - (1/2) sum_i alpha_i G_i(x - y), G_i(r) = L^-3 sum_k (sin k_i / E(k)) e^{i k.r} = i g_i(r), g_i real;
g_1(r) = (8/L^3) sum over the octant k_j in (0, pi) of sin k_1 sin(k_1 r_1) cos(k_2 r_2) cos(k_3 r_3) / E(k) (the summand is even in each
k_j), by three tensor contractions with exactly reduced phases ((2n + 1) r mod 2L); g_2, g_3 by coordinate permutation.  On the even
antiperiodic grid k_j -> pi - k_j is a symmetry, so g_i(r) = 0 unless r_i is odd and the other two components are even (printed per table).

Reductions (the review's S6; each is checked numerically by the validate mode):
 (R1) Omega_x = alpha_1^{x_1} alpha_2^{x_2} alpha_3^{x_3}: Omega_x^dag alpha_i Omega_{x+e_i} = eta_i(x) I_4, eta_i(x) = (-1)^{sum_{j<i} x_j},
      Omega_{x+2e_j} = Omega_x, so Omega^dag C Omega = I_4 (x) C_+, C_+(x, y) = (1/2) delta_xy - (1/2) eta_i(x) G_i(x - y) with i the unique
      odd component of x - y (zero otherwise): S = 4 S_+.
 (R2) on (even, odd) sites C_+ = 1/2 - (i/2)[[0, K], [-K^T, 0]], K(x, y) = eta_i(x) g_i(x - y) real; the eigenvalues of C_+ on the ball are
      (1 +- s_j)/2 (s_j the singular values of K on the ball) and |n_e - n_o| times 1/2: S_+ = 2 sum_j s((1 + s_j)/2) + |n_e - n_o| ln 2.
 (R3) O_h (48 signed permutations g, x -> g x) acts on scalar functions by (V_g f)(x) = phi_g(x) f(g^-1 x), phi_g(x) I_4 =
      Omega_x^dag rho_g(Omega_{g^-1 x}), rho_g(alpha_i) = sum_j g_ji alpha_j: an ordinary representation preserving the sublattices with
      V^e_g K = K V^o_g; with the ten real irreps A1+-, A2+-, E+-, T1+-, T2+- (dims 1, 1, 2, 3, 3), K = (+)_lambda I_{d_lambda} (x) K_lambda,
      K_lambda[(x, k), (y, l)] = (At_x^T U(x, y) At_y)_{kl},  U(x, y) = sum_g D^lambda(g) phi_g(g y) K(x, g y)  (x, y orbit
      representatives x_1 >= x_2 >= x_3 >= 0), At_x = Q_x |H_x|^(-1/2), Q_x the eigenvectors of eigenvalue |H_x| of
      sum_{g in H_x} D^lambda(g) phi_g(x) (H_x the stabilizer; Q = I, At = I on free orbits):
      S = 4 sum_lambda d_lambda [2 sum_j s((1 + s_lambda_j)/2) + |m_lambda_e - m_lambda_o| ln 2].
 Checked at every run: sum_lambda d_lambda m_lambda_{e,o} = n_{e,o}; sum_lambda d_lambda ||K_lambda||_F^2 = ||K||_F^2 =
 sum_r g(r)^2 N_eo(r) (N_eo(r) = #{x even, y odd in the ball, x - y = r}, by FFT correlation of the masks).

Modes:
  validate --R --L [--ref]   one ball: the dense 4-component C_A (eigvalsh), the numerical Omega transform against the analytic C_+, the scalar
                             C_+ (eigvalsh), the full sublattice SVD, the irrep blocks; group, phase, commutation and kernel checks; JSON + md.
  run --R ... [--offsets16] [--box-ratios]   the production ladder by the irrep blocks; one JSONL line per (R, offset, L), one JSON per R.
  counts --R ... [--offsets16]   the exact multiplicities, block sizes and memory without kernel or SVD.
  bench --sizes ...          CPU timing of real FP64 SVD values (gesdd) and Gram eigvalsh, and complex128 eigvalsh at n = 4096 (the
                             lane's logged A6000/H100 calibration point).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import hashlib
import itertools
import json
import math
import socket
import sys
import time

import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla

LN2 = math.log(2.0)
OFFSETS16 = (-7, -5, -3, -1, 1, 3, 5, 7)
SCRIPT = "obs_ball_cut_v1.py"

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1.0, -1.0]).astype(complex)
I2 = np.eye(2, dtype=complex)
ALPHA = np.array([np.kron(SX, s) for s in (SX, SY, SZ)])  # the chiral representation of the lane's strand-3 code
BETA = np.kron(SZ, I2)
I4 = np.eye(4, dtype=complex)
IRREPS = ("A1+", "A2+", "E+", "T1+", "T2+", "A1-", "A2-", "E-", "T1-", "T2-")


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg):
    print("[%s] %s" % (utc(), msg), flush=True)


# ----------------------------------------------------------------------------------------------------------------------------- entropy
def hbin(x):
    """s(x) = -x ln x - (1 - x) ln(1 - x), for x in [0, 1/2]."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    ok = x > 0
    xo = x[ok]
    out[ok] = -xo * np.log(xo) - (1.0 - xo) * np.log1p(-xo)
    return out


def S_from_eigs(lam):
    lam = np.clip(np.asarray(lam, float), 0.0, 1.0)
    return float(hbin(np.minimum(lam, 1.0 - lam)).sum())


def S_from_svals(s):
    """2 sum_j s((1 + s_j)/2), evaluated as 2 sum_j s((1 - s_j)/2) (s clipped to [0, 1])."""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    return 2.0 * float(hbin(0.5 * (1.0 - s)).sum())


# ------------------------------------------------------------------------------------------------------------------------------ kernel
def kernel_g1(L, rmax):
    """g_1(r), r in [-rmax, rmax]^3 (array index r + rmax)."""
    if L % 2:
        raise ValueError("L must be even")
    h = L // 2
    n = np.arange(h)
    k = np.pi * (2 * n + 1) / L
    s = np.sin(k)
    s2 = s * s
    rr = np.arange(-rmax, rmax + 1)
    ang = np.pi * np.mod(np.outer(rr, 2 * n + 1), 2 * L) / L  # k_n r reduced exactly
    A1 = np.sin(ang) * s[None, :]
    C = np.cos(ang)
    Einv = 1.0 / np.sqrt(s2[:, None, None] + s2[None, :, None] + s2[None, None, :])
    T1 = np.tensordot(Einv, C, axes=([2], [1]))  # (n1, n2, r3)
    del Einv
    T2 = np.tensordot(T1, C, axes=([1], [1]))  # (n1, r3, r2)
    del T1
    T3 = np.tensordot(A1, T2, axes=([1], [0]))  # (r1, r3, r2)
    del T2
    return np.ascontiguousarray(T3.transpose(0, 2, 1)) * (8.0 / float(L) ** 3)


def G_direct(L, r, i):
    """G_i(r) by the direct sum over the whole antiperiodic grid (complex; small L only)."""
    k = np.pi * (2 * np.arange(L) + 1) / L
    K1, K2, K3 = np.meshgrid(k, k, k, indexing="ij")
    Ks = (K1, K2, K3)
    E = np.sqrt(np.sin(K1) ** 2 + np.sin(K2) ** 2 + np.sin(K3) ** 2)
    ph = np.exp(1j * (K1 * r[0] + K2 * r[1] + K3 * r[2]))
    return complex((np.sin(Ks[i]) / E * ph).sum() / float(L) ** 3)


def kernel_tables(L, rmax):
    """g_1 (raw), the per-axis g_i (unmasked, for the dense check), the masked table gtab(r) = g_i(r) for r with exactly one odd component i,
    itab(r) = i (3 where none), and the largest |g_1| at parities where it must vanish."""
    G1 = kernel_g1(L, rmax)
    rr = np.arange(-rmax, rmax + 1)
    odd = (np.abs(rr) % 2) == 1
    o1, o2, o3 = odd[:, None, None], odd[None, :, None], odd[None, None, :]
    a1 = np.broadcast_to(o1 & ~o2 & ~o3, G1.shape)
    a2 = np.broadcast_to(~o1 & o2 & ~o3, G1.shape)
    a3 = np.broadcast_to(~o1 & ~o2 & o3, G1.shape)
    forbidden = float(np.abs(G1[~a1]).max()) if (~a1).any() else 0.0
    g2 = G1.transpose(1, 0, 2)
    g3 = G1.transpose(2, 1, 0)
    gtab = np.where(a1, G1, 0.0) + np.where(a2, g2, 0.0) + np.where(a3, g3, 0.0)
    itab = np.full(G1.shape, 3, np.int8)
    itab[a1] = 0
    itab[a2] = 1
    itab[a3] = 2
    return dict(G1=G1, gi=(G1, g2, g3), gtab=gtab, itab=itab, forbidden=forbidden)


def signed_tables(gtab, itab):
    """sgt[q](r) = eta_{i(r)}(x) g_{i(r)}(r) for the site-parity class q = 2 (x_1 mod 2) + (x_2 mod 2)."""
    out = []
    for q in range(4):
        b1, b2 = q >> 1, q & 1
        eta = np.array([1.0, -1.0 if b1 else 1.0, -1.0 if (b1 ^ b2) else 1.0, 0.0])
        out.append(np.ascontiguousarray(gtab * eta[itab]))
    return out


def lin_index(Z, W):
    Z = np.asarray(Z, np.int64)
    return (Z[..., 0] * W + Z[..., 1]) * W + Z[..., 2]


# ------------------------------------------------------------------------------------------------------------------------------- group
def cubic_group():
    G = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            g = np.zeros((3, 3), np.int64)
            for i in range(3):
                g[perm[i], i] = signs[i]  # g e_i = sign_i e_perm(i)
            G.append(g)
    G.sort(key=lambda g: 0 if np.array_equal(g, np.eye(3, dtype=np.int64)) else 1)
    return np.array(G)


def omega_p(p):
    M = I4.copy()
    for i in range(3):
        if int(p[i]) % 2:
            M = M @ ALPHA[i]
    return M


def parity_index(X):
    X = np.asarray(X, np.int64)
    return (X[..., 0] & 1) * 4 + (X[..., 1] & 1) * 2 + (X[..., 2] & 1)


PCLASSES = np.array([[(p >> 2) & 1, (p >> 1) & 1, p & 1] for p in range(8)], np.int64)


class Cubic:
    def __init__(self):
        G = cubic_group()
        self.G = G
        key = {g.tobytes(): i for i, g in enumerate(G)}
        self.mul = np.array([[key[(G[a] @ G[b]).tobytes()] for b in range(48)] for a in range(48)])
        self.inv = np.array([key[G[a].T.copy().tobytes()] for a in range(48)])
        B = np.array([[1 / math.sqrt(2), -1 / math.sqrt(2), 0.0], [1 / math.sqrt(6), 1 / math.sqrt(6), -2 / math.sqrt(6)]]).T
        D = {nm: [] for nm in IRREPS}
        for g in G:
            p = int(round(np.linalg.det(g)))
            r = (p * g).astype(float)
            P = np.abs(g).astype(float)
            sgn = float(round(np.linalg.det(P)))
            base = {"A1": np.eye(1), "A2": np.array([[sgn]]), "E": B.T @ P @ B, "T1": r, "T2": sgn * r}
            for nm, M in base.items():
                D[nm + "+"].append(M)
                D[nm + "-"].append(p * M)
        self.D = {nm: np.array(D[nm], float) for nm in IRREPS}
        self.dim = {nm: self.D[nm].shape[1] for nm in IRREPS}
        cols, c0, self.cols = [], 0, {}
        for nm in IRREPS:
            d = self.dim[nm]
            cols.append(self.D[nm].reshape(48, d * d))
            self.cols[nm] = slice(c0, c0 + d * d)
            c0 += d * d
        self.F = np.ascontiguousarray(np.concatenate(cols, axis=1))  # (48, 48): F[g, (lambda, b, c)] = D^lambda_bc(g)
        # phi_g(x) on the eight parity classes: Omega_x^dag rho_g(Omega_{g^-1 x}) = phi I_4
        phi = np.zeros((48, 8))
        dev = 0.0
        for gi, g in enumerate(G):
            rho = [sum(g[j, i] * ALPHA[j] for j in range(3)) for i in range(3)]
            for pidx in range(8):
                pv = PCLASSES[pidx]
                z = (np.abs(g.T) @ pv) % 2
                Rz = I4.copy()
                for i in range(3):
                    if z[i]:
                        Rz = Rz @ rho[i]
                M = omega_p(pv).conj().T @ Rz
                c = M[0, 0]
                dev = max(dev, float(np.abs(M - c * I4).max()), abs(abs(c) - 1.0), abs(c.imag))
                phi[gi, pidx] = c.real
        self.phi = phi
        self.phi_dev = dev

    def checks(self):
        out = {}
        G, mul = self.G, self.mul
        hom = 0.0
        for nm in IRREPS:
            Dn = self.D[nm]
            prod = np.einsum("aij,bjk->abik", Dn, Dn)
            hom = max(hom, float(np.abs(prod - Dn[mul]).max()))
        out["irrep_hom_maxdev"] = hom
        chi = np.array([[np.trace(self.D[nm][g]) for g in range(48)] for nm in IRREPS])
        out["character_orth_maxdev"] = float(np.abs(chi @ chi.T / 48.0 - np.eye(len(IRREPS))).max())
        Fo = self.F.T @ self.F  # sum_g D^l_bc(g) D^m_b'c'(g) = (48/d) delta
        dvec = np.concatenate([np.full(self.dim[nm] ** 2, 48.0 / self.dim[nm]) for nm in IRREPS])
        out["schur_orth_maxdev"] = float(np.abs(Fo - np.diag(dvec)).max())
        # phi: phi_{gh}(x) = phi_g(x) phi_h(g^-1 x) on every parity class
        pdev = 0.0
        for a in range(48):
            for b in range(48):
                for pidx in range(8):
                    z = parity_index((np.abs(G[a].T) @ PCLASSES[pidx]) % 2)
                    pdev = max(pdev, abs(self.phi[mul[a, b], pidx] - self.phi[a, pidx] * self.phi[b, z]))
        out["phi_rep_maxdev"] = pdev
        out["phi_scalar_maxdev"] = self.phi_dev
        # Omega relations
        odev = 0.0
        for pidx in range(8):
            pv = PCLASSES[pidx]
            for i in range(3):
                e = np.zeros(3, np.int64)
                e[i] = 1
                eta = (-1.0) ** int(pv[:i].sum())
                odev = max(odev, float(np.abs(omega_p(pv).conj().T @ ALPHA[i] @ omega_p(pv + e) - eta * I4).max()))
                odev = max(odev, float(np.abs(omega_p(pv + 2 * e) - omega_p(pv)).max()))
        out["omega_relation_maxdev"] = odev
        return out


# ------------------------------------------------------------------------------------------------------------------------ ball, orbits
def ball_sites(Reff):
    R2 = float(Reff) * float(Reff)
    b = int(math.floor(Reff))
    r = np.arange(-b, b + 1, dtype=np.int64)
    X = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    return X[(X * X).sum(1) <= R2]


def mask_hash(X):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(X, np.int32)).tobytes()).hexdigest()


def orbit_data(reps, cub):
    """per representative: orbit size, stabilizer order, and per irrep the padded At (d x d, kept columns first) and the multiplicity."""
    GX = np.einsum("gij,xj->xgi", cub.G, reps)
    stab = np.all(GX == reps[:, None, :], axis=2)
    hsize = stab.sum(1)
    par = parity_index(reps)
    w = stab * cub.phi[:, par].T
    At, mult = {}, {}
    gdev = 0.0
    for nm in IRREPS:
        Gam = np.einsum("xg,gab->xab", w, cub.D[nm])
        ev, Q = np.linalg.eigh(Gam)
        kept = ev > 0.5 * hsize[:, None]
        tgt = np.where(kept, hsize[:, None].astype(float), 0.0)
        gdev = max(gdev, float(np.abs(ev - tgt).max()))
        order = np.argsort(~kept, axis=1, kind="stable")
        Qs = np.take_along_axis(Q, order[:, None, :], axis=2)
        ks = np.take_along_axis(kept, order, axis=1)
        At[nm] = np.ascontiguousarray(Qs * ks[:, None, :] / np.sqrt(hsize)[:, None, None])
        mult[nm] = ks.sum(1).astype(np.int64)
    size = 48 // hsize
    per_orbit = sum(cub.dim[nm] * mult[nm] for nm in IRREPS)
    return dict(reps=reps, size=size, hsize=hsize, At=At, mult=mult, gram_dev=gdev,
                orbit_dim_check=int(np.abs(per_orbit - size).max()))


def ball_structure(Reff, cub):
    X = ball_sites(Reff)
    ev = (X.sum(1) % 2) == 0
    Xe, Xo = X[ev], X[~ev]
    rep = lambda Z: Z[(Z[:, 0] >= Z[:, 1]) & (Z[:, 1] >= Z[:, 2]) & (Z[:, 2] >= 0)]
    oe, oo = orbit_data(rep(Xe), cub), orbit_data(rep(Xo), cub)
    out = dict(Reff=float(Reff), X=X, Xe=Xe, Xo=Xo, oe=oe, oo=oo, n=len(X), n_e=len(Xe), n_o=len(Xo),
               max_x2=int((X * X).sum(1).max()), sha1=mask_hash(X))
    out["orbit_sum_check"] = [int(oe["size"].sum()) - len(Xe), int(oo["size"].sum()) - len(Xo)]
    out["m"] = {nm: [int(oe["mult"][nm].sum()), int(oo["mult"][nm].sum())] for nm in IRREPS}
    out["mult_sum_check"] = [sum(cub.dim[nm] * out["m"][nm][0] for nm in IRREPS) - len(Xe),
                             sum(cub.dim[nm] * out["m"][nm][1] for nm in IRREPS) - len(Xo)]
    return out


# ----------------------------------------------------------------------------------------------------------------------------- K, blocks
def K_full(Xe, Xo, sgt, rmax):
    W = 2 * rmax + 1
    off = int(lin_index(np.array([rmax] * 3), W))
    idx = (lin_index(Xe, W) + off)[:, None] - lin_index(Xo, W)[None, :]
    q = (Xe[:, 0] & 1) * 2 + (Xe[:, 1] & 1)
    K = np.empty(idx.shape)
    for qq in range(4):
        m = q == qq
        K[m] = sgt[qq].ravel()[idx[m]]
    return K


def assemble_blocks(bs, cub, sgt, rmax, max_elems=2.5e7):
    oe, oo = bs["oe"], bs["oo"]
    W = 2 * rmax + 1
    off = int(lin_index(np.array([rmax] * 3), W))
    Y = oo["reps"]
    ny = len(Y)
    GY = np.einsum("gij,yj->ygi", cub.G, Y)
    PhY = cub.phi[np.arange(48)[None, :], parity_index(GY)]
    linGY = lin_index(GY, W)
    Xr = oe["reps"]
    q = (Xr[:, 0] & 1) * 2 + (Xr[:, 1] & 1)
    baseX = lin_index(Xr, W) + off
    blocks = {nm: np.zeros((bs["m"][nm][0], bs["m"][nm][1])) for nm in IRREPS}
    rowoff = {nm: 0 for nm in IRREPS}
    colmask = {nm: (np.arange(cub.dim[nm])[None, :] < oo["mult"][nm][:, None]).ravel() for nm in IRREPS}
    cx = max(1, int(max_elems // (48 * ny)))
    for qq in range(4):
        sel = np.nonzero(q == qq)[0]
        tab = sgt[qq].ravel()
        for c0 in range(0, len(sel), cx):
            ch = sel[c0:c0 + cx]
            idx = baseX[ch][:, None, None] - linGY[None, :, :]
            v = tab[idx]
            v *= PhY[None, :, :]
            U = (v.reshape(-1, 48) @ cub.F).reshape(len(ch), ny, 48)
            del v, idx
            for nm in IRREPS:
                d = cub.dim[nm]
                Ul = U[:, :, cub.cols[nm]].reshape(len(ch), ny, d, d)
                Ul = np.matmul(np.transpose(oe["At"][nm][ch], (0, 2, 1))[:, None], Ul)
                Ul = np.matmul(Ul, oo["At"][nm][None])
                Kc = Ul.transpose(0, 2, 1, 3).reshape(len(ch) * d, ny * d)
                rm = (np.arange(d)[None, :] < oe["mult"][nm][ch][:, None]).ravel()
                Kc = Kc[rm][:, colmask[nm]]
                blocks[nm][rowoff[nm]:rowoff[nm] + Kc.shape[0]] = Kc
                rowoff[nm] += Kc.shape[0]
            del U
    for nm in IRREPS:
        assert rowoff[nm] == blocks[nm].shape[0]
    return blocks


def frob_direct(bs, gtab, rmax):
    """||K||_F^2 = sum_r gtab(r)^2 N_eo(r), N_eo the even-odd mask correlation (FFT, rounded to integers)."""
    X = bs["X"]
    b = int(np.abs(X).max())
    P = 4 * b + 2
    me = np.zeros((P, P, P))
    mo = np.zeros((P, P, P))
    Xe, Xo = bs["Xe"] + b, bs["Xo"] + b
    me[Xe[:, 0], Xe[:, 1], Xe[:, 2]] = 1.0
    mo[Xo[:, 0], Xo[:, 1], Xo[:, 2]] = 1.0
    corr = sfft.irfftn(sfft.rfftn(me, workers=8) * np.conj(sfft.rfftn(mo, workers=8)), s=(P, P, P), workers=8)
    Neo = np.rint(corr).astype(np.int64)
    rounding = float(np.abs(corr - Neo).max())
    rr = np.arange(-2 * b, 2 * b + 1)
    Nsub = Neo[np.ix_(rr % P, rr % P, rr % P)]  # N_eo(r) = sum_x me(x) mo(x - r)
    gsub = gtab[np.ix_(rr + rmax, rr + rmax, rr + rmax)]
    return float((gsub * gsub * Nsub).sum()), int(Nsub.sum()), rounding


# ------------------------------------------------------------------------------------------------------------------------------ solvers
def singular_values(K, solver="svd", device="cpu"):
    """the min(m, n) singular values of K; svd = LAPACK gesdd values (QR + gesdd, then gesvd, on failure); gram = eigvalsh of the smaller
    Gram matrix; device cuda:N runs either route by torch."""
    m, n = K.shape
    if min(m, n) == 0:
        return np.zeros(0), "empty"
    if device != "cpu":
        import torch
        T = torch.as_tensor(K, device=device)
        if solver == "svd":
            s = torch.linalg.svdvals(T).cpu().numpy()
            return s, "torch.svdvals"
        Gm = T @ T.T if m <= n else T.T @ T
        mu = torch.linalg.eigvalsh(Gm).cpu().numpy()
        return np.sqrt(np.clip(mu, 0.0, None)), "torch.gram_eigvalsh"
    if solver == "gram":
        Gm = K @ K.T if m <= n else K.T @ K
        mu = sla.eigvalsh(Gm, check_finite=False, overwrite_a=True)
        return np.sqrt(np.clip(mu, 0.0, None)), "gram_eigvalsh"
    try:
        return np.linalg.svd(K, compute_uv=False), "gesdd"
    except np.linalg.LinAlgError:
        try:
            Rq = np.linalg.qr(K if m >= n else K.T, mode="r")
            return np.linalg.svd(Rq, compute_uv=False), "qr+gesdd"
        except np.linalg.LinAlgError:
            return sla.svd(K, compute_uv=False, lapack_driver="gesvd"), "gesvd"


def block_entropies(blocks, cub, solver="svd", device="cpu", check_gram=False):
    rows, Ssum, frob = {}, 0.0, 0.0
    for nm in IRREPS:
        Kb = blocks[nm]
        me, mo = Kb.shape
        fb = float((Kb * Kb).sum())
        t0 = time.time()
        s, route = singular_values(Kb, solver, device)
        ts = time.time() - t0
        Sb = S_from_svals(s) + abs(me - mo) * LN2
        row = dict(d=cub.dim[nm], m_e=me, m_o=mo, S_block=Sb, smax=float(s.max()) if len(s) else 0.0,
                   smin=float(s.min()) if len(s) else 0.0, n_above_1=int((s > 1.0).sum()), sum_s2=float((s * s).sum()), frob2=fb,
                   t_solve=ts, route=route)
        if check_gram and solver == "svd" and min(me, mo) > 0:
            t0 = time.time()
            s2, r2 = singular_values(Kb, "gram", device)
            row["t_gram"] = time.time() - t0
            row["S_block_gram_minus_svd"] = (S_from_svals(s2) + abs(me - mo) * LN2) - Sb
        rows[nm] = row
        Ssum += cub.dim[nm] * Sb
        frob += cub.dim[nm] * fb
    return rows, 4.0 * Ssum, frob


# ------------------------------------------------------------------------------------------------------------------------------- modes
def index_lookup(X):
    b = int(np.abs(X).max())
    W = 2 * b + 1
    tab = -np.ones((W, W, W), np.int64)
    tab[X[:, 0] + b, X[:, 1] + b, X[:, 2] + b] = np.arange(len(X))
    return lambda Z: tab[Z[..., 0] + b, Z[..., 1] + b, Z[..., 2] + b]


def mode_validate(a):
    cub = Cubic()
    t0 = time.time()
    gchk = cub.checks()
    Reff, L = a.R, a.L
    rmax = 2 * int(math.ceil(Reff)) + 2
    tabs = kernel_tables(L, rmax)
    sgt = signed_tables(tabs["gtab"], tabs["itab"])
    bs = ball_structure(Reff, cub)
    X, Xe, Xo, n = bs["X"], bs["Xe"], bs["Xo"], bs["n"]
    rec = dict(script=SCRIPT, mode="validate", host=socket.gethostname(), utc_start=utc(), R=Reff, L=L, rmax=rmax, n=n, n_e=bs["n_e"],
               n_o=bs["n_o"], max_x2=bs["max_x2"], mask_sha1=bs["sha1"], group_checks=gchk)
    # kernel checks: the parity selection rule, the direct sum at a few displacements
    rec["kernel_forbidden_parity_max"] = tabs["forbidden"]
    if L <= 128:
        dd = 0.0
        for r in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [3, 2, 0], [-1, 4, 2], [2, -3, 4], [5, 2, -2], [1, 1, 1], [2, 2, 0]):
            for i in range(3):
                gd = G_direct(L, np.array(r), i)
                gt = 1j * tabs["gi"][i][r[0] + rmax, r[1] + rmax, r[2] + rmax]
                dd = max(dd, abs(gd - gt))
        rec["kernel_direct_sum_maxdev"] = dd
    # (0) the dense 4-component C_A
    W = 2 * rmax + 1
    off = int(lin_index(np.array([rmax] * 3), W))
    idx = (lin_index(X, W) + off)[:, None] - lin_index(X, W)[None, :]
    C4 = np.zeros((n, 4, n, 4), complex)
    for i in range(3):
        gi = tabs["gi"][i].ravel()[idx]
        C4 += (-0.5j) * gi[:, None, :, None] * ALPHA[i][None, :, None, :]
    C4 += 0.5 * np.eye(n)[:, None, :, None] * I4[None, :, None, :]
    t1 = time.time()
    lam4 = sla.eigvalsh(C4.reshape(4 * n, 4 * n), check_finite=False)
    S4 = S_from_eigs(lam4)
    rec["S_dense4"] = S4
    rec["dense4_eig_range"] = [float(lam4.min()), float(lam4.max())]
    rec["t_dense4_eigvalsh"] = time.time() - t1
    log("validate R %g L %d: n %d, S_dense4 %.12f (eigvalsh %.1f s)" % (Reff, L, n, S4, rec["t_dense4_eigvalsh"]))
    # (1) the Omega transform, numerically, against the analytic C_+
    Om = np.array([omega_p(p) for p in PCLASSES])
    OmX = Om[parity_index(X)]
    T = np.einsum("xba,xbyc,ycd->xayd", OmX.conj(), C4, OmX, optimize=True)
    c = T[:, 0, :, 0].copy()
    offd = 0.0
    diagd = 0.0
    for aa in range(4):
        for dd_ in range(4):
            if aa == dd_:
                diagd = max(diagd, float(np.abs(T[:, aa, :, aa] - c).max()))
            else:
                offd = max(offd, float(np.abs(T[:, aa, :, dd_]).max()))
    del T, C4
    q = (X[:, 0] & 1) * 2 + (X[:, 1] & 1)
    Cp = np.empty((n, n), complex)
    for qq in range(4):
        m = q == qq
        Cp[m] = -0.5j * sgt[qq].ravel()[idx[m]]
    Cp += 0.5 * np.eye(n)
    rec["omega_transform"] = dict(internal_offdiag_max=offd, internal_diag_spread_max=diagd,
                                  numerical_minus_analytic_Cplus_max=float(np.abs(c - Cp).max()),
                                  Cplus_hermiticity_max=float(np.abs(Cp - Cp.conj().T).max()))
    ev = (X.sum(1) % 2) == 0
    rec["same_sublattice_offdiag_max"] = float(max(np.abs(Cp[np.ix_(ev, ev)] - 0.5 * np.eye(ev.sum())).max(),
                                                   np.abs(Cp[np.ix_(~ev, ~ev)] - 0.5 * np.eye((~ev).sum())).max()))
    lamp = sla.eigvalsh(Cp, check_finite=False)
    Sp_num = S_from_eigs(sla.eigvalsh(c, check_finite=False))
    Sp = S_from_eigs(lamp)
    rec["S_scalar_x4_numerical_transform"] = 4 * Sp_num
    rec["S_scalar_x4"] = 4 * Sp
    # (2) the full sublattice SVD
    K = K_full(Xe, Xo, sgt, rmax)
    sK = np.linalg.svd(K, compute_uv=False)
    S_sub = 4 * (S_from_svals(sK) + abs(len(Xe) - len(Xo)) * LN2)
    rec["S_sublattice_svd"] = S_sub
    rec["K_svals_max"] = float(sK.max())
    rec["frob_K_full"] = float((K * K).sum())
    # (3) symmetry: V^e_g K = K V^o_g on the sublattice sites, by index arithmetic
    look_e, look_o = index_lookup(Xe), index_lookup(Xo)
    cdev = 0.0
    for gi in range(48):
        g = cub.G[gi]
        ginv = cub.G[cub.inv[gi]]
        # (V^e_g K)(x, y) = phi_g(x) K(g^-1 x, y);  (K V^o_g)(x, y) = K(x, g y) phi_g(g y)
        rows = look_e(Xe @ ginv.T)
        cols = look_o(Xo @ g.T)
        assert (rows >= 0).all() and (cols >= 0).all()
        left = cub.phi[gi, parity_index(Xe)][:, None] * K[rows, :]
        right = K[:, cols] * cub.phi[gi, parity_index(Xo @ g.T)][None, :]
        cdev = max(cdev, float(np.abs(left - right).max()))
    rec["commutation_VK_minus_KV_max"] = cdev
    # character multiplicities against the Gram ranks
    chk = {}
    for side, Z in (("e", Xe), ("o", Xo)):
        look = look_e if side == "e" else look_o
        chiV = np.zeros(48)
        for gi in range(48):
            fixed = np.all(Z @ cub.G[gi].T == Z, axis=1)
            chiV[gi] = cub.phi[gi, parity_index(Z[fixed])].sum()
        for nm in IRREPS:
            chi = np.array([np.trace(cub.D[nm][g]) for g in range(48)])
            chk[nm + "_" + side] = [float(chiV @ chi / 48.0), bs["m"][nm][0 if side == "e" else 1]]
    rec["multiplicity_character_vs_gram"] = chk
    rec["multiplicities"] = bs["m"]
    rec["orbit_checks"] = dict(orbit_sum=bs["orbit_sum_check"], mult_sum=bs["mult_sum_check"],
                               gram_dev=[bs["oe"]["gram_dev"], bs["oo"]["gram_dev"]],
                               orbit_dim=[bs["oe"]["orbit_dim_check"], bs["oo"]["orbit_dim_check"]])
    # (4) the irrep blocks
    t1 = time.time()
    blocks = assemble_blocks(bs, cub, sgt, rmax)
    rows, S_irr, frob_b = block_entropies(blocks, cub, "svd", "cpu", check_gram=True)
    rec["t_irrep"] = time.time() - t1
    rec["S_irrep_blocks"] = S_irr
    rec["blocks"] = rows
    fd, npairs, rnd = frob_direct(bs, tabs["gtab"], rmax)
    rec["frob"] = dict(blocks=frob_b, K_full=rec["frob_K_full"], mask_correlation=fd, pairs=npairs, fft_rounding=rnd)
    sb = np.sort(np.concatenate([np.repeat(np.sort(np.linalg.svd(blocks[nm], compute_uv=False)), cub.dim[nm]) for nm in IRREPS
                                 if min(blocks[nm].shape) > 0]))
    nz = abs(len(sb) - len(sK))
    sKs = np.sort(sK)
    # the blocks carry extra exact zeros (sum_l d_l |m_le - m_lo| >= |n_e - n_o|): compare the nonzero tails
    kk = min(len(sb), len(sKs))
    rec["svals_blocks_vs_full_maxdev_top"] = float(np.abs(sb[-kk:] - sKs[-kk:]).max())
    rec["svals_count"] = dict(blocks=int(len(sb)), full=int(len(sKs)), diff=int(nz),
                              full_extra_tail_max=float(sKs[:len(sKs) - kk].max()) if len(sKs) > kk else 0.0)
    # differences
    base = S4
    rec["diff_vs_dense4"] = {k: rec[k] - base for k in ("S_scalar_x4_numerical_transform", "S_scalar_x4", "S_sublattice_svd", "S_irrep_blocks")}
    if a.ref is not None:
        rec["review_ref"] = a.ref
        rec["diff_vs_review"] = {k: rec[k] - a.ref for k in ("S_dense4", "S_scalar_x4", "S_sublattice_svd", "S_irrep_blocks")}
    rec["t_total"] = time.time() - t0
    rec["utc_end"] = utc()
    with open(a.out_json, "w") as fh:
        json.dump(rec, fh, indent=1)
    md = ["## obs_ball_cut_v1 validate: R %g, L %d (site-centred, antiperiodic, m 0)" % (Reff, L), "",
          "sites %d (even %d, odd %d), max |x|^2 %d, mask sha1 %s" % (n, bs["n_e"], bs["n_o"], bs["max_x2"], bs["sha1"]), "",
          "| route | S | minus dense 4-component |" + (" minus review |" if a.ref is not None else ""),
          "|---|---|---|" + ("---|" if a.ref is not None else "")]
    for k in ("S_dense4", "S_scalar_x4_numerical_transform", "S_scalar_x4", "S_sublattice_svd", "S_irrep_blocks"):
        line = "| %s | %.10f | %.3e |" % (k, rec[k], rec[k] - base)
        if a.ref is not None:
            line += " %.3e |" % (rec[k] - a.ref)
        md.append(line)
    if a.ref is not None:
        md += ["", "review's value %.10f (comparison only)" % a.ref]
    md += ["", "checks: kernel forbidden-parity max %.2e%s; Omega transform: internal off-diagonal max %.2e, diagonal spread %.2e, "
           "numerical minus analytic C_+ %.2e; same-sublattice off-diagonal %.2e; V K - K V max %.2e; group: %s; orbit checks %s; "
           "Frobenius blocks %.12e, full K %.12e, mask correlation %.12e; top singular values blocks vs full max dev %.2e" % (
               tabs["forbidden"], (", direct-sum max dev %.2e" % rec["kernel_direct_sum_maxdev"]) if "kernel_direct_sum_maxdev" in rec else "",
               offd, diagd, rec["omega_transform"]["numerical_minus_analytic_Cplus_max"], rec["same_sublattice_offdiag_max"], cdev,
               json.dumps({k: float("%.2e" % v) for k, v in gchk.items()}), json.dumps(rec["orbit_checks"]), frob_b, rec["frob_K_full"], fd,
               rec["svals_blocks_vs_full_maxdev_top"]),
           "", "multiplicities (even, odd; character formula = Gram rank): " + ", ".join(
               "%s %d/%d" % (nm, bs["m"][nm][0], bs["m"][nm][1]) for nm in IRREPS),
           "", "times: dense eigvalsh %.1f s, irrep route %.2f s, total %.1f s" % (rec["t_dense4_eigvalsh"], rec["t_irrep"], rec["t_total"])]
    with open(a.md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    log("validate R %g L %d done: dense4 %.10f, scalar %.10f, sublattice %.10f, irrep %.10f" % (
        Reff, L, S4, rec["S_scalar_x4"], S_sub, S_irr))


def mode_run(a):
    cub = Cubic()
    gchk = cub.checks()
    log("group checks %s" % json.dumps(gchk))
    os.makedirs(a.out_dir, exist_ok=True)
    jl = open(os.path.join(a.out_dir, "ballcut_%s.jsonl" % a.tag), "a")
    for R in a.R:
        Rmaxeff = R + max(abs(o) for o in a.offsets16) / 16.0
        rmax = 2 * int(math.ceil(Rmaxeff)) + 2
        Ls = []
        for c in a.box_ratios:
            L = int(round(c * R))
            L += L % 2
            Ls.append((c, L))
        recs = []
        tabs = {}
        for c, L in Ls:
            t0 = time.time()
            tb = kernel_tables(L, rmax)
            tabs[L] = dict(sgt=signed_tables(tb["gtab"], tb["itab"]), gtab=tb["gtab"], forbidden=tb["forbidden"], t=time.time() - t0)
            del tb
            log("R %g: kernel table L %d (rmax %d) in %.1f s, forbidden-parity max %.2e" % (R, L, rmax, tabs[L]["t"], tabs[L]["forbidden"]))
        for j in a.offsets16:
            Reff = R + j / 16.0
            t0 = time.time()
            bs = ball_structure(Reff, cub)
            t_orb = time.time() - t0
            for c, L in Ls:
                t1 = time.time()
                blocks = assemble_blocks(bs, cub, tabs[L]["sgt"], rmax)
                t_asm = time.time() - t1
                rows, S, frob_b = block_entropies(blocks, cub, a.solver, a.device, check_gram=a.check_gram)
                del blocks
                t2 = time.time()
                fd, npairs, rnd = frob_direct(bs, tabs[L]["gtab"], rmax)
                t_frob = time.time() - t2
                rec = dict(script=SCRIPT, host=socket.gethostname(), utc=utc(), R=R, delta16=j, Reff=Reff, L=L, box_ratio=c,
                           n=bs["n"], n_e=bs["n_e"], n_o=bs["n_o"], max_x2=bs["max_x2"], mask_sha1=bs["sha1"], S=S,
                           S_over_4piR2=S / (4 * math.pi * Reff * Reff), blocks=rows,
                           frob=dict(blocks=frob_b, mask_correlation=fd, rel_diff=(frob_b - fd) / fd, fft_rounding=rnd),
                           checks=dict(orbit_sum=bs["orbit_sum_check"], mult_sum=bs["mult_sum_check"],
                                       gram_dev=[bs["oe"]["gram_dev"], bs["oo"]["gram_dev"]],
                                       orbit_dim=[bs["oe"]["orbit_dim_check"], bs["oo"]["orbit_dim_check"]],
                                       kernel_forbidden_parity=tabs[L]["forbidden"]),
                           solver=a.solver, device=a.device, threads=os.environ.get("OPENBLAS_NUM_THREADS"),
                           t_orbits=t_orb, t_assemble=t_asm, t_solve=sum(r["t_solve"] for r in rows.values()), t_frob=t_frob,
                           t_table=tabs[L]["t"])
                recs.append(rec)
                jl.write(json.dumps(rec) + "\n")
                jl.flush()
                big = max(rows.values(), key=lambda r: r["m_e"] * r["m_o"])
                log("R %g d %+d/16 L %d: n %d S %.10f S/4piR^2 %.6f frob rel %.1e | assemble %.1f s, solve %.1f s (largest %dx%d %.1f s)" % (
                    R, j, L, bs["n"], S, rec["S_over_4piR2"], rec["frob"]["rel_diff"], t_asm, rec["t_solve"], big["m_e"], big["m_o"],
                    big["t_solve"]))
        with open(os.path.join(a.out_dir, "ballcut_%s_R%g.json" % (a.tag, R)), "w") as fh:
            json.dump(dict(script=SCRIPT, R=R, offsets16=list(a.offsets16), boxes=[L for _, L in Ls], group_checks=gchk, records=recs), fh,
                      indent=1)
    jl.close()


def mode_counts(a):
    cub = Cubic()
    out = []
    for R in a.R:
        for j in a.offsets16:
            Reff = R + j / 16.0
            t0 = time.time()
            bs = ball_structure(Reff, cub)
            m = bs["m"]
            big = max(IRREPS, key=lambda nm: m[nm][0] * m[nm][1])
            work = sum(min(m[nm]) ** 2 * max(m[nm]) for nm in IRREPS)
            row = dict(R=R, delta16=j, Reff=Reff, n=bs["n"], n_e=bs["n_e"], n_o=bs["n_o"], mask_sha1=bs["sha1"], m=m,
                       n_orbits=[len(bs["oe"]["reps"]), len(bs["oo"]["reps"])], largest=[big] + m[big],
                       largest_GB=m[big][0] * m[big][1] * 8 / 1e9,
                       all_blocks_GB=sum(m[nm][0] * m[nm][1] for nm in IRREPS) * 8 / 1e9,
                       sum_m3=float(sum(max(m[nm]) ** 3 for nm in IRREPS)), work_proxy=int(work),
                       checks=dict(orbit_sum=bs["orbit_sum_check"], mult_sum=bs["mult_sum_check"],
                                   gram_dev=[bs["oe"]["gram_dev"], bs["oo"]["gram_dev"]]), t=time.time() - t0)
            out.append(row)
            log("counts R %g%+d/16: n %d, largest %s %dx%d (%.3f GB), all blocks %.2f GB" % (R, j, bs["n"], big, m[big][0], m[big][1],
                                                                                         row["largest_GB"], row["all_blocks_GB"]))
    with open(a.out_json, "w") as fh:
        json.dump(dict(script=SCRIPT, mode="counts", rows=out), fh, indent=1)


def mode_bench(a):
    rng = np.random.default_rng(12345)
    out = dict(script=SCRIPT, mode="bench", host=socket.gethostname(), threads=os.environ.get("OPENBLAS_NUM_THREADS"), rows=[])
    for m in a.sizes:
        A = rng.standard_normal((m, m)) / math.sqrt(4 * m)
        t0 = time.time()
        s1 = np.linalg.svd(A.copy(), compute_uv=False)
        t_svd = time.time() - t0
        t0 = time.time()
        Gm = A.T @ A
        mu = sla.eigvalsh(Gm, check_finite=False, overwrite_a=True)
        t_gram = time.time() - t0
        row = dict(m=m, t_svd_gesdd=t_svd, t_gram_eigvalsh=t_gram,
                   svals_gram_vs_svd_maxdev=float(np.abs(np.sort(np.sqrt(np.clip(mu, 0, None))) - np.sort(s1)).max()))
        out["rows"].append(row)
        log("bench m %d: gesdd %.2f s, gram+eigvalsh %.2f s" % (m, t_svd, t_gram))
    if a.complex_calib:
        n = 4096
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (A + A.conj().T) / 2
        t0 = time.time()
        sla.eigh(H, check_finite=False)
        out["complex128_eigh_n4096_vectors"] = time.time() - t0
        log("bench complex128 eigh (vectors) n 4096: %.2f s" % out["complex128_eigh_n4096_vectors"])
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--R", type=float, required=True)
    v.add_argument("--L", type=int, required=True)
    v.add_argument("--ref", type=float, default=None)
    v.add_argument("--out-json", required=True)
    v.add_argument("--md", required=True)
    r = sub.add_parser("run")
    r.add_argument("--R", type=float, nargs="+", required=True)
    r.add_argument("--offsets16", type=int, nargs="+", default=list(OFFSETS16))
    r.add_argument("--box-ratios", type=float, nargs="+", default=[4.0, 6.0, 8.0])
    r.add_argument("--solver", choices=["svd", "gram"], default="svd")
    r.add_argument("--device", default="cpu")
    r.add_argument("--check-gram", action="store_true")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--tag", default="v1")
    c = sub.add_parser("counts")
    c.add_argument("--R", type=float, nargs="+", required=True)
    c.add_argument("--offsets16", type=int, nargs="+", default=list(OFFSETS16))
    c.add_argument("--out-json", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000])
    b.add_argument("--complex-calib", action="store_true")
    b.add_argument("--out-json", required=True)
    a = ap.parse_args()
    log("%s %s on %s, threads %s" % (SCRIPT, a.mode, socket.gethostname(), os.environ.get("OPENBLAS_NUM_THREADS")))
    {"validate": mode_validate, "run": mode_run, "counts": mode_counts, "bench": mode_bench}[a.mode](a)


if __name__ == "__main__":
    main()
