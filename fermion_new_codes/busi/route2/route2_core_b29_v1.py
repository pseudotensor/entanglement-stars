#!/usr/bin/env python3
"""route2_core_b29_v1.py (busi lane, inbox #349; B29's core stage, REVIEWA_REVIEW_web40j_s1 B29, fix 31, FS3, Z16, Z17):
the second route of the dynamic core's Hartree-Fock level on the stored collapse-core slices.

Independent of route 1 (solve lane, core_hf_slices_v1*.py) and of the code route 1 imports (hf_star_variance_v3,
qh_branch_witness_v1b, engine_ops_v1, engine_fields_v2, slices144_io_v1): the chain, the Gaussian kernel, the radial rows,
the windows and the closed-form Hessian are route 2's own (route2_common.py, the ROUTE2 v1 conventions); the slice is read
by numpy's own npz loader; every channel-side contraction is new code here (torch, complex128, on one free box GPU).

Per slice (history prod or s8, clock T/r_s), at R_in, R_50, R_t (and R_out for the sea part's zero):
  (1) the state per channel: Phi = [sea | dust] Loewdin-orthonormalized, f the stored occupations; P_ref = R2 R2^+
      (Loewdin); the continued occupations p = mask (asserted equal to e < 0 of route 2's own eigh of h_m);
      ||(1 - P_ref) h_m P_ref||_F/||h_m||_F per channel, UNRESOLVED above 1e-6 (then no response is formed);
  (2) the mixed-state variance Tr[C A (1 - C) A] for C = Phi diag(f) Phi^+ (general formula, and its residual form on
      the pure state), the sea part Tr[P A (1 - P) A], the signed excess, the sea/dust/blocking split, the raw-column
      covariance as the complex64 floor control;
  (3) the rows from the record's eps_tot, J_tot (route 2's RK4 at dr 0.1, R_out = R[-1] + 8 ell + 1, N(R_out) = 1), the
      windows b(R) = dM(R)/ds, the Hessian B by enclosure products, the field check (b(R_out) against N_applied, v_applied);
  (4) the reference response P^A = L^* R_P L, (R_P X)_ou = (p_o - p_u)/(e_o - e_u) X_ou, applied exactly in the
      eigenbasis (the operator; GMRES uses only this), and as an explicit matrix from the SVD of the Cauchy kernel
      W_ou = 1/(e_o - e_u) truncated at a stated singular value (a rigorous bound on the dropped part is printed);
  (5) the dressed vertex beta = (I + B P^A)^-1 b by full GMRES from zero on the matrix-free exact operator (ROUTE2
      ruling 4's criterion: true residual below 1e-10 of ||b||), beside the dense solve with the explicit matrix;
  (6) the native signed exchange E_F = -(1/2) sum_k w_k <V, T_k>, T_k[mu, nu] = Tr(A_mu D_k A_nu D_k), D = C - P_ref,
      the Hartree pair energy E_pair = (1/2) s^T V s, both for V = B (bare) and V = (I + B P^A)^-1 B (dressed), the
      energy, mixed and current blocks, the Gram bound |E_F| <= (1/2) ||V||_2 sum_k w_k Tr_mu T_k^{|D|}, and the
      dressed pair energy also by GMRES on the exact operator; sigma_min(I + B P^A) with the certified lower bound.
Modes: --selftest (the known-answer gates; seconds to minutes, small fixtures), --slice HIST T (the core stage),
--kappas a channel subset (the route-1 fixture), --dr the rows' grid (the discretization budget).
Outputs under --out (default <internal>/route2_core): <hist>_T<TTTT.TT>[_tag].json and _arrays.npz. No plots.
Box only (nice 10, at most 8 CPU threads, one GPU named by --device).
"""
import os
import sys

_T = '8'
if '--threads' in sys.argv:
    _T = sys.argv[sys.argv.index('--threads') + 1]
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = str(min(int(_T), 8))
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route2_common as RC  # noqa: E402

VERSION = 'route2_core_b29_v1'
CT = torch.complex128
RT = torch.float64
FOURPI = 4.0 * math.pi
UNRESOLVED_TOL = 1e-6
GMRES_TOL = 1e-10
T0 = time.time()

HIST = {
    'prod': dict(tag='c19_rs64_A_R2_tau8_ell4_avg_r11',
                 records='<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json',
                 fields='<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz',
                 slice_dirs=['<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11']),
    's8': dict(tag='c27_rs64_A_R2_tau8_ell8_prep4_avg',
               records='<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json',
               fields='<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz',
               slice_dirs=['<internal>/s8_slices']),
}
WELL_DIR = '<internal>/well'


def log(*a):
    print('[%8.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def md5_8(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()[:8]


CODE_MD5 = md5_8(os.path.abspath(__file__))
COMMON_MD5 = md5_8(RC.__file__)


def eigvalsh_dev(X, dev):
    """the eigenvalues of a real symmetric matrix on the device (the box's numpy/scipy LAPACK is slow for eigh, inbox #346)."""
    return torch.linalg.eigvalsh(torch.as_tensor(X, dtype=RT, device=dev)).cpu().numpy()


# ============================================================================ banded operators on the device
class Band:
    """a banded (2 ns x 2 ns) operator as its diagonals: a[d][i] = op[i, i + d] (zero where i + d is out of range)."""

    def __init__(self, diags, n, dev):
        self.n, self.dev = n, dev
        self.d = {int(k): torch.as_tensor(v, dtype=CT, device=dev) for k, v in diags.items()}

    @staticmethod
    def from_scipy(A, dev):
        A = A.tocsr()
        n = A.shape[0]
        r, c = A.nonzero()
        offs = sorted(set((c - r).tolist()))
        diags = {}
        for d in offs:
            a = np.zeros(n, complex)
            dg = A.diagonal(d)
            if d >= 0:
                a[:n - d] = dg
            else:
                a[-d:] = dg
            diags[d] = a
        return Band(diags, n, dev)

    def left(self, X):
        """op @ X (X: n x m)."""
        n = self.n
        out = torch.zeros_like(X)
        for d, a in self.d.items():
            if d >= 0:
                out[:n - d] += a[:n - d, None] * X[d:]
            else:
                out[-d:] += a[-d:, None] * X[:n + d]
        return out

    def right(self, X):
        """X @ op (X: m x n): (X op)[:, j] = sum_d X[:, j - d] op[j - d, j] = sum_d X[:, j - d] a_d[j - d]."""
        n = self.n
        out = torch.zeros_like(X)
        for d, a in self.d.items():
            if d >= 0:
                out[:, d:] += X[:, :n - d] * a[None, :n - d]
            else:
                out[:, :n + d] += X[:, -d:] * a[None, -d:]
        return out

    def dense(self):
        return self.left(torch.eye(self.n, dtype=CT, device=self.dev))

    def fro2(self):
        return float(sum(torch.sum(torch.abs(a) ** 2).item() for a in self.d.values()))


def vertex_band(hb, qb, x, ns):
    """A_x = (1/2){D_e, h} + (1/2){D_p, Q} as a Band: a_d[i] = (1/2)(x_e[s(i)] + x_e[s(i + d)]) h_d[i] + the same with Q."""
    dev = hb.dev
    n = 2 * ns
    xe = torch.as_tensor(np.repeat(np.asarray(x[:ns], float), 2), dtype=CT, device=dev)
    xp = torch.as_tensor(np.repeat(np.asarray(x[ns:], float), 2), dtype=CT, device=dev)
    diags = {}
    for src, xx in ((hb, xe), (qb, xp)):
        for d, a in src.d.items():
            s = torch.zeros(n, dtype=CT, device=dev)
            if d >= 0:
                s[:n - d] = 0.5 * (xx[:n - d] + xx[d:])
            else:
                s[-d:] = 0.5 * (xx[-d:] + xx[:n + d])
            diags[d] = diags.get(d, torch.zeros(n, dtype=CT, device=dev)) + s * a
    B = Band({}, n, dev)
    B.d = diags
    return B


def blocksum_T(P, Q, ns):
    """S[n, n'] = sum_{i in n, j in n'} P_ij Q_ji (two components per site)."""
    return (P * Q.transpose(0, 1)).reshape(ns, 2, ns, 2).sum(dim=(1, 3))


def bilinear(hs, X, Y, ns):
    """T[mu, nu] = Tr(A_mu X A_nu Y) for the site vertices A_mu = (1/2){Pi_n, h_t} (t = 0 energy: h, t = 1 current: Q), X and Y
    Hermitian; T = (1/4)[S(h_t X, h_t' Y) + S(h_t X h_t', Y) + S(X, h_t' Y h_t) + S(X h_t', Y h_t)] (route 2's own expansion of the
    four products); complex (2 ns x 2 ns), energy block first."""
    d = 2 * ns
    T = torch.zeros((d, d), dtype=CT, device=X.device)
    hX = [h.left(X) for h in hs]          # h_t X
    hY = [h.left(Y) for h in hs]          # h_t Y
    for t1 in range(2):
        for t2 in range(2):
            ht2 = hs[t2]
            ht1 = hs[t1]
            S = blocksum_T(hX[t1], hY[t2], ns)                                  # S(h_t X, h_t' Y)
            S = S + blocksum_T(ht2.right(hX[t1]), Y, ns)                        # S(h_t X h_t', Y)
            S = S + blocksum_T(X, ht1.right(hY[t2]), ns)                        # S(X, h_t' Y h_t)
            S = S + blocksum_T(ht2.right(X), ht1.right(Y), ns)                  # S(X h_t', Y h_t)
            T[t1 * ns:(t1 + 1) * ns, t2 * ns:(t2 + 1) * ns] = 0.25 * S
            del S
    return T


def fro2(X):
    return float(torch.sum(torch.abs(X) ** 2).item())


def adj(X):
    return X.conj().transpose(0, 1)


def herm(X):
    return 0.5 * (X + adj(X))


def site_density(op, Phi, ns, f=None):
    """Re Tr(Pi_n op C) per site for C = Phi diag(f) Phi^+ (Phi may be non-orthonormal)."""
    X = op.left(Phi)
    v = torch.real(torch.conj(Phi) * X)
    if f is not None:
        v = v * f[None, :]
    return v.sum(dim=1).reshape(ns, 2).sum(dim=1).cpu().numpy()


def loewdin(X):
    S = adj(X) @ X
    S = herm(S)
    d0 = float(torch.max(torch.abs(S - torch.eye(S.shape[0], dtype=CT, device=X.device))).item())
    ev, V = torch.linalg.eigh(S)
    Sm = (V * (ev ** -0.5)[None, :].to(CT)) @ adj(V)
    Y = X @ Sm
    S2 = adj(Y) @ Y
    d1 = float(torch.max(torch.abs(S2 - torch.eye(S.shape[0], dtype=CT, device=X.device))).item())
    return Y, d0, d1


# ============================================================================ variances (route 2's forms on a covariance)
def var_cov(A, Phi, f=None):
    """the mixed-state variance Tr[C A (1 - C) A] for C = Phi diag(f) Phi^+: sum_k f_k ||A phi_k||^2 - sum_kl f_k f_l |M_kl|^2,
    M = Phi^+ A Phi (Phi need not be orthonormal; then C is the raw covariance)."""
    AP = A.left(Phi)
    M = adj(Phi) @ AP
    if f is None:
        return fro2(AP) - fro2(M)
    sf = torch.sqrt(f.to(RT)).to(CT)
    return float(torch.sum(f * torch.sum(torch.abs(AP) ** 2, dim=0)).item()) - fro2(sf[:, None] * M * sf[None, :])


def var_resid(A, Phi):
    """the residual form for a projector C = Phi Phi^+ (orthonormal Phi): ||A Phi - Phi (Phi^+ A Phi)||_F^2 >= 0."""
    AP = A.left(Phi)
    R = AP - Phi @ (adj(Phi) @ AP)
    return fro2(R)


def var_eig_block(A, Uo, Uu):
    """||U_u^+ A U_o||_F^2 (the eigenbasis sea part)."""
    return fro2(adj(Uu) @ A.left(Uo))


def cross_block(A, X, Y):
    """||X^+ A Y||_F^2."""
    return fro2(adj(X) @ A.left(Y))


# ============================================================================ the response
def response_apply(chans, x, ns):
    """y = P^A x, y_mu = sum_k w_k Tr(A_mu Pi_k(A_x)), Pi(A) = U_o M U_u^+ + h.c., M_ou = (U_o^+ A U_u)_ou (p_o - p_u)/(e_o - e_u)
    (exact, in the eigenbasis); y read off the band of Pi only: Tr(hcal_n Pi) = Re sum_{i in n} (h Pi)_ii."""
    y = np.zeros(2 * ns)
    for ch in chans:
        A = vertex_band(ch['hb'], ch['qb'], x, ns)
        Mm = (adj(ch['Uo']) @ A.left(ch['Uu'])) * ch['Wou']
        Z = ch['Uo'] @ Mm
        Uu = ch['Uu']
        n = 2 * ns
        for t, op in ((0, ch['hb']), (1, ch['qb'])):
            out = torch.zeros(n, dtype=CT, device=Z.device)
            for d, a in op.d.items():
                # (op Pi)_ii = sum_d op[i, i+d] Pi[i+d, i], Pi[j, i] = sum_l Z[j, l] conj(Uu[i, l]) + Uu[j, l] conj(Z[i, l])
                if d >= 0:
                    i = slice(0, n - d)
                    j = slice(d, n)
                else:
                    i = slice(-d, n)
                    j = slice(0, n + d)
                pji = torch.sum(Z[j] * torch.conj(Uu[i]), dim=1) + torch.sum(Uu[j] * torch.conj(Z[i]), dim=1)
                out[i] += a[i] * pji
            y[t * ns:(t + 1) * ns] += ch['w'] * torch.real(out).reshape(ns, 2).sum(dim=1).cpu().numpy()
    return y


def response_matrix_svd(ch, ns, rank_tol=1e-15, log_fn=log):
    """the explicit P^A of one channel (unweighted) from the SVD of the Cauchy kernel W_ou = 1/(e_o - e_u):
    W = sum_r s_r phi_r psi_r^T + E, ||E||_2 = s_(r+1); P^A[mu, nu] = sum_r s_r 2 Re Tr(A_mu X_r A_nu Y_r),
    X_r = U_o diag(phi_r) U_o^+, Y_r = U_u diag(psi_r) U_u^+. Returns (P, info) with the dropped singular value."""
    t0 = time.time()
    W = ch['Wou'].real
    Us, s, Vh = torch.linalg.svd(W, full_matrices=False)
    s_np = s.cpu().numpy()
    r = int(np.sum(s_np > rank_tol * s_np[0]))
    r = max(1, min(r, len(s_np) - 1))
    hs = (ch['hb'], ch['qb'])
    d = 2 * ns
    P = torch.zeros((d, d), dtype=RT, device=W.device)
    Uo, Uu = ch['Uo'], ch['Uu']
    for k in range(r):
        X = (Uo * Us[:, k].to(CT)[None, :]) @ adj(Uo)
        Y = (Uu * Vh[k, :].to(CT)[None, :]) @ adj(Uu)
        T = bilinear(hs, X, Y, ns)
        P += 2.0 * s[k] * torch.real(T)
        del X, Y, T
    asym = float(torch.linalg.norm(P - P.T).item() / max(torch.linalg.norm(P).item(), 1e-300))
    P = 0.5 * (P + P.T)
    info = dict(rank=r, s_max=float(s_np[0]), s_dropped=float(s_np[r]), s_dropped_rel=float(s_np[r] / s_np[0]), asym_raw=asym,
                seconds=time.time() - t0)
    log_fn('    P^A kappa %d by the SVD of the Cauchy kernel: rank %d, s_max %.3e, dropped %.2e (rel %.1e), asym %.1e, %.0f s'
           % (ch['kappa'], r, s_np[0], s_np[r], s_np[r] / s_np[0], asym, info['seconds']))
    return P, info


def gram_identity(ch, ns):
    """Gm[mu, nu] = Tr(A_mu A_nu) (the vertex map's Gram; lambda_max bounds ||A_x||_F^2 / ||x||^2)."""
    I = torch.eye(2 * ns, dtype=CT, device=ch['Uo'].device)
    return torch.real(bilinear((ch['hb'], ch['qb']), I, I, ns))


def gmres(apply_op, b, tol=1e-12, maxit=200, label='', logf=log):
    """full GMRES from zero (Arnoldi, modified Gram-Schmidt, one reorthogonalization), as route2_dressing_gmres.gmres."""
    nb = float(np.linalg.norm(b))
    n = len(b)
    V = np.zeros((maxit + 1, n))
    H = np.zeros((maxit + 1, maxit))
    V[0] = b / nb
    hist = []
    yk = None
    for j in range(maxit):
        w = apply_op(V[j])
        for _ in range(2):
            for i in range(j + 1):
                hij = float(V[i] @ w)
                H[i, j] += hij
                w = w - hij * V[i]
        H[j + 1, j] = float(np.linalg.norm(w))
        e1 = np.zeros(j + 2)
        e1[0] = nb
        yk, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], e1, rcond=None)
        est = float(np.linalg.norm(H[:j + 2, :j + 1] @ yk - e1) / nb)
        hist.append(est)
        if est < tol or H[j + 1, j] < 1e-300:
            return V[:j + 1].T @ yk, hist, j + 1
        V[j + 1] = w / H[j + 1, j]
    return V[:maxit].T @ yk, hist, maxit


def dress_solve(B, chans, ns, b, label):
    t0 = time.time()
    napp = [0]

    def op(x):
        napp[0] += 1
        return x + B @ response_apply(chans, x, ns)

    x, hist, it = gmres(op, b, label=label)
    res = float(np.linalg.norm(b - op(x)) / np.linalg.norm(b))
    log('  GMRES %s: %d iterations, true residual %.2e (%s 1e-10), %.0f s' % (label, it, res, 'below' if res < GMRES_TOL else 'ABOVE', time.time() - t0))
    return x, dict(iterations=it, applications=napp[0], residual_history=hist, residual_rel=res, residual_ok=bool(res < GMRES_TOL), seconds=time.time() - t0)


# ============================================================================ the slice and the record
def find_slice(hist, t):
    H = HIST[hist]
    name = 'orbitals_%s_T%07.2f.npz' % (H['tag'], float(t))
    for d in H['slice_dirs']:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(name)


def load_record(hist, t):
    H = HIST[hist]
    J = json.load(open(H['records']))
    A = J['args']
    rs = float(A['rs'])
    rec = None
    for r in J['records']:
        if r.get('E') is None:
            continue
        tr = float(r['T_over_rs']) if r.get('T_over_rs') is not None else float(r['T']) / rs
        if abs(tr - t) < 1e-6:
            rec = r
    if rec is None:
        raise RuntimeError('no record at %g' % t)
    Z = np.load(H['fields'], allow_pickle=True)
    Tz = np.asarray(Z['T'], float)
    i = int(np.argmin(np.abs(Tz / rs - t)))
    assert abs(Tz[i] / rs - t) < 1e-6
    out = dict(m=float(A['m']), g=float(A['g']), ell=float(A['ell']), rs=rs, half=bool(A.get('half_offset', 0)), index=i,
               eps_tot=np.asarray(Z['eps_tot'][i], float), J_tot=np.asarray(Z['J_tot'][i], float),
               N_applied=np.asarray(Z['N_applied'][i], float), v_applied=np.asarray(Z['v_applied'][i], float),
               E=float(rec['E']), trapping_zeros=list(rec.get('trapping_zeros') or []), R_c=rec.get('R_c'),
               E_dust_shells=rec.get('E_dust_shells'), E_pairs_shells=rec.get('E_pairs_shells'), converged=rec.get('converged'),
               gauge=(rec.get('gauge') or {}).get('label'), mask_equals_negative_set=rec.get('mask_equals_negative_set'),
               inversions_total=rec.get('inversions_total'), commutator_h_R2_F=rec.get('commutator_h_R2_F'))
    return out


def frac_radius_r2(R, cum, q):
    """route 2's: the radius where the cumulative fraction (inclusive of site n at R_n) reaches q, linear between site radii."""
    k = int(np.argmax(cum >= q))
    if k == 0:
        return float(R[0])
    return float(R[k - 1] + (q - cum[k - 1]) / (cum[k] - cum[k - 1]) * (R[k] - R[k - 1]))


# ============================================================================ one channel
def ref_status(hmb, R2, mask_ok):
    """||(1 - P_ref) h_m P_ref||_F/||h_m||_F and the status: UNRESOLVED above UNRESOLVED_TOL or when the stored occupations are not the
    instantaneous negative set of route 2's own eigh (then the response built here does not apply)."""
    HP = hmb.left(R2)
    res = HP - R2 @ (adj(R2) @ HP)
    c = math.sqrt(fro2(res)) / math.sqrt(hmb.fro2())
    return c, ('resolved' if (c <= UNRESOLVED_TOL and mask_ok) else 'UNRESOLVED')


def build_channel(Z, k, ns, m, N, v, dev, want_D_eig=True, keep_raw=False):
    t0 = time.time()
    h, Q, R = RC.chain_ops(ns, m, float(k), False)
    hb, qb = Band.from_scipy(h, dev), Band.from_scipy(Q, dev)
    hm = RC.clocked(h, Q, N, v)
    hmb = Band.from_scipy(hm, dev)
    Hm = hmb.dense()
    e, U = torch.linalg.eigh(herm(Hm))
    del Hm
    e_np = e.cpu().numpy()
    w = 16.0 * k
    ch = dict(kappa=int(k), w=w, hb=hb, qb=qb, hmb=hmb)
    es = np.asarray(Z['e_%d' % k], float).reshape(-1)
    ms = np.unpackbits(np.asarray(Z['mask_%d' % k], np.uint8))[:2 * ns].astype(bool)
    mask = e_np < 0
    ch['e_vs_stored_abs'] = float(np.max(np.abs(es - e_np)))
    ch['mask_equals_own_neg'] = bool(np.array_equal(ms, mask))
    ch['n_neg'] = int(mask.sum())
    mt = torch.as_tensor(mask, device=dev)
    ch['Uo'] = U[:, mt].contiguous()
    ch['Uu'] = U[:, ~mt].contiguous()
    eo, eu = e[mt], e[~mt]
    ch['Wou'] = (1.0 / (eo[:, None] - eu[None, :])).to(CT)
    ch['gap'] = float((eu.min() - eo.max()).item())
    ch['e'] = e_np
    del U
    # the state
    sea = torch.as_tensor(np.asarray(Z['sea_%d' % k]), device=dev).to(CT)
    dkey = 'dust_%d' % k
    dust = torch.as_tensor(np.asarray(Z[dkey]), device=dev).to(CT) if dkey in Z.files else torch.zeros((2 * ns, 0), dtype=CT, device=dev)
    fd = np.asarray(Z['f_%d' % k], float).reshape(-1)
    n_s, n_d = sea.shape[1], dust.shape[1]
    Phi_raw = torch.cat([sea, dust], dim=1)
    del sea, dust
    f = np.concatenate([np.ones(n_s), fd])
    ch['n_sea'], ch['n_dust'], ch['n_occ'] = int(n_s), int(n_d), int(n_s + n_d)
    ch['f_min'], ch['f_max'] = float(f.min()), float(f.max())
    ch['pure'] = bool(np.all((f < 1e-12) | (f > 1 - 1e-12)))
    Phi, d0, d1 = loewdin(Phi_raw)
    ch['stored_orthonormality'], ch['loewdin_orthonormality'] = d0, d1
    R2raw = torch.as_tensor(np.asarray(Z['R2_%d' % k]), device=dev).to(CT)
    R2, r0, r1 = loewdin(R2raw)
    ch['ref_stored_orthonormality'], ch['ref_loewdin_orthonormality'] = r0, r1
    ch['f'] = torch.as_tensor(f, dtype=RT, device=dev)
    ch['Phi'], ch['R2'] = Phi, R2
    if keep_raw:
        ch['Phi_raw'] = Phi_raw
    # sources (Loewdin state; raw columns with f)
    rhoC, jC = site_density(hb, Phi, ns, ch['f']), site_density(qb, Phi, ns, ch['f'])
    rhoR, jR = site_density(hb, R2, ns), site_density(qb, R2, ns)
    ch['s'] = np.concatenate([w * (rhoC - rhoR), w * (jC - jR)])
    rr, jr = site_density(hb, Phi_raw, ns, ch['f']), site_density(qb, Phi_raw, ns, ch['f'])
    ch['s_raw'] = np.concatenate([w * (rr - site_density(hb, R2raw, ns)), w * (jr - site_density(qb, R2raw, ns))])
    if n_d:
        Dc = Phi_raw[:, n_s:]
        fdt = ch['f'][n_s:]
        ch['dust_site'] = w * (torch.abs(Dc) ** 2 * fdt[None, :]).sum(dim=1).reshape(ns, 2).sum(dim=1).cpu().numpy()
    else:
        ch['dust_site'] = np.zeros(ns)
    del R2raw
    if not keep_raw:
        del Phi_raw
    # the reference: resolution against route 2's own eigenvectors, the commutator with h_m
    leak = fro2(adj(ch['Uu']) @ R2)
    ch['pref_resolution_defect'] = math.sqrt(2.0 * max(leak, 0.0)) / math.sqrt(R2.shape[1])
    c_, st_ = ref_status(hmb, R2, ch['mask_equals_own_neg'])
    ch['comm_h_Pref_rel'] = c_
    ch['comm_h_Pref_F'] = math.sqrt(2.0 * hmb.fro2()) * c_
    ch['resolved'] = bool(st_ == 'resolved')
    # vertex identity h_m = A_(N, v)
    Av = vertex_band(hb, qb, np.concatenate([N, v]), ns)
    ch['vertex_identity_rel'] = math.sqrt(sum(float(torch.sum(torch.abs(Av.d.get(d, 0) - a) ** 2).item()) for d, a in hmb.d.items())) / math.sqrt(hmb.fro2())
    # C, P, D; the exchange blocks T = Tr(A D A D); the pair content; |D| for the Gram bound
    C = herm((Phi * ch['f'].to(CT)[None, :]) @ adj(Phi))
    P = herm(R2 @ adj(R2))
    D = C - P
    ch['TrD'] = float(torch.real(torch.trace(D)).item())
    ch['TrD2'] = fro2(D)
    del C, P
    hs = (hb, qb)
    T = bilinear(hs, D, D, ns)
    ch['T_herm_defect'] = float(torch.linalg.norm(T - adj(T)).item() / max(torch.linalg.norm(T).item(), 1e-300))
    ch['T'] = torch.real(T).cpu().numpy()
    del T
    if want_D_eig:
        ev, VD = torch.linalg.eigh(D)
        ch['eigD'] = ev.cpu().numpy()
        absD = (VD * torch.abs(ev).to(CT)[None, :]) @ adj(VD)
        del VD
        Ta = bilinear(hs, absD, absD, ns)
        ch['trace_T_absD'] = float(torch.real(torch.trace(Ta)).item())
        del Ta, absD
    del D
    torch.cuda.empty_cache()
    ch['seconds'] = time.time() - t0
    return ch


def channel_row(ch):
    keys = ('kappa', 'w', 'n_sea', 'n_dust', 'n_occ', 'pure', 'f_min', 'f_max', 'stored_orthonormality', 'loewdin_orthonormality',
            'ref_stored_orthonormality', 'ref_loewdin_orthonormality', 'e_vs_stored_abs', 'mask_equals_own_neg', 'n_neg', 'gap',
            'pref_resolution_defect', 'comm_h_Pref_rel', 'comm_h_Pref_F', 'resolved', 'vertex_identity_rel', 'TrD', 'TrD2', 'T_herm_defect',
            'trace_T_absD', 'seconds')
    out = {k: ch.get(k) for k in keys}
    if 'eigD' in ch:
        ev = ch['eigD']
        out['n_eigD_above_1e-6'] = int(np.sum(np.abs(ev) > 1e-6))
        out['eigD_min'], out['eigD_max'] = float(ev.min()), float(ev.max())
    out['status'] = 'resolved' if ch['resolved'] else 'UNRESOLVED'
    return out


# ============================================================================ the variances of one vertex
def vertex_variances(chans, x, ns, raw=False, split=True, eig=True):
    VC = VCg = VP = VPe = Vraw = 0.0
    sea = dust = block = 0.0
    per = []
    for ch in chans:
        A = vertex_band(ch['hb'], ch['qb'], x, ns)
        vc_res = var_resid(A, ch['Phi']) if ch['pure'] else None
        vc_gen = var_cov(A, ch['Phi'], ch['f'])
        vc = vc_res if vc_res is not None else vc_gen
        vp = var_resid(A, ch['R2'])
        row = dict(kappa=ch['kappa'], w=ch['w'], VarC=vc, VarC_general_formula=vc_gen, VarP=vp)
        if eig:
            row['VarP_eig'] = var_eig_block(A, ch['Uo'], ch['Uu'])
            VPe += ch['w'] * row['VarP_eig']
        if split and ch['n_dust']:
            ns_ = ch['n_sea']
            Ps, Pd = ch['Phi'][:, :ns_], ch['Phi'][:, ns_:]
            row['Var_sea_cols'] = var_resid(A, Ps)
            row['Var_dust_cols'] = var_resid(A, Pd)
            row['blocking'] = -2.0 * cross_block(A, Pd, Ps)
        elif split:
            row['Var_sea_cols'] = vc
            row['Var_dust_cols'] = 0.0
            row['blocking'] = 0.0
        if raw and 'Phi_raw' in ch:
            row['VarC_raw_columns'] = var_cov(A, ch['Phi_raw'], ch['f'])
            Vraw += ch['w'] * row['VarC_raw_columns']
        VC += ch['w'] * vc
        VCg += ch['w'] * vc_gen
        VP += ch['w'] * vp
        if split:
            sea += ch['w'] * row['Var_sea_cols']
            dust += ch['w'] * row['Var_dust_cols']
            block += ch['w'] * row['blocking']
        per.append(row)
    out = dict(Var_C=VC, Var_C_general_formula=VCg, Var_C0=VP, signed_excess=VC - VP, per_channel=per)
    if eig:
        out['Var_C0_eig'] = VPe
    if split:
        out['split'] = dict(sea_columns=sea, dust_columns=dust, blocking=block, sum_minus_VarC=sea + dust + block - VC)
    if raw:
        out['Var_C_raw_columns'] = Vraw
        out['floor_footprint_rel'] = abs(Vraw - VC) / max(abs(VC), 1e-300)
    return out


# ============================================================================ the rows (route 2's own, route2_common's formulas on the device)
def gauss_kernels_t(r, Rs, ell):
    """route2_common.gauss_kernels on the device (same formulas, same series cut z < 1e-2)."""
    r = r[:, None]
    Rs = Rs[None, :]
    z = r * Rs / ell ** 2
    em = torch.exp(-(r - Rs) ** 2 / (2 * ell ** 2))
    ep = torch.exp(-(r + Rs) ** 2 / (2 * ell ** 2))
    ea = torch.exp(-(r ** 2 + Rs ** 2) / (2 * ell ** 2))
    small = z < 1e-2
    zs = torch.where(small, torch.ones_like(z), z)
    i0 = torch.where(small, ea * (1 + z ** 2 / 6), (em - ep) / (2 * zs))
    i1 = torch.where(small, ea * (z / 3) * (1 + z ** 2 / 10), ((zs - 1) * em + (zs + 1) * ep) / (2 * zs ** 2))
    pref = 1.0 / ((2 * math.pi) ** 1.5 * ell ** 3)
    return pref * i0, pref * i1


def cumsimpson_t(f, h):
    out = torch.zeros_like(f)
    pan = h / 3.0 * (f[0:-1:2] + 4.0 * f[1::2] + f[2::2])
    out[2::2] = torch.cumsum(pan, dim=0)
    out[1::2] = out[0:-1:2] + h / 12.0 * (5.0 * f[0:-1:2] + 8.0 * f[1::2] - f[2::2])
    return out


class GridT:
    """route2_common.Grid on the device: r = 0 .. R_out at dr (even intervals), K_0 normalized column by column under Simpson, K_1, the
    half-point tables with the same column normalization."""

    def __init__(self, Rs, ell, R_out, dr, dev):
        n = int(round(R_out / dr))
        n += n % 2
        self.n, self.dr, self.R_out, self.dev = n, R_out / n, R_out, dev
        self.r = np.linspace(0.0, R_out, n + 1)
        self.W = RC.simpson_weights(n, self.dr)
        self.rt = torch.as_tensor(self.r, dtype=RT, device=dev)
        self.Wt = torch.as_tensor(self.W, dtype=RT, device=dev)
        self.Rst = torch.as_tensor(np.asarray(Rs, float), dtype=RT, device=dev)
        self.ell = float(ell)
        K0, K1 = gauss_kernels_t(self.rt, self.Rst, self.ell)
        w2 = self.Wt * FOURPI * self.rt ** 2
        self.norm_raw = w2 @ K0
        self.K0 = K0 / self.norm_raw[None, :]
        self.K1 = K1
        self.norm_err = float(torch.max(torch.abs(w2 @ self.K0 - 1.0)).item())
        self.norm_raw_err = float(torch.max(torch.abs(self.norm_raw - 1.0)).item())
        self.mom1_err = float(torch.max(torch.abs((w2 * self.rt) @ self.K1 - self.Rst) / self.Rst).item())
        rh = self.rt[:-1] + 0.5 * self.dr
        a0, a1 = gauss_kernels_t(rh, self.Rst, self.ell)
        self.K0h = a0 / self.norm_raw[None, :]
        self.K1h = a1

    def sources(self, e, j):
        et = torch.as_tensor(e, dtype=RT, device=self.dev)
        jt = torch.as_tensor(j, dtype=RT, device=self.dev)
        return [x.cpu().numpy() for x in (self.K0 @ et, self.K1 @ jt, self.K0h @ et, self.K1h @ jt)]


def rows_t(grid, e, j, G):
    rho, J, rho_h, J_h = grid.sources(e, j)
    return RC.rows_from_profiles(grid.r, grid.dr, rho, J, rho_h, J_h, G)


def gradient_E_t(grid, sol):
    wr = grid.Wt * FOURPI * grid.rt ** 2
    Nt = torch.as_tensor(sol['N'], dtype=RT, device=grid.dev)
    vt = torch.as_tensor(sol['v'], dtype=RT, device=grid.dev)
    return (torch.cat([(wr * Nt) @ grid.K0, (wr * vt) @ grid.K1]) / Nt[-1]).cpu().numpy()


def windows_t(grid, sol):
    pref = FOURPI * grid.rt ** 2
    Nt = torch.as_tensor(sol['N'], dtype=RT, device=grid.dev)
    vt = torch.as_tensor(sol['v'], dtype=RT, device=grid.dev)
    a = cumsimpson_t((pref * Nt)[:, None] * grid.K0, grid.dr)
    c = cumsimpson_t((pref * vt)[:, None] * grid.K1, grid.dr)
    return a, c


def window_at_t(a, c, sol, i):
    return (torch.cat([a[i], c[i]]) / sol['N'][i]).cpu().numpy()


def hessian_closed_t(grid, sol, a, c, G):
    """route2_common.hessian_closed on the device (the same enclosure-product blocks)."""
    r, W = grid.rt, grid.Wt
    N = torch.as_tensor(sol['N'], dtype=RT, device=grid.dev)
    w = torch.as_tensor(sol['w'], dtype=RT, device=grid.dev)
    J = torch.as_tensor(sol['J'], dtype=RT, device=grid.dev)
    j3 = torch.zeros_like(r)
    z = torch.zeros_like(r)
    j3[1:] = FOURPI * J[1:] / (N[1:] * torch.abs(w[1:]) ** 3)
    z[1:] = 1.0 / (r[1:] * w[1:])
    dj = (j3 * W)[:, None]
    dz = (W * z * FOURPI * r ** 2)[:, None]
    Bee = G * G * (a.T @ (dj * a))
    Bep = G * (a.T @ (dz * grid.K1)) + G * G * (a.T @ (dj * c))
    T = c.T @ (dz * grid.K1)
    Bpp = G * (T + T.T) + G * G * (c.T @ (dj * c))
    return torch.cat([torch.cat([Bee, Bep], 1), torch.cat([Bep.T, Bpp], 1)], 0).cpu().numpy()


def rows_stage(R, ell, eps, jt, g, N_app, v_app, E_rec, dr, dev):
    t0 = time.time()
    ns = len(R)
    R_out = float(R[-1] + 8.0 * ell + 1.0)
    grid = GridT(R, ell, R_out, dr, dev)
    sol = rows_t(grid, eps, jt, g)
    a_, c_ = windows_t(grid, sol)
    Phi_res = gradient_E_t(grid, sol)
    dN = float(np.max(np.abs(Phi_res[:ns] - N_app)) / np.max(np.abs(N_app)))
    dv = float(np.max(np.abs(Phi_res[ns:] - v_app)) / np.max(np.abs(v_app)))
    t1 = time.time()
    B = hessian_closed_t(grid, sol, a_, c_, g)
    asym = float(np.linalg.norm(B - B.T) / np.linalg.norm(B))
    B = 0.5 * (B + B.T)
    info = dict(R_out=R_out, dr=grid.dr, n=grid.n, norm_raw_err=grid.norm_raw_err, norm_err=grid.norm_err, mom1_err=grid.mom1_err,
                field_check=dict(N_rel_maxdev=dN, v_rel_maxdev=dv, label='CORE' if max(dN, dv) < 1e-3 else 'SCHEME CONTROL',
                                 E_rows_R_out=float(sol['E']), E_record=E_rec, E_rel=abs(sol['E'] - E_rec) / abs(E_rec),
                                 M_at_R_last=float(sol['M'][int(round(R[-1] / grid.dr))]), N_centre=float(sol['N'][0])),
                hessian=dict(asym_raw=asym, norm_F=float(np.linalg.norm(B)), seconds=time.time() - t1),
                first_law_b_Rout_vs_gradient=float(np.linalg.norm(window_at_t(a_, c_, sol, grid.n) - Phi_res) / np.linalg.norm(Phi_res)),
                seconds=time.time() - t0)
    log('  rows (route 2): R_out %.1f dr %.4f n %d; field check dN %.2e dv %.2e (%s); E(R_out) %.10f record %.10f; B asym %.1e; %.0f s'
        % (R_out, grid.dr, grid.n, dN, dv, info['field_check']['label'], sol['E'], E_rec, asym, info['seconds']))
    return grid, sol, a_, c_, B, Phi_res, info


# ============================================================================ the core stage
def run_slice(a):
    dev = torch.device(a.device)
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t) + (('_' + a.tag) if a.tag else '')
    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, tag + '.json')
    if os.path.exists(pj) and not a.force:
        log('%s exists; skipped' % pj)
        return 0
    path = find_slice(hist, t)
    rec = load_record(hist, t)
    log('=== %s %s T/r_s %g: %s (%d bytes); m %g g %g ell %g; device %s (%s) ===' % (VERSION, hist, t, path, os.path.getsize(path), rec['m'], rec['g'], rec['ell'],
                                                                                    dev, torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'))
    Z = np.load(path, allow_pickle=False)
    R = np.asarray(Z['R'], float).reshape(-1)
    ns = len(R)
    N = np.asarray(Z['N_applied'], float).reshape(-1)
    v = np.asarray(Z['v_applied'], float).reshape(-1)
    kap = [int(k) for k in np.atleast_1d(Z['kappas'])]
    wts = [float(x) for x in np.atleast_1d(Z['weights'])]
    c64 = np.asarray(Z['sea_%d' % kap[0]]).dtype == np.complex64
    checks = []

    def chk(name, val, tol, info=False):
        ok = bool(val <= tol)
        checks.append(dict(name=name, value=float(val), tol=tol, status=('INFO' if info else ('PASS' if ok else 'FAIL'))))
        log('  [%s] %s: %.3e (tol %.0e)' % (checks[-1]['status'], name, val, tol))

    chk('R = 1..ns', float(np.max(np.abs(R - np.arange(1, ns + 1)))), 0.0)
    chk('weights = 16 kappa', float(max(abs(wv - 16.0 * k) for k, wv in zip(kap, wts))), 0.0)
    chk('slice applied fields = fields-npz applied row', float(max(np.max(np.abs(N - rec['N_applied'])), np.max(np.abs(v - rec['v_applied'])))), 0.0)
    kappas = [k for k in kap if (not a.kappas or k in a.kappas)]
    chans = []
    for k in kappas:
        ch = build_channel(Z, k, ns, rec['m'], N, v, dev, want_D_eig=not a.no_D_eig, keep_raw=c64)
        log('  kappa %d: n_occ %d (dust %d) %s; stored orth %.1e -> %.1e; e vs e_k %.1e; mask==own neg %s; gap %.4f; P_ref res %.1e; '
            '||(1-P)h_m P||/||h_m|| %.1e %s; Tr D %.6f; %.0f s' % (k, ch['n_occ'], ch['n_dust'], 'pure' if ch['pure'] else 'MIXED', ch['stored_orthonormality'],
                                                                 ch['loewdin_orthonormality'], ch['e_vs_stored_abs'], ch['mask_equals_own_neg'], ch['gap'],
                                                                 ch['pref_resolution_defect'], ch['comm_h_Pref_rel'], 'resolved' if ch['resolved'] else 'UNRESOLVED',
                                                                 ch['TrD'], ch['seconds']))
        chans.append(ch)
    s_tot = sum(ch['s'] for ch in chans)
    s_raw = sum(ch['s_raw'] for ch in chans)
    dust_site = sum(ch['dust_site'] for ch in chans)
    full = not a.kappas
    nrm = lambda x: float(np.max(np.abs(x)))  # noqa: E731
    if full:
        chk('sources (raw columns) vs the record eps_tot (max rel)', nrm(s_raw[:ns] - rec['eps_tot']) / nrm(rec['eps_tot']), 1e-4 if c64 else 1e-11)
        chk('sources (raw columns) vs the record J_tot (max rel)', nrm(s_raw[ns:] - rec['J_tot']) / nrm(rec['J_tot']), 1e-4 if c64 else 1e-11)
        chk('sources (Loewdin) vs eps_tot (max rel)', nrm(s_tot[:ns] - rec['eps_tot']) / nrm(rec['eps_tot']), 1e-4 if c64 else 1e-9)
        NB = float(sum(ch['w'] * ch['TrD'] for ch in chans))
        chk('N_B = sum 16 kappa Tr D vs 640', abs(NB - 640.0), 1e-5 if c64 else 1e-9)
    else:
        NB = float(sum(ch['w'] * ch['TrD'] for ch in chans))
    chk('Loewdin orthonormality (max)', max(ch['loewdin_orthonormality'] for ch in chans), 1e-13)
    chk('e vs stored e_k (max abs)', max(ch['e_vs_stored_abs'] for ch in chans), 1e-10)
    chk('stored mask == (e < 0) of route 2 own eigh (channels failing)', float(sum(not ch['mask_equals_own_neg'] for ch in chans)), 0.0)
    chk('unit occupations max |f - 1|', max(max(abs(ch['f_min'] - 1), abs(ch['f_max'] - 1)) for ch in chans), 1e-12)
    chk('vertex identity h_m = A_(N, v) (max rel)', max(ch['vertex_identity_rel'] for ch in chans), 1e-13)
    chk('||(1 - P_ref) h_m P_ref||_F/||h_m||_F (max; UNRESOLVED above %.0e)' % UNRESOLVED_TOL, max(ch['comm_h_Pref_rel'] for ch in chans), UNRESOLVED_TOL)
    unresolved = [ch['kappa'] for ch in chans if not ch['resolved']]
    # radii
    radii = {}
    tz = rec['trapping_zeros']
    if len(tz) >= 2:
        radii['R_in'] = dict(r=float(tz[0]), source="the record's dense inner root")
    dc = np.cumsum(dust_site)
    if dc[-1] > 0:
        radii['R_50'] = dict(r=frac_radius_r2(R, dc / dc[-1], 0.5), source='the dust columns, route 2 frac radius')
    if len(tz) >= 1:
        radii['R_t'] = dict(r=float(tz[-1]), source="the record's dense outer root")
    wj = os.path.join(WELL_DIR, 'well_%s_T%07.2f.json' % (hist, t))
    if os.path.exists(wj) and 'R_50' in radii and full:
        W_ = json.load(open(wj))
        chk('R_50 vs the well json (abs)', abs(radii['R_50']['r'] - float(W_['dust']['R_50'])), 1e-9)
    # the rows
    grid, sol, a_, c_, B, Phi_res, rows_info = rows_stage(R, rec['ell'], rec['eps_tot'], rec['J_tot'], rec['g'], N, v, rec['E'], a.dr, dev)
    radii['R_out'] = dict(r=float(grid.r[-1]), source='the rows outer radius')
    order = [l for l in ('R_in', 'R_50', 'R_t', 'R_out') if l in radii]
    bR, MH = {}, {}
    ref_eval = json.load(open(a.r_eval_from))['radii'] if a.r_eval_from else None
    for l in order:
        i = min(int(round(radii[l]['r'] / grid.dr)), grid.n)
        if ref_eval is not None and l in ref_eval:
            i = int(round(ref_eval[l]['r_eval'] / grid.dr))          # the discretization study: the same physical evaluation point
        radii[l].update(grid_row=i, r_eval=float(grid.r[i]), site=min(int(np.searchsorted(R, radii[l]['r'])), ns - 1),
                        M_H=float(sol['M'][i]), M_interp=float(np.interp(radii[l]['r'], grid.r, sol['M'])), N_H=float(sol['N'][i]),
                        N_enclosed=float(np.interp(radii[l]['r'], R, dc)))
        bR[l] = window_at_t(a_, c_, sol, i)
        MH[l] = radii[l]['M_H']
    Nx = Phi_res[:ns]
    eB = eigvalsh_dev(B, dev)
    grid_r = grid.r
    del a_, c_, grid.K0, grid.K1, grid.K0h, grid.K1h
    torch.cuda.empty_cache()
    # the response (exact operator; explicit matrix by the SVD of the Cauchy kernel)
    resp = dict(unresolved_channels=unresolved)
    PA = None
    if unresolved:
        resp['stopped'] = 'UNRESOLVED: ||[h_m, P_ref]|| not small on channels %s; no response formed' % unresolved
        log('  ' + resp['stopped'])
    elif a.pa_from:
        t1 = time.time()
        PA = np.load(a.pa_from)['PA']
        J0 = json.load(open(a.pa_from.replace('_arrays.npz', '.json')))
        resp.update({k: J0['response'][k] for k in ('per_channel', 'dropped_part_operator_norm_bound', 'PA_norm_2', 'PA_norm_F', 'dropped_bound_rel')})
        resp['loaded_from'] = a.pa_from
        PAn2 = resp['PA_norm_2']
        bound = resp['dropped_part_operator_norm_bound']
        xr = np.random.default_rng(11).standard_normal(2 * ns)
        ye = response_apply(chans, xr, ns)
        resp['matrix_vs_exact_operator_rel'] = dict(random1=float(np.linalg.norm(PA @ xr - ye) / np.linalg.norm(ye)))
        resp['seconds_matrix'] = time.time() - t1
        log('  P^A loaded from %s; matrix vs exact operator %.1e' % (a.pa_from, resp['matrix_vs_exact_operator_rel']['random1']))
    else:
        t1 = time.time()
        PA = np.zeros((2 * ns, 2 * ns))
        per = {}
        bound = 0.0
        for ch in chans:
            P_k, inf = response_matrix_svd(ch, ns, a.rank_tol)
            Gm = gram_identity(ch, ns)
            lam = float(torch.linalg.eigvalsh(Gm)[-1].item())
            inf['lambda_max_vertex_gram'] = lam
            inf['bound_contrib'] = 2.0 * ch['w'] * inf['s_dropped'] * lam
            bound += inf['bound_contrib']
            PA += ch['w'] * P_k.cpu().numpy()
            per[ch['kappa']] = inf
            del P_k, Gm
            torch.cuda.empty_cache()
        evP = eigvalsh_dev(PA, dev)
        PAn2 = float(np.max(np.abs(evP)))
        resp.update(per_channel=per, dropped_part_operator_norm_bound=bound, PA_norm_2=PAn2, PA_norm_F=float(np.linalg.norm(PA)),
                    dropped_bound_rel=bound / PAn2, seconds_matrix=time.time() - t1)
        rng = np.random.default_rng(11)
        phi = np.concatenate([N, v])
        tests = dict(random1=rng.standard_normal(2 * ns), random2=rng.standard_normal(2 * ns), b_R_50=bR.get('R_50', bR[order[0]]), phi_applied=phi)
        mvo = {}
        for nm, xx in tests.items():
            ye = response_apply(chans, xx, ns)
            ym = PA @ xx
            mvo[nm] = float(np.linalg.norm(ym - ye) / max(np.linalg.norm(ye), 1e-300)) if nm != 'phi_applied' else None
            if nm == 'phi_applied':
                resp['PA_phi_applied_over_phi_exact'] = float(np.linalg.norm(ye) / np.linalg.norm(phi))
                resp['PA_phi_applied_over_phi_matrix'] = float(np.linalg.norm(ym) / np.linalg.norm(phi))
                resp['PA_phi_applied_over_PAnorm_phi'] = float(np.linalg.norm(ye) / (PAn2 * np.linalg.norm(phi)))
        resp['matrix_vs_exact_operator_rel'] = {k: v_ for k, v_ in mvo.items() if v_ is not None}
        resp['PA_eig_min'], resp['PA_eig_max'] = float(evP[0]), float(evP[-1])
        log('  P^A: ||P^A||_2 %.4e; dropped-part bound %.2e (rel %.1e); matrix vs exact operator %s; P^A (N, v)/(N, v) exact %.1e; eig [%.3e, %.3e]; %.0f s'
            % (PAn2, bound, bound / PAn2, {k: '%.1e' % v_ for k, v_ in resp['matrix_vs_exact_operator_rel'].items()},
               resp['PA_phi_applied_over_phi_exact'], evP[0], evP[-1], resp['seconds_matrix']))
    # the dressed kernel (dense solve with the explicit matrix) and sigma_min with its certified bound
    kern = dict(B_eig_min=float(eB[0]), B_eig_max=float(eB[-1]), B_norm_2=float(max(abs(eB[0]), abs(eB[-1]))))
    V = None
    if PA is not None:
        Bt = torch.as_tensor(B, dtype=RT, device=dev)
        Pt = torch.as_tensor(PA, dtype=RT, device=dev)
        Jm = torch.eye(2 * ns, dtype=RT, device=dev) + Bt @ Pt
        Vt = torch.linalg.solve(Jm, Bt)
        kern['V_solve_residual_rel'] = float(torch.linalg.norm(Jm @ Vt - Bt).item() / torch.linalg.norm(Bt).item())
        kern['V_asym_raw'] = float(torch.linalg.norm(Vt - Vt.T).item() / torch.linalg.norm(Vt).item())
        Vt = 0.5 * (Vt + Vt.T)
        sv = torch.linalg.svdvals(Jm)
        smin = float(sv[-1].item())
        kern['sigma_min_I_plus_BPA_computed'] = smin
        kern['sigma_max_I_plus_BPA'] = float(sv[0].item())
        kern['sigma_min_certified_lower'] = smin - kern['B_norm_2'] * resp['dropped_part_operator_norm_bound'] - 1e-13 * float(sv[0].item())
        kern['norm_PA_B_2'] = float(torch.linalg.matrix_norm(Pt @ Bt, ord=2).item())
        V = Vt.cpu().numpy()
        Jm_keep = Jm
        eV = eigvalsh_dev(V, dev)
        kern['V_eig_min'], kern['V_eig_max'] = float(eV[0]), float(eV[-1])
        kern['V_norm_2'] = float(max(abs(eV[0]), abs(eV[-1])))
        kern['V_minus_B_rel_F'] = float(np.linalg.norm(V - B) / np.linalg.norm(B))
        del Bt, Pt, Vt
        torch.cuda.empty_cache()
        log('  kernel: eig B [%.4f, %.4f], eig V [%.4f, %.4f]; ||V - B||_F/||B||_F %.4f; sigma_min(I + B P^A) %.6f (certified >= %.6f); ||P^A B||_2 %.4f'
            % (eB[0], eB[-1], eV[0], eV[-1], kern['V_minus_B_rel_F'], smin, kern['sigma_min_certified_lower'], kern['norm_PA_B_2']))
    # the exchange and the pair energies, with blocks; the Gram bound
    exch = {}
    T_tot = sum(ch['w'] * ch['T'] for ch in chans)
    for name, K in (('bare', B), ('dressed', V)):
        if K is None:
            continue
        per = [-0.5 * ch['w'] * float(np.sum(K * ch['T'])) for ch in chans]
        EF = float(sum(per))
        Ep = 0.5 * float(s_tot @ (K @ s_tot))
        n_ = ns
        blk = lambda X, i, j: X[i * n_:(i + 1) * n_, j * n_:(j + 1) * n_]  # noqa: E731
        parts = (('energy', [(0, 0)]), ('mixed', [(0, 1), (1, 0)]), ('current', [(1, 1)]))
        bF = {nm: float(-0.5 * sum(float(np.sum(blk(K, i, j) * blk(T_tot, i, j))) for i, j in pr)) for nm, pr in parts}
        bP = {nm: float(0.5 * sum(float(s_tot[i * n_:(i + 1) * n_] @ (blk(K, i, j) @ s_tot[j * n_:(j + 1) * n_])) for i, j in pr)) for nm, pr in parts}
        Knorm = kern['B_norm_2'] if name == 'bare' else kern['V_norm_2']
        gram_b = 0.5 * Knorm * sum(ch['w'] * ch.get('trace_T_absD', float('nan')) for ch in chans)
        exch[name] = dict(E_F=EF, E_F_per_channel=per, E_pair=Ep, y=EF / Ep, E_F_over_abs_E_pair=EF / abs(Ep), sign_E_F='positive' if EF > 0 else 'negative',
                          sign_E_pair='positive' if Ep > 0 else 'negative', blocks=dict(E_F=bF, E_pair=bP, E_F_blocks_sum_minus=sum(bF.values()) - EF,
                                                                                     E_pair_blocks_sum_minus=sum(bP.values()) - Ep,
                                                                                     cancellation_E_F=abs(EF) / sum(abs(x) for x in bF.values()),
                                                                                     cancellation_E_pair=abs(Ep) / sum(abs(x) for x in bP.values())),
                          gram_bound=gram_b, gram_bound_over_abs_E_F=gram_b / abs(EF), kernel_norm_2=Knorm,
                          comparators={'1/N_B': 1.0 / 640.0, '1/N_50': 1.0 / radii['R_50']['N_enclosed'] if 'R_50' in radii else None,
                                       '1/N_t': 1.0 / radii['R_t']['N_enclosed'] if 'R_t' in radii else None})
        log('  exchange [%s]: E_F %.8e E_pair %.8e E_F/|E_pair| %.6e; blocks E_F %s (cancellation %.3f); E_pair %s; Gram bound %.3e (%.1f |E_F|)'
            % (name, EF, Ep, EF / abs(Ep), {k: '%.4e' % x for k, x in bF.items()}, exch[name]['blocks']['cancellation_E_F'],
               {k: '%.4e' % x for k, x in bP.items()}, gram_b, gram_b / abs(EF)))
    if V is not None:
        x_, gi = dress_solve(B, chans, ns, B @ s_tot, 'E_pair rhs B s')
        exch['dressed']['E_pair_gmres_exact_operator'] = 0.5 * float(s_tot @ x_)
        exch['dressed']['E_pair_gmres_vs_dense_rel'] = abs(exch['dressed']['E_pair_gmres_exact_operator'] - exch['dressed']['E_pair']) / abs(exch['dressed']['E_pair'])
        exch['dressed']['E_pair_gmres'] = gi
        exch['dressed']['dressed_over_bare_E_F_minus_1'] = exch['dressed']['E_F'] / exch['bare']['E_F'] - 1.0
        exch['sign_change_bare_to_dressed'] = dict(E_F=bool(np.sign(exch['dressed']['E_F']) != np.sign(exch['bare']['E_F'])),
                                                   E_pair=bool(np.sign(exch['dressed']['E_pair']) != np.sign(exch['bare']['E_pair'])))
    # the variances: noref, full (GMRES exact, and the dense solve), nocur (route 2's), lapse and sharp windows
    var = {}
    betas = {}
    Bnc = B.copy()
    Bnc[:, ns:] = 0.0
    for l in order:
        b = bR[l]
        ic = radii[l]['site']
        ent = dict(noref=b)
        dinfo = {}
        if PA is not None:
            beta, gi = dress_solve(B, chans, ns, b, '%s full' % l)
            ent['full'] = beta
            dinfo['full'] = gi
            bd = torch.linalg.solve(Jm_keep, torch.as_tensor(b, dtype=RT, device=dev)).cpu().numpy()
            ent['full_dense_matrix'] = bd
            dinfo['gmres_vs_dense_beta_over_b'] = float(np.linalg.norm(beta - bd) / np.linalg.norm(b))
            bnc = b.copy()
            bnc[ns:] = 0.0
            if not a.skip_nocur:
                bn, gn = dress_solve(Bnc, chans, ns, bnc, '%s nocur' % l)
                ent['nocur_dressed'] = bn
                dinfo['nocur_dressed'] = gn
        lapse = np.zeros(2 * ns)
        lapse[:ic + 1] = Nx[:ic + 1] / radii[l]['N_H']
        ent['lapse_window'] = lapse
        sharp = np.zeros(2 * ns)
        sharp[:ic + 1] = 1.0
        ent['sharp_window'] = sharp
        row = dict(radii[l], variants={}, dressing=dinfo)
        for nm, x in ent.items():
            vv = vertex_variances(chans, x, ns, raw=(c64 and nm in ('full', 'noref')), split=(nm in ('full', 'noref')), eig=(nm in ('full', 'noref')))
            Mref = MH[l]
            vv['sqrt_VarC_over_M'] = math.sqrt(max(vv['Var_C'], 0.0)) / Mref
            vv['sqrt_VarC_over_M_interp'] = math.sqrt(max(vv['Var_C'], 0.0)) / radii[l]['M_interp']
            vv['sqrt_VarC0_over_M'] = math.sqrt(max(vv['Var_C0'], 0.0)) / Mref
            vv['signed_sqrt_excess_over_M'] = math.copysign(math.sqrt(abs(vv['signed_excess'])), vv['signed_excess']) / Mref
            vv['N_r_inv_sqrt'] = radii[l]['N_enclosed'] ** -0.5 if radii[l]['N_enclosed'] > 0 else None
            vv['negative'] = bool(vv['Var_C'] < 0 or vv['Var_C0'] < 0)
            row['variants'][nm] = vv
            betas['beta_%s_%s' % (l, nm)] = x
        if 'full' in row['variants']:
            vf, v0 = row['variants']['full'], row['variants']['noref']
            row['reference_response_rel'] = vf['Var_C'] / v0['Var_C'] - 1.0
            row['gmres_vs_dense_VarC_rel'] = abs(vf['Var_C'] - row['variants']['full_dense_matrix']['Var_C']) / vf['Var_C']
        var[l] = row
        log('  %s r %.4f (row %.2f) M_H %.4f N_< %.2f: ' % (l, radii[l]['r'], radii[l]['r_eval'], MH[l], radii[l]['N_enclosed'])
            + '; '.join('%s Var_C %.6e Var_C0 %.6e exc %+.4e rms/M %.5f' % (nm, x['Var_C'], x['Var_C0'], x['signed_excess'], x['sqrt_VarC_over_M'])
                        for nm, x in row['variants'].items() if nm in ('noref', 'full', 'nocur_dressed')))
    # route 1's stored vectors on route 2's contractions (isolates the rows and the dressing from the contraction code)
    cross = None
    if a.route1_arrays:
        Z1 = np.load(a.route1_arrays)
        cross = dict(path=a.route1_arrays, md5=md5_8(a.route1_arrays), radii={})
        if 'r_grid' in Z1.files and len(Z1['r_grid']) == len(grid.r):
            cross['r_grid_max_abs_diff'] = float(np.max(np.abs(Z1['r_grid'] - grid.r)))
            cross['M_grid_max_rel_diff'] = float(np.max(np.abs(Z1['M_grid'] - sol['M'])) / np.max(np.abs(sol['M'])))
            cross['N_grid_max_abs_diff'] = float(np.max(np.abs(Z1['N_grid'] - sol['N'])))
        cross['s_total_rel_diff'] = float(np.linalg.norm(Z1['s_total'] - s_tot) / np.linalg.norm(s_tot))
        cross['Phi_resolved_rel_diff'] = float(np.linalg.norm(Z1['Phi_resolved'] - Phi_res) / np.linalg.norm(Phi_res))
        cross['B_eigs_max_abs_diff'] = float(np.max(np.abs(np.sort(Z1['B_eigs']) - eB)))
        if 'V_eigs' in Z1.files and V is not None:
            cross['V_eigs_max_abs_diff'] = float(np.max(np.abs(np.sort(Z1['V_eigs']) - eigvalsh_dev(V, dev))))
        for kk_, nm_ in (('B', 'bare'), ('V', 'dressed'), ('PA', None)):
            if kk_ in Z1.files:
                K1_ = Z1[kk_]
                mine_ = {'B': B, 'V': V, 'PA': PA}[kk_]
                if mine_ is not None:
                    cross['%s_rel_diff_F' % kk_] = float(np.linalg.norm(K1_ - mine_) / np.linalg.norm(K1_))
                if nm_ is not None:
                    cross['E_F_%s_route1_kernel_on_route2_T' % nm_] = float(-0.5 * np.sum(K1_ * T_tot))
                    cross['E_pair_%s_route1_kernel_on_route2_s' % nm_] = float(0.5 * s_tot @ (K1_ @ s_tot))
        cross['T_diag_rel_diff'] = {}
        for ch in chans:
            kk = 'T_diag_kappa%d' % ch['kappa']
            if kk in Z1.files:
                d2 = np.diag(ch['T'])
                cross['T_diag_rel_diff'][ch['kappa']] = float(np.linalg.norm(Z1[kk] - d2) / np.linalg.norm(d2))
        for l in order:
            ent = {}
            for nm in ('noref', 'full'):
                key = 'beta_%s_%s' % (l, nm)
                if key in Z1.files and nm in var[l]['variants']:
                    x1 = Z1[key]
                    vv = vertex_variances(chans, x1, ns, split=False, eig=False)
                    mine = betas['beta_%s_%s' % (l, nm)]
                    ent[nm] = dict(beta_rel_diff=float(np.linalg.norm(x1 - mine) / np.linalg.norm(mine)), Var_C_route1_beta=vv['Var_C'],
                                   Var_C0_route1_beta=vv['Var_C0'])
            cross['radii'][l] = ent
        log('  route 1 arrays on route 2: M_grid %.1e, s %.1e, Phi_res %.1e, eig B %.1e; betas %s' % (
            cross.get('M_grid_max_rel_diff', float('nan')), cross['s_total_rel_diff'], cross['Phi_resolved_rel_diff'], cross['B_eigs_max_abs_diff'],
            {l: {nm: '%.1e' % e_['beta_rel_diff'] for nm, e_ in v_.items()} for l, v_ in cross['radii'].items()}))
    # the angular regulator: the per-channel sea variance of the R_50 vertex (Z17's statement on the stored channels)
    reg = None
    if 'R_50' in var:
        nm = 'full' if 'full' in var['R_50']['variants'] else 'noref'
        pc = var['R_50']['variants'][nm]['per_channel']
        reg = dict(vertex='R_50 ' + nm, kappa=[p['kappa'] for p in pc], VarP_per_copy=[p['VarP'] for p in pc],
                   weighted_partial_sums=list(np.cumsum([p['w'] * p['VarP'] for p in pc])),
                   label='the channel regulator kappa <= %d (the production channel set); the per-copy sea variance of a nonconstant window tends to a '
                         'constant, so the weighted channel sum grows as K^2 without the regulator (Z17)' % max(kappas))
    out = dict(script=VERSION, code_md5=CODE_MD5, route2_common_md5=COMMON_MD5, history=hist, T_over_rs=t, slice=path, slice_size=os.path.getsize(path),
               complex64=bool(c64), device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None, kappas=kappas, channel_subset=bool(a.kappas),
               record=dict((k, rec[k]) for k in ('m', 'g', 'ell', 'rs', 'E', 'trapping_zeros', 'R_c', 'E_dust_shells', 'E_pairs_shells', 'converged', 'gauge',
                                                 'mask_equals_negative_set', 'inversions_total')),
               N_B=NB, channels=[channel_row(ch) for ch in chans], rows=rows_info, radii={l: radii[l] for l in order}, response=resp, kernel=kern,
               exchange=exch, variance=var, angular_regulator=reg, route1_cross=cross, checks=checks, n_fail=sum(c['status'] == 'FAIL' for c in checks),
               args=vars(a), seconds=time.time() - T0)
    RC.dump_json(pj, out)
    arr = dict(R=R, N_applied=N, v_applied=v, s_total=s_tot, s_raw=s_raw, dust_site=dust_site, r_grid=grid_r, M_grid=sol['M'], N_grid=sol['N'],
               Phi_resolved=Phi_res, B_eigs=eB, T_tot=(T_tot if a.save_kernels else T_tot.astype(np.float32)), kappas=np.array(kappas))
    for ch in chans:
        arr['e_kappa%d' % ch['kappa']] = ch['e']
        arr['T_diag_kappa%d' % ch['kappa']] = np.diag(ch['T']).copy()
        if 'eigD' in ch:
            arr['eigD_kappa%d' % ch['kappa']] = ch['eigD']
    for l in order:
        arr['b_%s' % l] = bR[l]
    arr.update(betas)
    if V is not None:
        arr['V_eigs'] = eigvalsh_dev(V, dev)
    if a.save_kernels:
        arr['B'] = B
        if PA is not None:
            arr['PA'] = PA
            arr['V'] = V
    np.savez(os.path.join(a.out, tag + '_arrays.npz'), **arr)
    log('saved %s (%d FAIL); %.0f s' % (pj, out['n_fail'], out['seconds']))
    return 0


# ============================================================================ known-answer gates
def selftest(a):
    from scipy.linalg import eigh as seigh
    dev = torch.device(a.device)
    res = []

    def rep(name, val, tol, note=''):
        ok = bool(val <= tol)
        res.append(dict(name=name, value=float(val), tol=tol, status='PASS' if ok else 'FAIL', note=note))
        log('  [%s] %s: %.3e (tol %.0e) %s' % ('PASS' if ok else 'FAIL', name, val, tol, note))

    # ---- Z17: the two-shell sea variance V_kappa = kappa^2/(72 kappa^2 + 32) (m = 0, N = 1, v = 0, A = {chi_1, h}/2)
    # (gate v1 put the unreduced covariance form at 1e-13 too and failed at kappa 1000 with 7.7e-9: a test defect, that form's roundoff is
    # eps ||A Phi||^2/Var, which grows as kappa^2; v2 holds it to 100 eps ||A Phi||^2/Var, the block and residual forms to 1e-13)
    worst = worst_cov = 0.0
    vals = {}
    for kap in (1, 2, 3, 10, 100, 1000):
        h, Q, R = RC.chain_ops(2, 0.0, float(kap))
        hb, qb = Band.from_scipy(h, dev), Band.from_scipy(Q, dev)
        e, U = seigh(h.toarray())
        Ut = torch.as_tensor(U, dtype=CT, device=dev)
        Uo, Uu = Ut[:, e < 0], Ut[:, e > 0]
        A = vertex_band(hb, qb, np.array([1.0, 0.0, 0.0, 0.0]), 2)
        V1 = var_eig_block(A, Uo, Uu)
        V2 = var_resid(A, Uo)
        V3 = var_cov(A, Uo, torch.ones(Uo.shape[1], dtype=RT, device=dev))
        ex = kap ** 2 / (72.0 * kap ** 2 + 32.0)
        vals[kap] = V1
        worst = max(worst, abs(V1 - ex) / ex, abs(V2 - ex) / ex)
        scale = fro2(A.left(Uo)) / ex
        worst_cov = max(worst_cov, abs(V3 - ex) / ex / (100 * 2.2e-16 * scale))
    rep('Z17 two-shell sea variance = kappa^2/(72 kappa^2 + 32) at kappa 1, 2, 3, 10, 100, 1000 (block and residual forms)', worst, 1e-13,
        'kappa=1: %.12f' % vals[1])
    rep('Z17 the same, unreduced covariance form, in units of 100 eps ||A Phi||^2/Var', worst_cov, 1.0)
    # m = 0.2 at kappa = 1 (the reader's 0.008636) and the K^2 growth of the weighted channel sum
    h, Q, R = RC.chain_ops(2, 0.2, 1.0)
    e, U = seigh(h.toarray())
    Ut = torch.as_tensor(U, dtype=CT, device=dev)
    A = vertex_band(Band.from_scipy(h, dev), Band.from_scipy(Q, dev), np.array([1.0, 0, 0, 0]), 2)
    v02 = var_eig_block(A, Ut[:, e < 0], Ut[:, e > 0])
    rep('Z17 two-shell at m a = 0.2, kappa = 1: 0.008636 (reader, 4 digits)', abs(v02 - 0.008636) / 0.008636, 6e-5, '%.7f' % v02)
    sums = {}
    acc = 0.0
    for kap in range(1, 101):
        h, Q, R = RC.chain_ops(2, 0.0, float(kap))
        e, U = seigh(h.toarray())
        Ut = torch.as_tensor(U, dtype=CT, device=dev)
        A = vertex_band(Band.from_scipy(h, dev), Band.from_scipy(Q, dev), np.array([1.0, 0, 0, 0]), 2)
        acc += 16 * kap * var_eig_block(A, Ut[:, e < 0], Ut[:, e > 0])
        if kap in (10, 30, 100):
            sums[kap] = acc
    rep('Z17 weighted channel sum 16 kappa V_kappa to K = 10, 30, 100 against 11.97, 102.98, 1121.7 (reader; K^2/9 growth)',
        max(abs(sums[10] - 11.97) / 11.97, abs(sums[30] - 102.98) / 102.98, abs(sums[100] - 1121.7) / 1121.7), 1e-3,
        '%.4f %.4f %.4f; over K^2/9: %.4f %.4f %.4f' % (sums[10], sums[30], sums[100], sums[10] / (100 / 9), sums[30] / (900 / 9), sums[100] / (10000 / 9)))
    # ---- Z16: counts fix neither: two 4-shell preparations with equal counts and currents, variances 1/2 against 0 (one mode per shell)
    s2 = 1 / math.sqrt(2)
    Xp = np.array([[s2, 0], [s2, 0], [0, s2], [0, s2]], complex)        # (|1> + |2>), (|3> + |4>)
    Yp = np.array([[s2, 0], [0, s2], [s2, 0], [0, s2]], complex)        # (|1> + |3>), (|2> + |4>)
    Aw = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)                  # the window count over shells 1, 2
    cnt = lambda P: np.real(np.diag(P @ P.conj().T))  # noqa: E731
    vX = float(np.real(np.trace((Xp @ Xp.conj().T) @ Aw @ (np.eye(4) - Xp @ Xp.conj().T) @ Aw)))
    vY = float(np.real(np.trace((Yp @ Yp.conj().T) @ Aw @ (np.eye(4) - Yp @ Yp.conj().T) @ Aw)))
    # the Fock-space brute force (16 states) of Var(N_window) in the two determinants
    fock = fock_tools(4)
    vXf = fock['slater_var'](Xp, Aw)
    vYf = fock['slater_var'](Yp, Aw)
    rep('Z16 equal counts per shell (max diff)', float(np.max(np.abs(cnt(Xp) - cnt(Yp)))), 1e-15)
    rep('Z16 variance of the window count: 0 (prep X) and 1/2 (prep Y), covariance formula and Fock brute force', max(abs(vX - 0), abs(vY - 0.5), abs(vXf), abs(vYf - 0.5)), 1e-14,
        'X %.3e Y %.6f; Fock X %.3e Y %.6f' % (vX, vY, vXf, vYf))
    # ---- the mixed-state variance, the witness identity and the mixed-state exchange against the Fock space (6 modes, thermal Gaussian)
    rng = np.random.default_rng(5)
    nm_ = 6
    F6 = fock_tools(nm_)
    Hh = rng.standard_normal((nm_, nm_)) + 1j * rng.standard_normal((nm_, nm_))
    Hh = Hh + Hh.conj().T
    eh, Uh = np.linalg.eigh(Hh)
    fth = 1.0 / (1.0 + np.exp(0.7 * eh))
    Cm = (Uh * fth) @ Uh.conj().T
    A1 = rng.standard_normal((nm_, nm_)) + 1j * rng.standard_normal((nm_, nm_))
    A1 = A1 + A1.conj().T
    A2 = rng.standard_normal((nm_, nm_)) + 1j * rng.standard_normal((nm_, nm_))
    A2 = A2 + A2.conj().T
    rho = F6['thermal'](Hh, 0.7)
    vF = F6['var'](rho, A1)
    Ct = torch.as_tensor(Uh, dtype=CT, device=dev)
    vC = var_cov(DenseBand(A1, dev), Ct, torch.as_tensor(fth, dtype=RT, device=dev))
    vT = float(np.real(np.trace(Cm @ A1 @ (np.eye(nm_) - Cm) @ A1)))
    rep('mixed-state variance Tr[C A (1 - C) A] = Fock variance of a thermal Gaussian (6 modes)', max(abs(vC - vF), abs(vT - vF)) / abs(vF), 1e-12, '%.10f' % vF)
    Pp = np.linalg.qr(rng.standard_normal((nm_, 3)) + 1j * rng.standard_normal((nm_, 3)))[0]
    Pm = Pp @ Pp.conj().T
    Dm = Cm - Pm
    wP = F6['ordered_pair'](rho, A1, A2, Pm)
    wF = float(np.real(np.trace(Dm @ A1) * np.trace(Dm @ A2) - np.trace(Dm @ A1 @ Dm @ A2)))
    rep('mixed-state reference-ordered pair <W_P(A, B)> = Tr(DA)Tr(DB) - Tr(DADB) (6 modes, Fock)', abs(wP - wF) / abs(wF), 1e-11, '%.10f' % wF)
    Pmix = 0.999 * Pm + 0.0005 * np.eye(nm_)
    w0 = F6['ordered_pair'](F6['gauss_state'](Pmix), A1, A2, Pmix)
    rep('the ordering: <W_P>_P = 0 (P near a projector, its own quasi-free state)', abs(w0) / abs(wF), 1e-10, '%.2e' % w0)
    # ---- the device port of route2_common's rows against route2_common itself (ns 64, ell 4, dr 0.1, a smooth positive source)
    Rr = np.arange(1, 65, dtype=float)
    e_s = 0.02 * np.exp(-((Rr - 20) / 8) ** 2) * Rr ** 2 / 50 + 1e-4
    j_s = 0.004 * np.sin(Rr / 9) * np.exp(-((Rr - 20) / 10) ** 2)
    gc = RC.Grid(Rr, 4.0, 64 + 33.0, 0.1, kind='gauss')
    gc.half_tables()
    sc = RC.rows_rk4(gc, e_s, j_s, 0.25)
    ac, cc = RC.windows(gc, sc)
    Bc = RC.hessian_closed(gc, sc, ac, cc, 0.25)
    gt = GridT(Rr, 4.0, 64 + 33.0, 0.1, dev)
    st = rows_t(gt, e_s, j_s, 0.25)
    at, ct = windows_t(gt, st)
    Bt_ = hessian_closed_t(gt, st, at, ct, 0.25)
    dmax = max(float(np.max(np.abs(st['M'] - sc['M'])) / np.max(np.abs(sc['M']))), float(np.linalg.norm(Bt_ - Bc) / np.linalg.norm(Bc)),
               float(np.linalg.norm(window_at_t(at, ct, st, 300) - RC.window_at(ac, cc, sc, 300)) / np.linalg.norm(RC.window_at(ac, cc, sc, 300))),
               float(np.linalg.norm(gradient_E_t(gt, st) - RC.gradient_E(gc, sc)) / np.linalg.norm(RC.gradient_E(gc, sc))))
    rep('the device port of route2_common (kernel tables, RK4 rows, windows, closed-form Hessian, gradient) vs route2_common (ns 64)', dmax, 1e-12)
    # ---- the chain fixture: ns = 12, kappa 1, 2, a non-stationary state with dust and a rotated reference
    ns = 12
    m = 0.2
    Rr = np.arange(1, ns + 1, dtype=float)
    Nf = 0.5 + 0.5 * (1 - np.exp(-Rr / 4))
    vf = -0.6 * np.exp(-((Rr - 6) / 4) ** 2)
    chans = []
    for kap, nd in ((1, 2), (2, 1)):
        h, Q, _ = RC.chain_ops(ns, m, float(kap))
        hm = RC.clocked(h, Q, Nf, vf)
        e, U = seigh(hm.toarray())
        o = e < 0
        Uu = U[:, ~o]
        # a non-eigen state: the occupied block and nd + 3 empty modes rotated together (pair content), the first n_o + nd columns kept
        X = np.concatenate([U[:, o], Uu[:, :nd + 3]], axis=1)
        Wr = unitary(X.shape[1], rng)
        Y = X @ Wr
        Phi = Y[:, :int(o.sum()) + nd]
        R2 = U[:, o] @ unitary(int(o.sum()), rng)
        chans.append(make_fixture_channel(kap, h, Q, hm, Phi, np.ones(Phi.shape[1]), R2, ns, dev))
    # bilinear against the dense trace
    ch = chans[0]
    Xd = herm(torch.as_tensor(rng.standard_normal((2 * ns, 2 * ns)) + 1j * rng.standard_normal((2 * ns, 2 * ns)), dtype=CT, device=dev))
    Yd = herm(torch.as_tensor(rng.standard_normal((2 * ns, 2 * ns)) + 1j * rng.standard_normal((2 * ns, 2 * ns)), dtype=CT, device=dev))
    Tb = bilinear((ch['hb'], ch['qb']), Xd, Yd, ns).cpu().numpy()
    Ad = site_vertices_dense(ch, ns)
    Xn, Yn = Xd.cpu().numpy(), Yd.cpu().numpy()
    Tdn = np.array([[np.trace(Ad[i] @ Xn @ Ad[j] @ Yn) for j in range(2 * ns)] for i in range(2 * ns)])
    rep('bilinear Tr(A_mu X A_nu Y) against the dense trace (ns 12)', float(np.max(np.abs(Tb - Tdn)) / np.max(np.abs(Tdn))), 1e-13)
    # P^A: the SVD route and the exact operator against the eigenbasis double sum
    Pex = np.zeros((2 * ns, 2 * ns))
    for chh in chans:
        Uo, Uu = chh['Uo'].cpu().numpy(), chh['Uu'].cpu().numpy()
        Wn = chh['Wou'].cpu().numpy()
        Fs = [Uo.conj().T @ Ad_ @ Uu for Ad_ in site_vertices_dense(chh, ns)]
        for i in range(2 * ns):
            for j in range(2 * ns):
                Pex[i, j] += chh['w'] * 2 * np.real(np.sum(Wn * Fs[j] * np.conj(Fs[i])))
    Psvd = sum(chh['w'] * response_matrix_svd(chh, ns, 1e-15, log_fn=lambda *x: None)[0].cpu().numpy() for chh in chans)
    rep('P^A by the SVD of the Cauchy kernel vs the eigenbasis double sum', float(np.max(np.abs(Psvd - Pex)) / np.max(np.abs(Pex))), 1e-12)
    xr = rng.standard_normal(2 * ns)
    rep('P^A exact operator vs the eigenbasis double sum (random vector)', float(np.linalg.norm(response_apply(chans, xr, ns) - Pex @ xr) / np.linalg.norm(Pex @ xr)), 1e-12)
    rep('P^A on the generator (N, v) of the fixture (over ||P^A|| ||(N, v)||)', float(np.linalg.norm(response_apply(chans, np.concatenate([Nf, vf]), ns))
                                                                                   / (np.linalg.norm(Pex, 2) * np.linalg.norm(np.concatenate([Nf, vf])))), 1e-13)
    # P^A negative semidefinite (the reference's response)
    rep('P^A negative semidefinite (largest eigenvalue over ||P^A||)', max(float(np.linalg.eigvalsh(Pex)[-1]), 0.0) / np.linalg.norm(Pex, 2), 1e-13)
    # variance of a vertex: covariance, residual and dense forms; one-particle exchange identity
    xv = rng.standard_normal(2 * ns)
    Ax = sum(xv[i] * Ad[i] for i in range(2 * ns))
    Pn = ch['Phi'].cpu().numpy()
    Cn = Pn @ Pn.conj().T
    vdense = float(np.real(np.trace(Cn @ Ax @ (np.eye(2 * ns) - Cn) @ Ax)))
    A_b = vertex_band(ch['hb'], ch['qb'], xv, ns)
    rep('variance: residual form and covariance formula vs dense Tr[C A (1 - C) A]', max(abs(var_resid(A_b, ch['Phi']) - vdense), abs(var_cov(A_b, ch['Phi'], ch['f']) - vdense)) / vdense, 1e-12)
    dvec = np.linalg.qr(rng.standard_normal((2 * ns, 1)) + 1j * rng.standard_normal((2 * ns, 1)))[0]
    Dd = torch.as_tensor(dvec @ dvec.conj().T, dtype=CT, device=dev)
    T1 = torch.real(bilinear((ch['hb'], ch['qb']), Dd, Dd, ns)).cpu().numpy()
    sd = np.array([np.real(dvec.conj().T @ Ad[i] @ dvec)[0, 0] for i in range(2 * ns)])
    Vr = rng.standard_normal((2 * ns, 2 * ns))
    Vr = Vr + Vr.T
    EF1 = -0.5 * float(np.sum(Vr * T1))
    Ep1 = 0.5 * float(sd @ Vr @ sd)
    rep('one-particle identity: E_F = -E_pair for D = |d><d| (y = -1)', abs(EF1 + Ep1) / abs(Ep1), 1e-13)
    # GMRES dressing against the dense solve, with a Hessian-like symmetric B
    Bf = 0.3 * Vr / np.linalg.norm(Vr, 2)
    bvec = rng.standard_normal(2 * ns)

    def op(x):
        return x + Bf @ response_apply(chans, x, ns)
    xg, hist_, it = gmres(op, bvec)
    xd = np.linalg.solve(np.eye(2 * ns) + Bf @ Pex, bvec)
    rep('GMRES on the exact operator vs the dense solve (fixture)', float(np.linalg.norm(xg - xd) / np.linalg.norm(xd)), 1e-11, '%d iterations' % it)
    # ---- fix 31: pair and exchange energies with and without Pi, the sign change recorded; UNRESOLVED on a reassigned block
    s_fx = sum(np.concatenate([chh['w'] * (site_density(chh['hb'], chh['Phi'], ns) - site_density(chh['hb'], chh['R2'], ns)),
                               chh['w'] * (site_density(chh['qb'], chh['Phi'], ns) - site_density(chh['qb'], chh['R2'], ns))]) for chh in chans)
    Tfx = sum(chh['w'] * torch.real(bilinear((chh['hb'], chh['qb']), chh['D'], chh['D'], ns)).cpu().numpy() for chh in chans)
    Bk = 2.0 * Bf
    Vd = np.linalg.solve(np.eye(2 * ns) + Bk @ Pex, Bk)
    Vd = 0.5 * (Vd + Vd.T)
    fx = {}
    for nm, K in (('without Pi (bare)', Bk), ('with Pi (dressed)', Vd)):
        fx[nm] = dict(E_pair=0.5 * float(s_fx @ K @ s_fx), E_F=-0.5 * float(np.sum(K * Tfx)))
    sc = {k: bool(np.sign(fx['without Pi (bare)'][k]) != np.sign(fx['with Pi (dressed)'][k])) for k in ('E_pair', 'E_F')}
    log('  fix 31 on the fixture: %s; sign change %s' % (fx, sc))
    res.append(dict(name='fix 31: pair and exchange energies with and without Pi (fixture), signs recorded', value=0.0, tol=0.0, status='PASS', note=json.dumps(dict(energies=fx, sign_change=sc))))
    # the reassigned block: (a) the top occupied and the lowest empty eigenvector rotated into each other by 0.3 rad inside P_ref
    # (a reference that is not a spectral projector of h_m); (b) the two exchanged outright (a spectral projector whose occupations
    # are not the instantaneous negative set: the response's (p_o - p_u)/(e_o - e_u) split is then not the one built here)
    chh = chans[0]
    e, U = torch.linalg.eigh(herm(chh['hmb'].dense()))
    o = (e < 0).cpu().numpy()
    io = np.nonzero(o)[0]
    iu = np.nonzero(~o)[0]
    Ua = U.clone()
    c_, s_ = math.cos(0.3), math.sin(0.3)
    Ua[:, int(io[-1])] = c_ * U[:, int(io[-1])] + s_ * U[:, int(iu[0])]
    Ra = Ua[:, torch.as_tensor(io, device=dev)]
    ca, sa = ref_status(chh['hmb'], Ra, True)
    idx = io.tolist()
    idx[-1] = int(iu[0])
    Rb = U[:, torch.as_tensor(idx, device=dev)]
    mask_b = np.zeros(len(o), bool)
    mask_b[idx] = True
    cb, sb = ref_status(chh['hmb'], Rb, bool(np.array_equal(mask_b, o)))
    cg, sg = ref_status(chh['hmb'], chh['R2'], True)
    rep('fix 31: a reassigned block returns UNRESOLVED ((a) rotated block, (b) exchanged pair), the spectral reference resolved',
        0.0 if (sa == 'UNRESOLVED' and sb == 'UNRESOLVED' and sg == 'resolved') else 1.0, 0.0,
        '(a) ||(1-P)h_m P||/||h_m|| %.3e %s; (b) %.1e with the mask test %s; spectral reference %.1e %s' % (ca, sa, cb, sb, cg, sg))
    n_fail = sum(r['status'] == 'FAIL' for r in res)
    os.makedirs(a.out, exist_ok=True)
    RC.dump_json(os.path.join(a.out, 'selftest_%s.json' % VERSION), dict(script=VERSION, code_md5=CODE_MD5, route2_common_md5=COMMON_MD5, results=res,
                                                                          n_fail=n_fail, n_pass=len(res) - n_fail, device=str(dev)))
    log('SELFTEST: %d PASS, %d FAIL' % (len(res) - n_fail, n_fail))
    return 1 if n_fail else 0


class DenseBand:
    """a dense Hermitian operator with Band's left() (for the small Fock fixtures)."""

    def __init__(self, A, dev):
        self.A = torch.as_tensor(A, dtype=CT, device=dev)

    def left(self, X):
        return self.A @ X


def unitary(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))[None, :]


def make_fixture_channel(kap, h, Q, hm, Phi, f, R2, ns, dev):
    hb, qb, hmb = Band.from_scipy(h, dev), Band.from_scipy(Q, dev), Band.from_scipy(hm, dev)
    e, U = torch.linalg.eigh(herm(hmb.dense()))
    mt = e < 0
    ch = dict(kappa=kap, w=16.0 * kap, hb=hb, qb=qb, hmb=hmb, Uo=U[:, mt].contiguous(), Uu=U[:, ~mt].contiguous(),
              Wou=(1.0 / (e[mt][:, None] - e[~mt][None, :])).to(CT), pure=True, n_sea=int(mt.sum().item()), n_dust=Phi.shape[1] - int(mt.sum().item()))
    ch['Phi'] = torch.as_tensor(Phi, dtype=CT, device=dev)
    ch['R2'] = torch.as_tensor(R2, dtype=CT, device=dev)
    ch['f'] = torch.as_tensor(f, dtype=RT, device=dev)
    ch['D'] = herm(ch['Phi'] @ adj(ch['Phi']) - ch['R2'] @ adj(ch['R2']))
    return ch


def site_vertices_dense(ch, ns):
    out = []
    for op in (ch['hb'], ch['qb']):
        Hd = op.dense().cpu().numpy()
        for n in range(ns):
            Pi = np.zeros((2 * ns, 2 * ns))
            Pi[2 * n, 2 * n] = Pi[2 * n + 1, 2 * n + 1] = 1.0
            out.append(0.5 * (Pi @ Hd + Hd @ Pi))
    return out


def fock_tools(n):
    """Jordan-Wigner annihilators on n modes and the Fock-space expectations used by the gates."""
    dim = 2 ** n
    cs = []
    for j in range(n):
        M = np.zeros((dim, dim))
        for s in range(dim):
            if (s >> j) & 1:
                sign = (-1) ** bin(s & ((1 << j) - 1)).count('1')
                M[s ^ (1 << j), s] = sign
        cs.append(M)
    cd = [c.T for c in cs]

    def dG(A):
        return sum(A[i, j] * cd[i] @ cs[j] for i in range(n) for j in range(n))

    def slater_var(Phi, A):
        vac = np.zeros(dim, complex)
        vac[0] = 1.0
        psi = vac
        for k in range(Phi.shape[1]):
            psi = sum(Phi[i, k] * cd[i] for i in range(n)) @ psi
        psi = psi / np.linalg.norm(psi)
        G = dG(A)
        m1 = np.vdot(psi, G @ psi)
        m2 = np.vdot(psi, G @ (G @ psi))
        return float(np.real(m2 - m1 ** 2))

    def thermal(H, beta):
        G = dG(H)
        ev, V = np.linalg.eigh(G)
        w = np.exp(-beta * (ev - ev.min()))
        return (V * w) @ V.conj().T / w.sum()

    def var(rho, A):
        G = dG(A)
        return float(np.real(np.trace(rho @ G @ G) - np.trace(rho @ G) ** 2))

    def ordered_pair(rho, A, Bm, P):
        """<W_P(A, B)>_rho with W_P = (1/2){G_A, G_B} + dGamma(K) + p_A p_B - Tr(A P B P), K = -(AB + BA)/2 + APB + BPA - p_B A - p_A B
        (the spec's one-body and constant terms of the ordering against P; <W_P>_P = 0 is checked by the caller)."""
        GA, GB = dG(A), dG(Bm)
        pA, pB = np.trace(P @ A), np.trace(P @ Bm)
        K = -0.5 * (A @ Bm + Bm @ A) + A @ P @ Bm + Bm @ P @ A - pB * A - pA * Bm
        val = np.trace(rho @ (0.5 * (GA @ GB + GB @ GA))) + np.trace(rho @ dG(K)) + pA * pB - np.trace(A @ P @ Bm @ P)
        return float(np.real(val))

    def gauss_state(Cc):
        """the quasi-free density matrix with covariance Cc (0 < eig < 1): exp(-dGamma(H))/Z with H = log((1 - C)/C)."""
        ev, V = np.linalg.eigh(Cc)
        ev = np.clip(ev, 1e-15, 1 - 1e-15)
        H = (V * np.log((1 - ev) / ev)) @ V.conj().T
        G = dG(H)
        e2, V2 = np.linalg.eigh(G)
        w = np.exp(-(e2 - e2.min()))
        return (V2 * w) @ V2.conj().T / w.sum()

    return dict(slater_var=slater_var, thermal=thermal, var=var, ordered_pair=ordered_pair, gauss_state=gauss_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--slice', nargs=2, metavar=('HIST', 'T'))
    ap.add_argument('--kappas', type=lambda s: [int(x) for x in s.split(',') if x], default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--dr', type=float, default=0.1)
    ap.add_argument('--rank-tol', type=float, default=1e-15)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='<internal>/route2_core')
    ap.add_argument('--save-kernels', action='store_true')
    ap.add_argument('--no-D-eig', action='store_true')
    ap.add_argument('--skip-nocur', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--route1-arrays', default=None, help="route 1's _arrays.npz of the same slice: its vectors on route 2's contractions")
    ap.add_argument('--r-eval-from', default=None, help='a route-2 json: evaluate at its r_eval rows (the dr study)')
    ap.add_argument('--pa-from', default=None, help='a route-2 _arrays.npz with PA (saved by --save-kernels): reuse the explicit response')
    ap.add_argument('--threads', default='8')
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(min(int(a.threads), 8))
    if a.selftest:
        return selftest(a)
    return run_slice(a)


if __name__ == '__main__':
    sys.exit(main())
