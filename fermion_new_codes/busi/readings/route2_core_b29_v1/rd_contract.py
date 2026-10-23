#!/usr/bin/env python3
"""rd_contract.py (busi lane, inbox #356): the reading's own contraction code for ROUTE2_core_B29_v1, the checking lane's form.

Nothing of either route is imported (no route2_*, core_hf_*, hf_star_variance_*, engine_*). Per slice:
  - the slice is read with numpy's npz loader; the chain of channel kappa is built here from its written definition (two components per
    site, R_n = n + 1, on-site m SZ + (kappa/R_n) SY, sine bond -(i/2) SX, two-step bond -(i/4) 1), as a Hermitian banded operator that
    stores only its upper diagonals; the clocked h_m = (1/2){N, h} + (1/2){v, Q} is the vertex of (N, v);
  - h_m is diagonalized here (torch eigh on the GPU) and compared with the stored e_k and mask;
  - the reference R2 and the state [sea | dust] are Loewdin-orthonormalized here;
  - the sea part Tr[P A (1 - P) A] of each compared vertex (route 1's stored beta and route 2's) is evaluated by FOUR forms: the residual
    form ||(1 - P) A R2||^2, the residual form after one reorthogonalization, the QR-complement form ||Q2^+ A R2||^2 (Householder QR of R2)
    and the Gram quadratic form beta^T G_P beta with G_P = Re Tr(A_mu (1 - P) A_nu P) rebuilt here; the cancellation scales of the
    unreduced difference (||A R2||^2 / Var_P) and of the Gram quadratic form (sum |beta_mu G_mu nu beta_nu| / Var_P) are printed;
  - the state variance by the residual form on the pure state; the signed excess; sqrt(Var_C)/M_H;
  - the sources s from the Loewdin state against the record's eps_tot and J_tot; N_B; R_50 from the dust columns;
  - the exchange E_F = -(1/2) sum_k w_k <K, T_k> for K = B (bare) and V (dressed), by an EIGEN-ROUTE that never forms T_k
    (K = sum_a lambda_a x_a x_a^T, <K, T_k> = sum_a lambda_a Tr(A_{x_a} D_k A_{x_a} D_k)) and by a T-matrix route; the blocks;
  - the reference response P^A on test vectors by the exact eigenbasis operator, against route 2's stored explicit matrix;
  - the dressing (I + B P^A) beta = b and V = (I + B P^A)^-1 B by dense solves on the stored B and P^A, sigma_min(I + B P^A), eig B, eig V,
    E_pair = (1/2) s^T K s.
Every number is set beside route 1's json (card run) and route 2's json. Output: <out>/rd_<hist>_T<TTTT.TT>.json (+ the Loewdin R2 columns
as float64 npy when --save-r2 is given, for the long-double check). Box only; one GPU named by --device; nice 10; at most 8 CPU threads.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CT = torch.complex128
RT = torch.float64
T0 = time.time()
VERSION = 'rd_contract_v1'

HIST = {
    'prod': dict(tag='c19_rs64_A_R2_tau8_ell4_avg_r11',
                 records='<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json',
                 fields='<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz',
                 slices='<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11',
                 r1='<internal>/prod'),
    's8': dict(tag='c27_rs64_A_R2_tau8_ell8_prep4_avg',
               records='<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json',
               fields='<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz',
               slices='<internal>/s8_slices',
               r1='<internal>/s8'),
}
R2DIR = '<internal>/route2_core'
WELL_DIR = '<internal>/well'
LABELS = ('R_in', 'R_50', 'R_t')


def log(*a):
    print('[%7.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def fro2(X):
    return float((X.real ** 2 + X.imag ** 2).sum().item()) if X.is_complex() else float((X ** 2).sum().item())


def adj(X):
    return X.conj().transpose(0, 1)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ============================================================================ Hermitian banded operators (upper diagonals only)
class Herm:
    """a Hermitian banded operator on C^n stored by its upper diagonals: u[d][i] = op[i, i + d] for d >= 0 (zero where i + d >= n);
    op[i + d, i] = conj(u[d][i]) is implied."""

    def __init__(self, n, dev):
        self.n, self.dev, self.u = n, dev, {}

    def put(self, d, vals):
        self.u[d] = torch.as_tensor(np.asarray(vals, complex), dtype=CT, device=self.dev)

    def left(self, X):
        """op @ X."""
        n = self.n
        out = torch.zeros_like(X)
        for d, u in self.u.items():
            if d == 0:
                out += u[:, None] * X
            else:
                out[:n - d] += u[:n - d, None] * X[d:]
                out[d:] += torch.conj(u[:n - d])[:, None] * X[:n - d]
        return out

    def right(self, X):
        """X @ op: (X op)[:, j] = sum_i X[:, i] op[i, j]; i = j - d gives op[j - d, j] = u[d][j - d], i = j + d gives conj(u[d][j])."""
        n = self.n
        out = torch.zeros_like(X)
        for d, u in self.u.items():
            if d == 0:
                out += X * u[None, :]
            else:
                out[:, d:] += X[:, :n - d] * u[None, :n - d]
                out[:, :n - d] += X[:, d:] * torch.conj(u[None, :n - d])
        return out

    def diag_of_left(self, X):
        """diag(op @ X)[i] = sum_j op[i, j] X[j, i]."""
        n = self.n
        out = torch.zeros(n, dtype=CT, device=self.dev)
        for d, u in self.u.items():
            if d == 0:
                out += u * torch.diagonal(X)
            else:
                out[:n - d] += u[:n - d] * torch.diagonal(X, offset=-d)      # X[i + d, i]
                out[d:] += torch.conj(u[:n - d]) * torch.diagonal(X, offset=d)   # X[i - d, i] for i >= d
        return out

    def fro2(self):
        tot = 0.0
        for d, u in self.u.items():
            s = float((u.real ** 2 + u.imag ** 2).sum().item())
            tot += s if d == 0 else 2.0 * s
        return tot

    def dense(self):
        return self.left(torch.eye(self.n, dtype=CT, device=self.dev))


def chain(ns, m, kappa, dev):
    """h (unit clock, no flow) and Q of channel kappa from the written definition."""
    n = 2 * ns
    R = np.arange(1, ns + 1, dtype=float)
    u0 = np.zeros(n, complex)
    u0[0::2] = m
    u0[1::2] = -m
    u1 = np.zeros(n, complex)
    u1[0::2] = -1j * kappa / R                     # (kappa/R_n) SY: entry (2n, 2n+1) = -i kappa/R_n, (2n+1, 2n) = +i kappa/R_n (implied)
    u1[1:2 * ns - 2:2] = -0.5j                      # sine bond -(i/2) SX between sites n, n+1: entry (2n+1, 2n+2)
    u3 = np.zeros(n, complex)
    u3[0:2 * ns - 2:2] = -0.5j                      # the same bond: entry (2n, 2n+3)
    u4 = np.zeros(n, complex)
    u4[0:2 * ns - 4] = -0.25j                       # two-step bond -(i/4) 1 between sites n, n+2: entries (2n, 2n+4), (2n+1, 2n+5)
    h = Herm(n, dev)
    h.put(0, u0)
    h.put(1, u1)
    h.put(3, u3)
    Q = Herm(n, dev)
    Q.put(4, u4)
    return h, Q


def vertex(h, Q, x, ns, parts='ep'):
    """A_x = (1/2){D_e, h} + (1/2){D_p, Q}: A[i, j] = (x_i + x_j)/2 op[i, j] with x_i the site value of component i."""
    n = 2 * ns
    A = Herm(n, h.dev)
    x = np.asarray(x, float)
    for src, xx, tag in ((h, x[:ns], 'e'), (Q, x[ns:], 'p')):
        if tag not in parts:
            continue
        xt = torch.as_tensor(np.repeat(xx, 2), dtype=CT, device=h.dev)
        for d, u in src.u.items():
            s = torch.zeros(n, dtype=CT, device=h.dev)
            if d == 0:
                s = xt
            else:
                s[:n - d] = 0.5 * (xt[:n - d] + xt[d:])
            A.u[d] = A.u.get(d, torch.zeros(n, dtype=CT, device=h.dev)) + s * u
    return A


def loewdin(X):
    """X (X^+ X)^(-1/2) by the eigenvectors of the overlap; returns (Y, max|S - I| before, after)."""
    S = adj(X) @ X
    S = 0.5 * (S + adj(S))
    n = S.shape[0]
    I = torch.eye(n, dtype=CT, device=X.device)
    d0 = float(torch.max(torch.abs(S - I)).item())
    w, V = torch.linalg.eigh(S)
    Y = X @ ((V * (w ** -0.5).to(CT)[None, :]) @ adj(V))
    S2 = adj(Y) @ Y
    d1 = float(torch.max(torch.abs(S2 - I)).item())
    return Y, d0, d1


def site_density(op, Phi, ns, f=None):
    """Re Tr(Pi_n op C) per site, C = Phi diag(f) Phi^+."""
    v = torch.real(torch.conj(Phi) * op.left(Phi))
    if f is not None:
        v = v * f[None, :]
    return v.sum(dim=1).reshape(ns, 2).sum(dim=1).cpu().numpy()


def blocksum(X, Y, ns):
    """S[n, n'] = sum_{i in n, j in n'} X_ij Y_ji."""
    return (X * Y.transpose(0, 1)).reshape(ns, 2, ns, 2).sum(dim=(1, 3))


def quartet(h, Q, X, Y, ns):
    """F[mu, nu] = Tr(A_mu X A_nu Y) for the site vertices A_mu = (1/2){Pi_n, h_t} (energy block first, then current): expanding
    {Pi_n, h}X{Pi_n', h'}Y into its four orderings and reading each as Tr(Pi_n [.] Pi_n' [.]):
    (1/4)[S(hX, h'Y) + S(hXh', Y) + S(X, h'Yh) + S(Xh', Yh)]."""
    d = 2 * ns
    F = torch.zeros((d, d), dtype=CT, device=X.device)
    ops = (h, Q)
    hX = [o.left(X) for o in ops]
    hY = [o.left(Y) for o in ops]
    for t in range(2):
        for tp in range(2):
            S = blocksum(hX[t], hY[tp], ns)
            S += blocksum(ops[tp].right(hX[t]), Y, ns)
            S += blocksum(X, ops[t].right(hY[tp]), ns)
            S += blocksum(ops[tp].right(X), ops[t].right(Y), ns)
            F[t * ns:(t + 1) * ns, tp * ns:(tp + 1) * ns] = 0.25 * S
    return F


def sea_forms(A, R2, Q2=None):
    """Var_P = Tr[P A (1 - P) A] for P = R2 R2^+ (orthonormal R2) by the residual, reorthogonalized-residual, unreduced and
    QR-complement forms, with the cancellation scale of the unreduced form."""
    X = A.left(R2)
    M = adj(R2) @ X
    Res = X - R2 @ M
    v_res = fro2(Res)
    g = adj(R2) @ Res
    Res2 = Res - R2 @ g
    big = fro2(X)
    out = dict(residual=v_res, reorth=fro2(Res2), unreduced=big - fro2(M), scale_unreduced=big / max(v_res, 1e-300),
               inspan_defect_rel=math.sqrt(fro2(g) / max(v_res, 1e-300)))
    if Q2 is not None:
        out['qr_complement'] = fro2(adj(Q2) @ X)
    return out


def state_var(A, Phi):
    X = A.left(Phi)
    Res = X - Phi @ (adj(Phi) @ X)
    return fro2(Res), fro2(X)


def gram_forms(G, beta):
    """beta^T G beta and its absolute-sum cancellation scale sum |beta_mu G_mu nu beta_nu| / |beta^T G beta|."""
    b = torch.as_tensor(np.asarray(beta, float), dtype=RT, device=G.device)
    Gb = G @ b
    val = float((b @ Gb).item())
    absum = float((torch.abs(b)[:, None] * torch.abs(G) * torch.abs(b)[None, :]).sum().item())
    return val, absum / max(abs(val), 1e-300)


# ============================================================================ the record
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
    Z = np.load(H['fields'], allow_pickle=True)
    Tz = np.asarray(Z['T'], float)
    i = int(np.argmin(np.abs(Tz / rs - t)))
    assert abs(Tz[i] / rs - t) < 1e-6
    return dict(m=float(A['m']), g=float(A['g']), ell=float(A['ell']), rs=rs, index=i, E=float(rec['E']),
                trapping_zeros=list(rec.get('trapping_zeros') or []), eps_tot=np.asarray(Z['eps_tot'][i], float), J_tot=np.asarray(Z['J_tot'][i], float),
                N_applied=np.asarray(Z['N_applied'][i], float), v_applied=np.asarray(Z['v_applied'][i], float))


def half_radius(R, cum):
    """the radius where the cumulative dust fraction reaches 1/2, linear between the site radii (the fraction at R_n inclusive of site n)."""
    q = cum / cum[-1]
    k = int(np.argmax(q >= 0.5))
    if k == 0:
        return float(R[0])
    return float(R[k - 1] + (0.5 - q[k - 1]) / (q[k] - q[k - 1]) * (R[k] - R[k - 1]))


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', nargs=2, required=True, metavar=('HIST', 'T'))
    ap.add_argument('--kappas', type=lambda s: [int(x) for x in s.split(',') if x], default=None)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default='<internal>/route2_core_b29_v1')
    ap.add_argument('--qr-labels', default='R_in,R_50', help='labels on which the QR-complement form is evaluated')
    ap.add_argument('--no-eigroute', action='store_true')
    ap.add_argument('--no-response', action='store_true')
    ap.add_argument('--save-r2', action='store_true', help='save the Loewdin R2 columns (float64) per channel for the long-double check')
    ap.add_argument('--eig-tol', type=float, default=1e-13, help='eigen-route: drop |lambda| below this times max|lambda|')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(8)
    dev = torch.device(a.device)
    hist, t = a.slice[0], float(a.slice[1])
    H = HIST[hist]
    tag = '%s_T%07.2f' % (hist, t)
    os.makedirs(a.out, exist_ok=True)
    sp = os.path.join(H['slices'], 'orbitals_%s_T%07.2f.npz' % (H['tag'], t))
    rec = load_record(hist, t)
    Z = np.load(sp, allow_pickle=False)
    R = np.asarray(Z['R'], float)
    ns = len(R)
    n = 2 * ns
    N = np.asarray(Z['N_applied'], float)
    v = np.asarray(Z['v_applied'], float)
    kap_all = [int(k) for k in np.atleast_1d(Z['kappas'])]
    wts = {int(k): float(w) for k, w in zip(kap_all, np.atleast_1d(Z['weights']))}
    kappas = [k for k in kap_all if not a.kappas or k in a.kappas]
    full = (kappas == kap_all)
    c64 = np.asarray(Z['sea_%d' % kap_all[0]]).dtype == np.complex64
    log('=== %s %s T/r_s %g: %s (%s); m %g g %g ell %g; device %s %s ===' % (VERSION, hist, t, sp, 'complex64' if c64 else 'complex128', rec['m'], rec['g'], rec['ell'],
                                                                           dev, torch.cuda.get_device_name(dev) if dev.type == 'cuda' else ''))
    J2 = json.load(open(os.path.join(R2DIR, tag + '.json')))
    A2 = np.load(os.path.join(R2DIR, tag + '_arrays.npz'))
    J1 = json.load(open(os.path.join(H['r1'], 'core_hf_%s.json' % tag)))
    A1 = np.load(os.path.join(H['r1'], 'core_hf_%s_arrays.npz' % tag))
    out = dict(script=VERSION, code_md5=md5(os.path.abspath(__file__)), history=hist, T_over_rs=t, slice=sp, slice_md5=None, complex64=bool(c64),
               device=str(dev), gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None, kappas=kappas, channel_subset=not full,
               route2_json=os.path.join(R2DIR, tag + '.json'), route1_json=os.path.join(H['r1'], 'core_hf_%s.json' % tag),
               inputs_md5=dict(route2_json=md5(os.path.join(R2DIR, tag + '.json')), route1_json=md5(os.path.join(H['r1'], 'core_hf_%s.json' % tag)),
                               route2_arrays=md5(os.path.join(R2DIR, tag + '_arrays.npz')), route1_arrays=md5(os.path.join(H['r1'], 'core_hf_%s_arrays.npz' % tag))))
    checks = []

    def chk(name, val, tol):
        ok = bool(val <= tol)
        checks.append(dict(name=name, value=float(val), tol=tol, status='PASS' if ok else 'FAIL'))
        log('  [%s] %s: %.3e (tol %.0e)' % ('PASS' if ok else 'FAIL', name, val, tol))

    chk('slice applied fields = the fields npz row', max(float(np.max(np.abs(N - rec['N_applied']))), float(np.max(np.abs(v - rec['v_applied'])))), 0.0)
    chk('R = 1..ns', float(np.max(np.abs(R - np.arange(1, ns + 1)))), 0.0)
    # ---------------- the vertices compared (route 1's stored betas, route 2's)
    betas = {}
    for l in LABELS:
        for nm in ('noref', 'full'):
            k1, k2 = 'beta_%s_%s' % (l, nm), 'beta_%s_%s' % (l, nm)
            if k1 in A1.files and k2 in A2.files:
                b1, b2 = np.asarray(A1[k1], float), np.asarray(A2[k2], float)
                betas[(l, nm)] = dict(r1=b1, r2=b2, rel_diff=float(np.linalg.norm(b1 - b2) / np.linalg.norm(b2)))
    log('  vertices: ' + ', '.join('%s %s %.1e' % (l, nm, e['rel_diff']) for (l, nm), e in betas.items()))
    # ---------------- the kernels and their spectra (route 2's stored B, P^A, V; checked here by dense solves)
    B = torch.as_tensor(np.asarray(A2['B'], float), dtype=RT, device=dev)
    PA = torch.as_tensor(np.asarray(A2['PA'], float), dtype=RT, device=dev)
    V2 = torch.as_tensor(np.asarray(A2['V'], float), dtype=RT, device=dev)
    kern = {}
    t1 = time.time()
    I = torch.eye(n, dtype=RT, device=dev)
    Jm = I + B @ PA
    Vm = torch.linalg.solve(Jm, B)
    kern['V_asym_raw'] = float(torch.linalg.norm(Vm - Vm.T).item() / torch.linalg.norm(Vm).item())
    Vm = 0.5 * (Vm + Vm.T)
    kern['V_mine_vs_route2_rel_F'] = float(torch.linalg.norm(Vm - V2).item() / torch.linalg.norm(V2).item())
    kern['B_asym'] = float(torch.linalg.norm(B - B.T).item() / torch.linalg.norm(B).item())
    kern['PA_asym'] = float(torch.linalg.norm(PA - PA.T).item() / torch.linalg.norm(PA).item())
    sv = torch.linalg.svdvals(Jm)
    kern['sigma_min_I_plus_BPA'] = float(sv[-1].item())
    kern['sigma_max_I_plus_BPA'] = float(sv[0].item())
    eB = torch.linalg.eigvalsh(B)
    eV = torch.linalg.eigvalsh(Vm)
    ePA = torch.linalg.eigvalsh(PA)
    kern.update(B_eig_min=float(eB[0]), B_eig_max=float(eB[-1]), V_eig_min=float(eV[0]), V_eig_max=float(eV[-1]), PA_eig_min=float(ePA[0]), PA_eig_max=float(ePA[-1]),
                V_minus_B_rel_F=float(torch.linalg.norm(Vm - B).item() / torch.linalg.norm(B).item()), norm_PA_B_2=float(torch.linalg.matrix_norm(PA @ B, ord=2).item()),
                seconds=time.time() - t1)
    log('  kernels: sigma_min %.10f; eig B [%.10f, %.10f]; eig V [%.10f, %.10f]; eig PA [%.3e, %.3e]; V mine vs route 2 %.1e; ||V-B||/||B|| %.10f; %.0f s'
        % (kern['sigma_min_I_plus_BPA'], eB[0], eB[-1], eV[0], eV[-1], ePA[0], ePA[-1], kern['V_mine_vs_route2_rel_F'], kern['V_minus_B_rel_F'], kern['seconds']))
    # my dressed betas from the stored noref vertices
    dress = {}
    for l in LABELS:
        if (l, 'noref') in betas and (l, 'full') in betas:
            b = torch.as_tensor(betas[(l, 'noref')]['r2'], dtype=RT, device=dev)
            bf = torch.linalg.solve(Jm, b).cpu().numpy()
            dress[l] = dict(mine_vs_route2_rel=float(np.linalg.norm(bf - betas[(l, 'full')]['r2']) / np.linalg.norm(bf)),
                            mine_vs_route1_rel=float(np.linalg.norm(bf - betas[(l, 'full')]['r1']) / np.linalg.norm(bf)),
                            residual_rel=float(np.linalg.norm((Jm @ torch.as_tensor(bf, dtype=RT, device=dev) - b).cpu().numpy()) / np.linalg.norm(b.cpu().numpy())))
            betas[(l, 'mine_full')] = dict(r1=bf, r2=bf, rel_diff=0.0)
    log('  dressed betas (dense solve on the stored B, P^A): ' + ', '.join('%s mine vs r2 %.1e vs r1 %.1e' % (l, d['mine_vs_route2_rel'], d['mine_vs_route1_rel']) for l, d in dress.items()))
    # the eigen-route's kernel eigenpairs
    eig_kernels = {}
    if not a.no_eigroute:
        for nmk, K in (('bare', B), ('dressed', Vm)):
            lam, X = torch.linalg.eigh(K)
            keep = torch.abs(lam) > a.eig_tol * torch.max(torch.abs(lam))
            eig_kernels[nmk] = (lam[keep].clone(), X[:, keep].contiguous())
            log('  eigen-route: %s kernel, %d of %d eigenpairs kept (|lambda| > %.0e max)' % (nmk, int(keep.sum().item()), n, a.eig_tol))
    del Jm
    torch.cuda.empty_cache()
    # ---------------- per channel
    chans = []
    s_tot = np.zeros(n)
    s_raw = np.zeros(n)
    dust_site = np.zeros(ns)
    NB = 0.0
    var_acc = {}      # (label, variant, source) -> dict of accumulated forms
    exch_acc = {'bare': dict(eig=0.0, eig_blocks=np.zeros(3), T=0.0, T_blocks=np.zeros(3)), 'dressed': dict(eig=0.0, eig_blocks=np.zeros(3), T=0.0, T_blocks=np.zeros(3))}
    rng = np.random.default_rng(20261020)
    tests = {'random1': rng.standard_normal(n), 'random2': rng.standard_normal(n)}
    if 'b_R_50' in A2.files:
        tests['b_R_50'] = np.asarray(A2['b_R_50'], float)
    tests['phi_applied'] = np.concatenate([N, v])
    resp_acc = {k: np.zeros(n) for k in tests}
    qr_labels = set(a.qr_labels.split(',')) if a.qr_labels else set()
    for k in kappas:
        tc = time.time()
        w = wts[k]
        h, Q = chain(ns, rec['m'], float(k), dev)
        hm = vertex(h, Q, np.concatenate([N, v]), ns)
        ch = dict(kappa=k, w=w)
        # eigh of h_m (mine) against the stored e_k and mask
        te = time.time()
        Hd = hm.dense()
        Hd = 0.5 * (Hd + adj(Hd))
        e, U = torch.linalg.eigh(Hd)
        del Hd
        e_np = e.cpu().numpy()
        es = np.asarray(Z['e_%d' % k], float).reshape(-1)
        ms = np.unpackbits(np.asarray(Z['mask_%d' % k], np.uint8))[:n].astype(bool)
        mask = e_np < 0
        ch['eigh_seconds'] = time.time() - te
        ch['e_vs_stored_max_abs'] = float(np.max(np.abs(es - e_np)))
        ch['stored_mask_equals_e_lt_0'] = bool(np.array_equal(ms, mask))
        ch['n_neg'] = int(mask.sum())
        ch['gap'] = float(e_np[~mask].min() - e_np[mask].max())
        mt = torch.as_tensor(mask, device=dev)
        Uo, Uu = U[:, mt].contiguous(), U[:, ~mt].contiguous()
        eo, eu = e[mt], e[~mask.__invert__() == False] if False else e[~mt]
        del U
        # the state and the reference
        sea = torch.as_tensor(np.asarray(Z['sea_%d' % k]), device=dev).to(CT)
        dk = 'dust_%d' % k
        dust = torch.as_tensor(np.asarray(Z[dk]), device=dev).to(CT) if dk in Z.files else torch.zeros((n, 0), dtype=CT, device=dev)
        fd = np.asarray(Z['f_%d' % k], float).reshape(-1)
        n_s, n_d = sea.shape[1], dust.shape[1]
        f = np.concatenate([np.ones(n_s), fd])
        ch.update(n_sea=int(n_s), n_dust=int(n_d), f_min=float(f.min()), f_max=float(f.max()))
        assert np.all(np.abs(f - 1.0) < 1e-12), 'the residual form needs unit occupations'
        Phi_raw = torch.cat([sea, dust], dim=1)
        del sea
        R2raw = torch.as_tensor(np.asarray(Z['R2_%d' % k]), device=dev).to(CT)
        Phi, d0, d1 = loewdin(Phi_raw)
        R2, r0, r1 = loewdin(R2raw)
        ch.update(stored_orthonormality=d0, loewdin_orthonormality=d1, ref_stored_orthonormality=r0, ref_loewdin_orthonormality=r1)
        ch['pref_resolution_defect'] = math.sqrt(2.0 * fro2(adj(Uu) @ R2)) / math.sqrt(R2.shape[1])
        HP = hm.left(R2)
        ch['comm_h_Pref_rel'] = math.sqrt(fro2(HP - R2 @ (adj(R2) @ HP))) / math.sqrt(hm.fro2())
        del HP
        if a.save_r2:
            d_r2 = os.path.join(a.out, 'r2cols_' + tag)
            os.makedirs(d_r2, exist_ok=True)
            np.save(os.path.join(d_r2, 'R2_loewdin_k%d.npy' % k), R2.cpu().numpy())
        # sources
        ft = torch.as_tensor(f, dtype=RT, device=dev)
        s_k = np.concatenate([w * (site_density(h, Phi, ns) - site_density(h, R2, ns)), w * (site_density(Q, Phi, ns) - site_density(Q, R2, ns))])
        s_tot += s_k
        s_raw += np.concatenate([w * (site_density(h, Phi_raw, ns, ft) - site_density(h, R2raw, ns)), w * (site_density(Q, Phi_raw, ns, ft) - site_density(Q, R2raw, ns))])
        if n_d:
            dust_site += w * (torch.abs(Phi_raw[:, n_s:]) ** 2 * ft[n_s:][None, :]).sum(dim=1).reshape(ns, 2).sum(dim=1).cpu().numpy()
        del Phi_raw, R2raw
        # C, P, D
        C = Phi @ adj(Phi)
        C = 0.5 * (C + adj(C))
        P = R2 @ adj(R2)
        P = 0.5 * (P + adj(P))
        D = C - P
        ch['TrD'] = float(torch.real(torch.trace(D)).item())
        NB += w * ch['TrD']
        # the Gram G_P (route 1's object, rebuilt here) and the QR complement of R2
        tg = time.time()
        Ieye = torch.eye(n, dtype=CT, device=dev)
        GP = torch.real(quartet(h, Q, Ieye - P, P, ns))
        GP = 0.5 * (GP + GP.T)
        del Ieye, C, P
        ch['gram_seconds'] = time.time() - tg
        Q2 = None
        if qr_labels:
            Qf, _ = torch.linalg.qr(R2, mode='complete')
            Q2 = Qf[:, R2.shape[1]:].contiguous()
            del Qf
        # the variances of the compared vertices
        ch['vertices'] = {}
        for (l, nm), e_ in betas.items():
            for src in ('r1', 'r2'):
                if src == 'r2' and nm != 'mine_full' and e_['rel_diff'] == 0.0:
                    continue
                if nm == 'mine_full' and src == 'r2':
                    continue
                beta = e_[src]
                A = vertex(h, Q, beta, ns)
                sf = sea_forms(A, R2, Q2 if l in qr_labels else None)
                vc, big_c = state_var(A, Phi)
                gv, gscale = gram_forms(GP, beta)
                row = dict(VarC_residual=vc, scale_C=big_c / max(vc, 1e-300), VarP=sf, VarP_gram=gv, gram_abs_cancellation=gscale)
                ch['vertices']['%s %s %s' % (l, nm, src)] = row
                acc = var_acc.setdefault((l, nm, src), dict(VarC=0.0, VarP_residual=0.0, VarP_reorth=0.0, VarP_unreduced=0.0, VarP_qr=0.0, VarP_gram=0.0,
                                                           max_scale_unreduced=0.0, max_gram_cancellation=0.0, per_channel=[]))
                acc['VarC'] += w * vc
                acc['VarP_residual'] += w * sf['residual']
                acc['VarP_reorth'] += w * sf['reorth']
                acc['VarP_unreduced'] += w * sf['unreduced']
                acc['VarP_qr'] += w * sf.get('qr_complement', float('nan'))
                acc['VarP_gram'] += w * gv
                acc['max_scale_unreduced'] = max(acc['max_scale_unreduced'], sf['scale_unreduced'])
                acc['max_gram_cancellation'] = max(acc['max_gram_cancellation'], gscale)
                acc['per_channel'].append(dict(kappa=k, w=w, VarC=vc, VarP_residual=sf['residual'], VarP_reorth=sf['reorth'], VarP_unreduced=sf['unreduced'],
                                               VarP_qr=sf.get('qr_complement'), VarP_gram=gv, scale_unreduced=sf['scale_unreduced'], gram_cancellation=gscale,
                                               inspan_defect_rel=sf['inspan_defect_rel']))
        del GP, Q2
        # the exchange: eigen-route and T-route
        texc = time.time()
        if not a.no_eigroute:
            for nmk, (lam, Xv) in eig_kernels.items():
                lam_np = lam.cpu().numpy()
                Xn = Xv.cpu().numpy()
                acc_tot = 0.0
                acc_blk = np.zeros(3)
                for ia in range(len(lam_np)):
                    x = Xn[:, ia]
                    Ae = vertex(h, Q, x, ns, parts='e')
                    Af = vertex(h, Q, x, ns, parts='ep')
                    Ye = Ae.left(D)
                    Y = Af.left(D)
                    Yp = Y - Ye
                    tee = float(torch.real(torch.sum(Ye * Ye.transpose(0, 1))).item())
                    tpp = float(torch.real(torch.sum(Yp * Yp.transpose(0, 1))).item())
                    tt = float(torch.real(torch.sum(Y * Y.transpose(0, 1))).item())
                    acc_tot += lam_np[ia] * tt
                    acc_blk += lam_np[ia] * np.array([tee, tt - tee - tpp, tpp])
                    del Ye, Y, Yp
                exch_acc[nmk]['eig'] += -0.5 * w * acc_tot
                exch_acc[nmk]['eig_blocks'] += -0.5 * w * acc_blk
        Tk = torch.real(quartet(h, Q, D, D, ns))
        for nmk, K in (('bare', B), ('dressed', Vm)):
            tot = float(torch.sum(K * Tk).item())
            blk = lambda X, i, j: X[i * ns:(i + 1) * ns, j * ns:(j + 1) * ns]  # noqa: E731
            bl = np.array([float(torch.sum(blk(K, 0, 0) * blk(Tk, 0, 0)).item()),
                           float(torch.sum(blk(K, 0, 1) * blk(Tk, 0, 1)).item()) + float(torch.sum(blk(K, 1, 0) * blk(Tk, 1, 0)).item()),
                           float(torch.sum(blk(K, 1, 1) * blk(Tk, 1, 1)).item())])
            exch_acc[nmk]['T'] += -0.5 * w * tot
            exch_acc[nmk]['T_blocks'] += -0.5 * w * bl
        ch['T_diag_vs_route2_rel'] = float(np.linalg.norm(torch.diagonal(Tk).cpu().numpy() - np.asarray(A2['T_diag_kappa%d' % k], float)) / np.linalg.norm(np.asarray(A2['T_diag_kappa%d' % k], float))) if ('T_diag_kappa%d' % k) in A2.files else None
        del Tk, D
        ch['exchange_seconds'] = time.time() - texc
        # the response on the test vectors (exact eigenbasis operator)
        if not a.no_response:
            tr_ = time.time()
            W = (1.0 / (eo[:, None] - eu[None, :])).to(CT)
            for nmv, x in tests.items():
                A = vertex(h, Q, x, ns)
                F = adj(Uo) @ A.left(Uu)
                Zm = Uo @ (F * W)
                Pi = Zm @ adj(Uu)
                Pi = Pi + adj(Pi)
                ye = torch.real(h.diag_of_left(Pi)).reshape(ns, 2).sum(dim=1).cpu().numpy()
                yp = torch.real(Q.diag_of_left(Pi)).reshape(ns, 2).sum(dim=1).cpu().numpy()
                resp_acc[nmv] += w * np.concatenate([ye, yp])
                del F, Zm, Pi
            del W
            ch['response_seconds'] = time.time() - tr_
        del Uo, Uu, Phi, R2
        torch.cuda.empty_cache()
        ch['seconds'] = time.time() - tc
        chans.append(ch)
        log('  kappa %d: n_neg %d dust %d; e vs stored %.1e; mask==e<0 %s; orth %.1e->%.1e; P_ref res %.1e; comm %.1e; Tr D %.6f; T diag vs r2 %s; %.0f s'
            % (k, ch['n_neg'], n_d, ch['e_vs_stored_max_abs'], ch['stored_mask_equals_e_lt_0'], d0, d1, ch['pref_resolution_defect'], ch['comm_h_Pref_rel'], ch['TrD'],
               ('%.1e' % ch['T_diag_vs_route2_rel']) if ch['T_diag_vs_route2_rel'] is not None else '-', ch['seconds']))
    # ---------------- slice-level results
    nrm = lambda x: float(np.max(np.abs(x)))  # noqa: E731
    src = dict(N_B=NB, s_total_vs_route2_rel=float(np.linalg.norm(s_tot - np.asarray(A2['s_total'])) / np.linalg.norm(s_tot)),
               s_total_vs_route1_rel=float(np.linalg.norm(s_tot - np.asarray(A1['s_total'])) / np.linalg.norm(s_tot)))
    if full:
        src['eps_raw_vs_record_max_rel'] = nrm(s_raw[:ns] - rec['eps_tot']) / nrm(rec['eps_tot'])
        src['J_raw_vs_record_max_rel'] = nrm(s_raw[ns:] - rec['J_tot']) / nrm(rec['J_tot'])
        src['eps_loewdin_vs_record_max_rel'] = nrm(s_tot[:ns] - rec['eps_tot']) / nrm(rec['eps_tot'])
        src['J_loewdin_vs_record_max_rel'] = nrm(s_tot[ns:] - rec['J_tot']) / nrm(rec['J_tot'])
        chk('N_B = sum w Tr D vs 640', abs(NB - 640.0), 1e-5 if c64 else 1e-9)
        chk('sources (raw columns) vs the record eps_tot (max rel)', src['eps_raw_vs_record_max_rel'], 1e-4 if c64 else 1e-11)
    chk('e vs stored e_k (max abs, all channels)', max(c['e_vs_stored_max_abs'] for c in chans), 1e-10)
    chk('stored mask == (e < 0) of my eigh (channels failing)', float(sum(not c['stored_mask_equals_e_lt_0'] for c in chans)), 0.0)
    chk('Loewdin orthonormality (max)', max(max(c['loewdin_orthonormality'], c['ref_loewdin_orthonormality']) for c in chans), 1e-13)
    radii = {}
    tz = rec['trapping_zeros']
    if len(tz) >= 2:
        radii['R_in'] = float(tz[0])
        radii['R_t'] = float(tz[-1])
    dc = np.cumsum(dust_site)
    if dc[-1] > 0:
        radii['R_50'] = half_radius(R, dc)
        wj = os.path.join(WELL_DIR, 'well_%s_T%07.2f.json' % (hist, t))
        if os.path.exists(wj) and full:
            src['R_50_vs_well_json_abs'] = abs(radii['R_50'] - float(json.load(open(wj))['dust']['R_50']))
    src['radii'] = radii
    src['N_enclosed'] = {l: float(np.interp(radii[l], R, dc)) for l in radii}
    # the exchange and the pair energies
    exch = {}
    st = torch.as_tensor(s_tot, dtype=RT, device=dev)
    s2 = torch.as_tensor(np.asarray(A2['s_total'], float), dtype=RT, device=dev)
    for nmk, K in (('bare', B), ('dressed', Vm)):
        Ep = 0.5 * float((st @ (K @ st)).item())
        Ep2 = 0.5 * float((s2 @ (K @ s2)).item())
        e_ = exch_acc[nmk]
        exch[nmk] = dict(E_F_eigroute=e_['eig'] if not a.no_eigroute else None, E_F_Troute=e_['T'], E_pair_my_s=Ep, E_pair_route2_s=Ep2,
                         blocks_eigroute=dict(zip(('energy', 'mixed', 'current'), e_['eig_blocks'].tolist())) if not a.no_eigroute else None,
                         blocks_Troute=dict(zip(('energy', 'mixed', 'current'), e_['T_blocks'].tolist())),
                         E_F_over_abs_E_pair=(e_['T'] / abs(Ep)), E_F_over_abs_E_pair_eig=(e_['eig'] / abs(Ep)) if not a.no_eigroute else None)
        if not a.no_eigroute:
            exch[nmk]['eig_vs_T_rel'] = rel(e_['eig'], e_['T'])
        log('  exchange [%s]: E_F eigen-route %s, T-route %.12e; E_pair %.12e (route 2 s: %.12e); blocks T-route %s'
            % (nmk, ('%.12e' % e_['eig']) if not a.no_eigroute else '-', e_['T'], Ep, Ep2, {kk: '%.6e' % vv for kk, vv in exch[nmk]['blocks_Troute'].items()}))
    exch['dressed_over_bare_E_F_minus_1'] = exch['dressed']['E_F_Troute'] / exch['bare']['E_F_Troute'] - 1.0
    # the response
    resp = {}
    if not a.no_response:
        PAn = PA.cpu().numpy()
        for nmv, x in tests.items():
            ym = PAn @ x
            ye = resp_acc[nmv]
            resp[nmv] = dict(mine_vs_route2_matrix_rel=float(np.linalg.norm(ym - ye) / max(np.linalg.norm(ye), 1e-300)), norm_exact=float(np.linalg.norm(ye)),
                             norm_matrix=float(np.linalg.norm(ym)))
            if nmv == 'phi_applied':
                resp[nmv]['PA_phi_over_phi_exact'] = float(np.linalg.norm(ye) / np.linalg.norm(x))
                resp[nmv]['PA_phi_over_phi_matrix'] = float(np.linalg.norm(ym) / np.linalg.norm(x))
        log('  response: my exact operator vs route 2 matrix: %s' % {k_: '%.2e' % v_['mine_vs_route2_matrix_rel'] for k_, v_ in resp.items()})
    # ---------------- the variances assembled, beside both routes
    MH2 = {l: J2['variance'][l]['M_H'] for l in LABELS if l in J2['variance']}
    var = {}
    for (l, nm, s_), acc in var_acc.items():
        key = '%s %s [%s beta]' % (l, nm, {'r1': 'route 1', 'r2': 'route 2', 'mine': 'mine'}[s_ if nm != 'mine_full' else 'mine'])
        ent = dict(acc)
        ent['signed_excess'] = acc['VarC'] - acc['VarP_residual']
        ent['sqrt_VarC_over_M_H'] = math.sqrt(acc['VarC']) / MH2[l] if l in MH2 else None
        ent['VarP_forms_spread_rel'] = max(rel(acc[kk], acc['VarP_residual']) for kk in ('VarP_reorth', 'VarP_unreduced', 'VarP_gram') + (('VarP_qr',) if not math.isnan(acc['VarP_qr']) else ()))
        ent['VarP_stable_forms_spread_rel'] = max(rel(acc[kk], acc['VarP_residual']) for kk in ('VarP_reorth',) + (('VarP_qr',) if not math.isnan(acc['VarP_qr']) else ()))
        nm1 = 'full' if nm == 'mine_full' else nm
        try:
          if full and l in J1['variance'] and nm1 in J1['variance'][l]['variants'] and l in J2['variance'] and nm1 in J2['variance'][l]['variants']:
              v1, v2 = J1['variance'][l]['variants'][nm1], J2['variance'][l]['variants'][nm1]
              ent['route1'] = dict(Var_C=v1['Var_C'], Var_C0=v1['Var_C0'], signed_excess=v1['signed_excess'], sqrt_VarC_over_M=v1['sqrt_VarC_over_M'])
              ent['route2'] = dict(Var_C=v2['Var_C'], Var_C0=v2['Var_C0'], signed_excess=v2['signed_excess'], sqrt_VarC_over_M=v2['sqrt_VarC_over_M'])
              ent['mine_vs_route1'] = dict(Var_C=rel(acc['VarC'], v1['Var_C']), Var_C0=rel(acc['VarP_residual'], v1['Var_C0']), signed_excess=rel(ent['signed_excess'], v1['signed_excess']),
                                           sqrt_VarC_over_M=rel(ent['sqrt_VarC_over_M_H'], v1['sqrt_VarC_over_M']), Var_C0_gram_form=rel(acc['VarP_gram'], v1['Var_C0']))
              ent['mine_vs_route2'] = dict(Var_C=rel(acc['VarC'], v2['Var_C']), Var_C0=rel(acc['VarP_residual'], v2['Var_C0']), signed_excess=rel(ent['signed_excess'], v2['signed_excess']),
                                           sqrt_VarC_over_M=rel(ent['sqrt_VarC_over_M_H'], v2['sqrt_VarC_over_M']))
              # per channel against route 1 ([kappa, w, w VarC, w VarP]) and route 2 (per copy)
              pc1 = {int(p[0]): p for p in v1['per_channel']}
              pc2 = {int(p['kappa']): p for p in v2['per_channel']}
              for p in ent['per_channel']:
                  k_ = p['kappa']
                  if k_ in pc1:
                      p['route1_VarC_percopy'] = pc1[k_][2] / pc1[k_][1]
                      p['route1_VarP_percopy'] = pc1[k_][3] / pc1[k_][1]
                      p['route1_minus_residual_rel'] = (p['route1_VarP_percopy'] - p['VarP_residual']) / p['VarP_residual']
                      p['route1_minus_mygram_rel'] = (p['route1_VarP_percopy'] - p['VarP_gram']) / p['VarP_residual']
                      p['mygram_minus_residual_rel'] = (p['VarP_gram'] - p['VarP_residual']) / p['VarP_residual']
                      p['route1_VarC_minus_mine_rel'] = (p['route1_VarC_percopy'] - p['VarC']) / p['VarC']
                  if k_ in pc2:
                      p['route2_VarC_percopy'] = pc2[k_]['VarC']
                      p['route2_VarP_percopy'] = pc2[k_]['VarP']
                      p['route2_minus_residual_rel'] = (pc2[k_]['VarP'] - p['VarP_residual']) / p['VarP_residual']
        except Exception as ex:  # noqa: BLE001
            ent['ERROR'] = repr(ex)
        var[key] = ent
        log('  %s: Var_C %.13e; Var_C0 residual %.13e reorth %.13e unreduced %.13e qr %s gram %.13e; max s %.2e; max gram-canc %.2e; route1 %s route2 %s'
            % (key, acc['VarC'], acc['VarP_residual'], acc['VarP_reorth'], acc['VarP_unreduced'], ('%.13e' % acc['VarP_qr']) if not math.isnan(acc['VarP_qr']) else '-', acc['VarP_gram'],
               acc['max_scale_unreduced'], acc['max_gram_cancellation'], ('%.13e' % ent['route1']['Var_C0']) if 'route1' in ent else '-', ('%.13e' % ent['route2']['Var_C0']) if 'route2' in ent else '-'))
    # ---------------- the comparison lines (full slice only)
    cmp_ = {}
    try:
      if full:
          e1, e2 = J1['exchange'], J2['exchange']
          for nmk in ('bare', 'dressed'):
              cmp_['E_F (%s)' % nmk] = dict(mine_T=exch[nmk]['E_F_Troute'], mine_eig=exch[nmk]['E_F_eigroute'], route1=e1[nmk]['E_F'], route2=e2[nmk]['E_F'],
                                            mine_T_vs_route1=rel(exch[nmk]['E_F_Troute'], e1[nmk]['E_F']), mine_T_vs_route2=rel(exch[nmk]['E_F_Troute'], e2[nmk]['E_F']),
                                            mine_eig_vs_route1=rel(exch[nmk]['E_F_eigroute'], e1[nmk]['E_F']) if exch[nmk]['E_F_eigroute'] is not None else None)
              cmp_['E_pair (%s)' % nmk] = dict(mine=exch[nmk]['E_pair_my_s'], route1=e1[nmk]['E_pair'], route2=e2[nmk]['E_pair'],
                                               mine_vs_route1=rel(exch[nmk]['E_pair_my_s'], e1[nmk]['E_pair']), mine_vs_route2=rel(exch[nmk]['E_pair_my_s'], e2[nmk]['E_pair']))
              cmp_['E_F/|E_pair| (%s)' % nmk] = dict(mine=exch[nmk]['E_F_over_abs_E_pair'], route1=e1[nmk]['E_F_over_abs_E_pair'], route2=e2[nmk]['E_F_over_abs_E_pair'],
                                                     mine_vs_route1=rel(exch[nmk]['E_F_over_abs_E_pair'], e1[nmk]['E_F_over_abs_E_pair']),
                                                     mine_vs_route2=rel(exch[nmk]['E_F_over_abs_E_pair'], e2[nmk]['E_F_over_abs_E_pair']))
          ki1, ki2 = J1['kernel_info'], J2['kernel']
          for q, k1, k2, mv in (('sigma_min(I + B P^A)', 'smin_I_plus_BPA', 'sigma_min_I_plus_BPA_computed', kern['sigma_min_I_plus_BPA']),
                                ('||V - B||_F/||B||_F', 'V_minus_B_rel', 'V_minus_B_rel_F', kern['V_minus_B_rel_F']),
                                ('eig B min', 'B_eig_min', 'B_eig_min', kern['B_eig_min']), ('eig B max', 'B_eig_max', 'B_eig_max', kern['B_eig_max']),
                                ('eig V min', 'V_eig_min', 'V_eig_min', kern['V_eig_min']), ('eig V max', 'V_eig_max', 'V_eig_max', kern['V_eig_max'])):
              cmp_[q] = dict(mine=mv, route1=ki1.get(k1), route2=ki2.get(k2), mine_vs_route1=rel(mv, ki1[k1]) if ki1.get(k1) is not None else None, mine_vs_route2=rel(mv, ki2[k2]))
          cmp_['N_B'] = dict(mine=NB, route1=J1['status'].get('N_B'), route2=J2['N_B'])
          cmp_['R_50'] = dict(mine=radii.get('R_50'), route1=J1['radii']['R_50']['r'], route2=J2['radii']['R_50']['r'],
                              mine_vs_route1=rel(radii['R_50'], J1['radii']['R_50']['r']), mine_vs_route2=rel(radii['R_50'], J2['radii']['R_50']['r']))
          cmp_['R_in'] = dict(mine=radii.get('R_in'), route1=J1['radii']['R_in']['r'], route2=J2['radii']['R_in']['r'])
          cmp_['R_t'] = dict(mine=radii.get('R_t'), route1=J1['radii']['R_t']['r'], route2=J2['radii']['R_t']['r'])
          cmp_['dressed/bare E_F - 1'] = dict(mine=exch['dressed_over_bare_E_F_minus_1'], route1=e1['dressed']['E_F'] / e1['bare']['E_F'] - 1.0, route2=e2['dressed']['dressed_over_bare_E_F_minus_1'])
          for nm_, d_ in cmp_.items():
              log('  CMP %s: mine %s route1 %s route2 %s | vs r1 %s vs r2 %s' % (nm_, d_.get('mine', d_.get('mine_T')), d_.get('route1'), d_.get('route2'),
                                                                                 ('%.1e' % d_['mine_vs_route1']) if d_.get('mine_vs_route1') is not None else ('%.1e' % d_['mine_T_vs_route1']) if d_.get('mine_T_vs_route1') is not None else '-',
                                                                                 ('%.1e' % d_['mine_vs_route2']) if d_.get('mine_vs_route2') is not None else ('%.1e' % d_['mine_T_vs_route2']) if d_.get('mine_T_vs_route2') is not None else '-'))
    except Exception as ex:  # noqa: BLE001
        import traceback
        cmp_['ERROR'] = traceback.format_exc()
        log('  comparison block failed: %s' % ex)
    out.update(checks=checks, n_fail=sum(c['status'] == 'FAIL' for c in checks), channels=chans, sources=src, kernel=kern, dressing=dress, exchange=exch, response=resp,
               variance=var, comparison=cmp_, vertices_rel_diff={('%s %s' % k_): v_['rel_diff'] for k_, v_ in betas.items()}, seconds=time.time() - T0)
    pj = os.path.join(a.out, 'rd_%s%s.json' % (tag, ('_' + a.tag) if a.tag else ''))
    json.dump(jsonable(out), open(pj, 'w'), indent=1)
    log('saved %s (%d FAIL); %.0f s' % (pj, out['n_fail'], out['seconds']))


if __name__ == '__main__':
    main()
