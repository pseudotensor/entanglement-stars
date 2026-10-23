#!/usr/bin/env python3
"""s46v31_leak_pilot_v1.py (busi, inbox #467 item 2; ReviewB_notcomp_v1 sec. 7.2, class C): the finite-sector leakage pilot
lambda_Pi^2 = || Pi E_hat_2 (I - Pi) E_hat_2 Pi || in a native number-conserving sector of n active complex modes, on ONE slice.

The sector (declared). One copy of channel kappa = 1 (the smallest gap at zero). Every other mode is frozen at its Hartree occupation:
the rest of the Hartree Slater state's occupied space is a filled core, everything orthogonal to it outside the active space is
empty. The active space is n orthonormal one-body modes:
  occupied (in Ran C): the copy's dust columns (all of them) and the k sea Ritz vectors of h_m nearest zero (in the sea span);
  empty (in Ran (1 - C)): the n_e lowest Ritz vectors of h_m in (1 - C) span{lowest positive eigenvectors of h_m}.
The sector is the N = n_dust + k particle space of the n modes (dimension binom(n, N)); the Hartree state is one basis vector.
Every operator is the compression to this sector of the paper's declared truncated operator, as built in s46v3/NOTCOMP 4.6 v3
(the native response-derived vertices, from s46v31_mass2save_v1.py's saved H_j, beta_j):
  M_hat_j = (M_j + <Q_j>) + N_{beta_j} - N_{Y_j} + (1/2) sum H_j,mu nu :N_mu N_nu:_Cbar   (j = R_in, R_50, R_t),
  E_hat_2 = the same at R_out (beta = phi, H = V = (I + B P^A)^-1 B; checked in v3 to 2.4e-9),
with N_x = c^+ a_x c - Tr(a_x Cbar), a_mu the active block of the site vertex A_mu, Y_j's active block from the full D = Cbar - P_ref of
the copy. Compression = dropping every term with a non-active index (exact for the frozen core/empty outside).
B_sec = 4 sum_j (M_hat_j/M_j - I)^2 (the compressed operators squared), Pi = 1_[0,1](B_sec) by dense eigh, and lambda_Pi = the largest
singular value of (I - Pi) E_hat_2 Pi.

Measured and printed per n: sector dimension, operator nnz, build times, sparse apply time (one matvec), dense times, peak memory,
the eigen-residuals, the Wick/compression residuals, max eig B_sec, dim Ran Pi, <B>_sec on the Hartree vector, lambda_Pi, and the scales
(the Hartree energy spread sqrt<(E_hat_2 - <E_hat_2>)^2>, the spectral half-range of E_hat_2).
Modes: --selftest (n = 6 brute-force normal ordering against the formulas, random data); --slice HIST T --ops NPZ --n 12,14,16.
Box only: nice 10, at most 8 threads, box GPU 1 (cuda:1) for dense linear algebra, never cuda:0.
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
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, '..', 'route2')):
    if p not in sys.path:
        sys.path.insert(0, p)
import route2_common as RC  # noqa: E402
import route2_core_b29_v1 as R2  # noqa: E402
from s46v3_mass2_v1 import superop  # noqa: E402

VERSION = 's46v31_leak_pilot_v1'
CT, RT = torch.complex128, torch.float64
T0 = time.time()
CODE_MD5 = hashlib.md5(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8]


def log(*a):
    print('[%8.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


# ============================================================================ the Fock sector
class Sector:
    """the N-particle states of n modes (bit a = mode a occupied), Jordan-Wigner order: c_a^+ picks up (-1)^(# occupied modes < a)."""

    def __init__(self, n, N):
        self.n, self.N = n, N
        st = [sum(1 << i for i in c) for c in itertools.combinations(range(n), N)]
        self.states = np.array(sorted(st), dtype=np.int64)
        self.D = len(self.states)
        self.index = -np.ones(1 << n, dtype=np.int64)
        self.index[self.states] = np.arange(self.D)
        self.pc = np.array([bin(i).count('1') for i in range(1 << n)], dtype=np.int64)

    def _ann(self, s, b):
        ok = ((s >> b) & 1) == 1
        sg = np.where(self.pc[s & ((1 << b) - 1)] % 2 == 0, 1, -1)
        return s ^ (1 << b), sg, ok

    def _cre(self, s, a):
        ok = ((s >> a) & 1) == 0
        sg = np.where(self.pc[s & ((1 << a) - 1)] % 2 == 0, 1, -1)
        return s | (1 << a), sg, ok

    def one_body_structure(self):
        """rows, cols, sign, pair index p = a n + b for c_a^+ c_b."""
        R, Cc, S, P = [], [], [], []
        s0 = self.states
        for a in range(self.n):
            for b in range(self.n):
                s1, g1, o1 = self._ann(s0, b)
                s2, g2, o2 = self._cre(s1, a)
                ok = o1 & o2
                R.append(self.index[s2[ok]]); Cc.append(np.nonzero(ok)[0]); S.append((g1 * g2)[ok]); P.append(np.full(int(ok.sum()), a * self.n + b))
        return np.concatenate(R), np.concatenate(Cc), np.concatenate(S), np.concatenate(P)

    def two_body_structure(self):
        """c_a^+ c_c^+ c_d c_b for a != c, b != d: rows, cols, sign and the quadruple index ((a n + b) n^2 + (c n + d))."""
        n = self.n
        R, Cc, S, Q = [], [], [], []
        s0 = self.states
        cols0 = np.arange(self.D)
        for b in range(n):
            s1, g1, o1 = self._ann(s0, b)
            for d in range(n):
                if d == b:
                    continue
                s2, g2, o2 = self._ann(s1, d)
                m = o1 & o2
                if not m.any():
                    continue
                t2, h2, c2 = s2[m], (g1 * g2)[m], cols0[m]
                for c in range(n):
                    s3, g3, o3 = self._cre(t2, c)
                    for a in range(n):
                        if a == c:
                            continue
                        s4, g4, o4 = self._cre(s3, a)
                        ok = o3 & o4
                        if not ok.any():
                            continue
                        R.append(self.index[s4[ok]]); Cc.append(c2[ok]); S.append((h2 * g3 * g4)[ok])
                        Q.append(np.full(int(ok.sum()), (a * n + b) * n * n + (c * n + d)))
        return np.concatenate(R), np.concatenate(Cc), np.concatenate(S), np.concatenate(Q)


def assemble(sec, ob, tb, c0, t, W):
    """c0 I + sum_ab t_ab c_a^+ c_b + sum_abcd W_(ab),(cd) c_a^+ c_c^+ c_d c_b as a csr matrix."""
    data = [np.full(sec.D, c0, dtype=complex)]
    rows = [np.arange(sec.D)]
    cols = [np.arange(sec.D)]
    R, Cc, S, P = ob
    data.append(S * t.reshape(-1)[P]); rows.append(R); cols.append(Cc)
    if W is not None:
        R2_, C2_, S2_, Q2_ = tb
        data.append(S2_ * W.reshape(-1)[Q2_]); rows.append(R2_); cols.append(C2_)
    return sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(sec.D, sec.D)).tocsr()


def selftest(a):
    """n = 6, N = 3: the formula's operator against the brute-force C-bar normal ordering, random Hermitian vertices and real symmetric H."""
    rng = np.random.default_rng(3)
    n, N, m = 6, 3, 5
    sec = Sector(n, N)
    ob, tb = sec.one_body_structure(), sec.two_body_structure()
    Cb = np.diag([1.0] * N + [0.0] * (n - N))
    As = []
    for _ in range(m):
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        As.append(0.5 * (X + X.conj().T))
    Hm = rng.standard_normal((m, m)); Hm = 0.5 * (Hm + Hm.T)
    # brute force: full Fock space matrices of c_a (2^n), the sector block at the end
    dim = 1 << n
    cs = []
    for k in range(n):
        Mk = np.zeros((dim, dim))
        for s in range(dim):
            if (s >> k) & 1:
                Mk[s ^ (1 << k), s] = (-1) ** bin(s & ((1 << k) - 1)).count('1')
        cs.append(Mk)
    occ = [True] * N + [False] * (n - N)

    def nord(ops):
        """ops: list of (is_creator, mode) in product order; C-bar normal order: quasi-creators left (stable), with the permutation sign."""
        qc = [((cr and not occ[md]) or ((not cr) and occ[md])) for cr, md in ops]
        left = [i for i in range(len(ops)) if qc[i]]
        right = [i for i in range(len(ops)) if not qc[i]]
        perm = left + right
        inv = sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j])
        M = np.eye(dim, dtype=complex) * (-1) ** inv
        for i in perm:
            cr, md = ops[i]
            M = M @ (cs[md].T if cr else cs[md])
        return M

    E = {(i, j): nord([(True, i), (False, j)]) for i in range(n) for j in range(n)}
    F = {}
    for i, j, k, l in itertools.product(range(n), repeat=4):
        F[(i, j, k, l)] = nord([(True, i), (False, j), (True, k), (False, l)])
    beta_blk = sum(rng.standard_normal() * A for A in As)
    Yb = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); Yb = 0.5 * (Yb + Yb.conj().T)
    const = 1.7
    full = const * np.eye(dim, dtype=complex)
    for i in range(n):
        for j in range(n):
            full += (beta_blk[i, j] - Yb[i, j]) * E[(i, j)]
    for mu in range(m):
        for nu in range(m):
            A, Bv = As[mu], As[nu]
            for i, j, k, l in itertools.product(range(n), repeat=4):
                if A[i, j] != 0 and Bv[k, l] != 0:
                    full += 0.5 * Hm[mu, nu] * A[i, j] * Bv[k, l] * F[(i, j, k, l)]
    idx = sec.states
    ref = full[np.ix_(idx, idx)]
    Wt = sum(Hm[mu, nu] * np.outer(As[mu].reshape(-1), As[nu].reshape(-1)) for mu in range(m) for nu in range(m))
    Mhat = build_sector_op(sec, ob, tb, Wt, beta_blk, Yb, Cb, const)
    err = float(np.abs(Mhat.toarray() - ref).max() / np.abs(ref).max())
    psi = sec.index[sum(1 << i for i in range(N))]
    e_mean = float(abs(Mhat[psi, psi] - const))
    out = dict(name='selftest n 6 N 3: sector operator vs brute-force C-bar normal ordering (max rel)', value=err, tol=1e-12,
               status='PASS' if err < 1e-12 else 'FAIL', mean_on_hartree_minus_const=e_mean)
    log(json.dumps(out))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'selftest_%s.json' % VERSION), 'w'), indent=1)
    return 0 if out['status'] == 'PASS' else 1


def build_sector_op(sec, ob, tb, Wt, a_lin, Y, Cb, const_mean):
    """const_mean + N_{a_lin} - N_Y + (1/2) sum H :N_mu N_nu:_Cb on the sector.
    With O(x) = c^+ x c, t_mu = Tr(a_mu Cb), S(X) = sum H a_mu X a_nu, a_Ht = sum H_mu nu t_mu a_nu:
      sum H O(a_mu) O(a_nu) = O(S(I)) + sum W c_a^+ c_c^+ c_d c_b;
      sum H N_mu N_nu = sum H O O - 2 O(a_Ht) + t^T H t;
      :N_x N_y: = N_x N_y - [O(z) - Tr(z Cb)] - Tr(Cb x (1 - Cb) y),  z = x (1 - Cb) y - y Cb x,  sum H z = S(1 - Cb) - S(Cb);
    so (1/2) sum H :N N: = (1/2) twobody(W) + O(S(Cb)) - O(a_Ht) + (1/2) t^T H t - (1/2) Tr(S(Cb) Cb) + Tr((S(1-Cb) - S(Cb)) Cb)/2
                           - (1/2) sum H Tr(Cb a_mu (1 - Cb) a_nu)  [the last two: Tr(S(1-Cb) Cb)/2 cancels the last term]."""
    n = Cb.shape[0]
    W4 = Wt.reshape(n, n, n, n)
    I = np.eye(n)
    S_of = lambda X: np.einsum('abcd,bc->ad', W4, X)  # noqa: E731
    aHt = np.einsum('abcd,ba->cd', W4, Cb)
    SC = S_of(Cb)
    tHt = np.einsum('abcd,ba,dc->', W4, Cb, Cb)
    t1 = (a_lin - Y) + SC - aHt
    c0 = const_mean - np.trace((a_lin - Y) @ Cb) + 0.5 * tHt - 0.5 * np.trace(SC @ Cb)
    return assemble(sec, ob, tb, complex(c0), t1, 0.5 * Wt)


# ============================================================================ the slice
def active_space(ch, n_e, k, dev):
    Phi, ns_ = ch['Phi'], ch['n_sea']
    hmb = ch['hmb']
    sea, dust = Phi[:, :ns_], Phi[:, ns_:]
    Hs = R2.herm(R2.adj(sea) @ hmb.left(sea))
    ev, V = torch.linalg.eigh(Hs)
    top = sea @ V[:, -k:] if k > 0 else sea[:, :0]
    occ = torch.cat([dust, top], dim=1)
    m = n_e + dust.shape[1] + 16
    Uu = ch['Uu'][:, :m]
    X = Uu - Phi @ (R2.adj(Phi) @ Uu)
    Q, Rr = torch.linalg.qr(X)
    svals = torch.linalg.svdvals(Rr)
    He = R2.herm(R2.adj(Q) @ hmb.left(Q))
    ee, Ve = torch.linalg.eigh(He)
    emp = Q @ Ve[:, :n_e]
    U = torch.cat([occ, emp], dim=1)
    info = dict(n_dust=int(dust.shape[1]), k_sea=k, n_e=n_e, sea_ritz_top=ev[-k:].cpu().numpy().tolist() if k else [],
                empty_ritz=ee[:n_e].cpu().numpy().tolist(), dust_energies=torch.real(torch.diagonal(R2.adj(dust) @ hmb.left(dust))).cpu().numpy().tolist(),
                proj_cond=float((svals.min() / svals.max()).item()),
                orthonormality=float(torch.max(torch.abs(R2.adj(U) @ U - torch.eye(U.shape[1], dtype=CT, device=dev))).item()),
                occ_in_C=float(torch.linalg.norm(occ - Phi @ (R2.adj(Phi) @ occ)).item()),
                emp_out_of_C=float(torch.linalg.norm(R2.adj(Phi) @ emp).item()))
    return U, info


def vertex_blocks(ch, U, ns):
    """a_mu = U^+ A_mu U for every site vertex A_(n,t) = (1/2){Pi_n, h_t}: (2 ns, n, n) complex."""
    out = []
    for hb in (ch['hb'], ch['qb']):
        X = hb.left(U)
        T = torch.conj(U)[:, :, None] * X[:, None, :]          # (2 ns, n, n): conj(U[i, a]) (h U)[i, b]
        T = T.reshape(ns, 2, U.shape[1], U.shape[1]).sum(dim=1)
        out.append(0.5 * (T + torch.conj(T).transpose(1, 2)))
    return torch.cat(out, dim=0)


def run(a):
    dev = torch.device(a.device)
    assert a.device != 'cuda:0', 'cuda:0 is reserved'
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t)
    os.makedirs(a.out, exist_ok=True)
    path = R2.find_slice(hist, t)
    rec = R2.load_record(hist, t)
    Z = np.load(path, allow_pickle=False)
    R = np.asarray(Z['R'], float).reshape(-1)
    ns = len(R)
    N = np.asarray(Z['N_applied'], float).reshape(-1)
    v = np.asarray(Z['v_applied'], float).reshape(-1)
    O = np.load(a.ops)
    order = [str(x) for x in O['order']]
    log('=== %s %s T/r_s %g; ops %s (%s); device %s ===' % (VERSION, hist, t, a.ops, ','.join(order), dev))
    t_ch = time.time()
    ch = R2.build_channel(Z, a.kappa, ns, rec['m'], N, v, dev, want_D_eig=False, keep_raw=False)
    t_ch = time.time() - t_ch
    log('  channel kappa %d: n_sea %d n_dust %d gap %.4f; %.0f s' % (a.kappa, ch['n_sea'], ch['n_dust'], ch['gap'], t_ch))
    Phi = ch['Phi']
    C = R2.herm(Phi @ R2.adj(Phi))
    D = C - R2.herm(ch['R2'] @ R2.adj(ch['R2']))
    phi = O['phi_app']
    results = []
    for n in a.n:
        r = dict(n=n)
        nd = ch['n_dust']
        Nn = n // 2 if a.half else nd + a.k
        k = Nn - nd
        n_e = n - Nn
        assert k >= 0 and n_e > 0
        ts = time.time()
        U, ainfo = active_space(ch, n_e, k, dev)
        Aact = vertex_blocks(ch, U, ns)                          # (2 ns, n, n)
        hm_blk = R2.adj(U) @ ch['hmb'].left(U)
        vid = float(torch.linalg.norm(torch.einsum('m,mab->ab', torch.as_tensor(phi, dtype=CT, device=dev), Aact) - hm_blk).item() / torch.linalg.norm(hm_blk).item())
        dsd = R2.site_density(ch['hb'], U[:, :1], ns)
        dchk = float(np.max(np.abs(dsd - torch.real(Aact[:ns, 0, 0]).cpu().numpy())) / max(np.max(np.abs(dsd)), 1e-300))
        Cb = np.diag([1.0] * Nn + [0.0] * n_e)
        r.update(N=Nn, active=ainfo, vertex_identity_rel=vid, diag_vs_site_density=dchk, t_active_s=time.time() - ts)
        ts = time.time()
        sec = Sector(n, Nn)
        ob = sec.one_body_structure()
        tb = sec.two_body_structure()
        r.update(D=sec.D, nnz_one_body=int(len(ob[0])), nnz_two_body_structure=int(len(tb[0])), t_structure_s=time.time() - ts)
        log('  n %d N %d: D %d, structure nnz %d + %d, %.1f s; vertex identity %.1e; active orthonormality %.1e, cond %.2e' % (
            n, Nn, sec.D, len(ob[0]), len(tb[0]), r['t_structure_s'], vid, ainfo['orthonormality'], ainfo['proj_cond']))
        At = Aact.reshape(2 * ns, n * n)
        ops, parts = {}, {}
        ts = time.time()
        for l in order:
            H = torch.as_tensor(O['H_' + l], dtype=RT, device=dev)
            Wt = (At.T @ (H.to(CT) @ At)).cpu().numpy()          # W_(ab),(cd) = sum H a_mu[ab] a_nu[cd]
            beta = O['beta_' + l]
            a_lin = torch.einsum('m,mab->ab', torch.as_tensor(beta, dtype=CT, device=dev), Aact).cpu().numpy()
            Yf = R2.herm(superop((ch['hb'], ch['qb']), H, D, ns))
            Yb = (R2.adj(U) @ Yf @ U).cpu().numpy()
            del Yf
            const = float(O['M_' + l]) + float(O['meanQ_' + l])
            op = build_sector_op(sec, ob, tb, Wt, a_lin, Yb, Cb, const)
            ops[l] = op
            psi = sec.index[sum(1 << i for i in range(Nn))]
            varL_act = float(np.real(np.trace(Cb @ a_lin @ (np.eye(n) - Cb) @ a_lin)))
            x = np.zeros(sec.D, complex); x[psi] = 1
            y = op @ x - const * x
            parts[l] = dict(M=float(O['M_' + l]), meanQ=float(O['meanQ_' + l]), mean_resid=float(abs(y[psi])), second_moment_act=float(np.vdot(y, y).real),
                            varL_act=varL_act, herm_defect=float(abs(op - op.getH()).max()), nnz=int(op.nnz),
                            beta_vs_phi_rel=(float(np.linalg.norm(beta - phi) / np.linalg.norm(phi)) if l == 'R_out' else None))
            del H
        r['t_ops_s'] = time.time() - ts
        # sparse apply time (one matvec, E_hat_2)
        x = np.random.default_rng(1).standard_normal(sec.D) + 0j
        ts = time.time()
        for _ in range(5):
            _y = ops['R_out'] @ x
        r['t_apply_sparse_s'] = (time.time() - ts) / 5
        # dense: B_sec, Pi, lambda_Pi
        ts = time.time()
        torch.cuda.reset_peak_memory_stats(dev)
        Id = torch.eye(sec.D, dtype=CT, device=dev)
        Bsec = torch.zeros((sec.D, sec.D), dtype=CT, device=dev)
        rep = [l for l in order if l != 'R_out']
        for l in rep:
            Md = torch.as_tensor(ops[l].toarray(), dtype=CT, device=dev) / parts[l]['M'] - Id
            Bsec += 4.0 * (Md @ Md)
            del Md
        Bsec = 0.5 * (Bsec + R2.adj(Bsec))
        t_b = time.time() - ts
        ts = time.time()
        lam, V = torch.linalg.eigh(Bsec)
        t_eig = time.time() - ts
        res = float(torch.max(torch.linalg.norm(Bsec @ V[:, :64] - V[:, :64] * lam[None, :64], dim=0)).item())
        res_hi = float(torch.max(torch.linalg.norm(Bsec @ V[:, -64:] - V[:, -64:] * lam[None, -64:], dim=0)).item())
        lam_np = lam.cpu().numpy()
        inP = lam_np <= 1.0
        Ed = torch.as_tensor(ops['R_out'].toarray(), dtype=CT, device=dev)
        Ed = 0.5 * (Ed + R2.adj(Ed))
        psi = sec.index[sum(1 << i for i in range(Nn))]
        e_psi = Ed[:, psi]
        EH = float(torch.real(e_psi[psi]).item())
        spread = float(torch.sqrt(torch.clamp(torch.real(torch.vdot(e_psi, e_psi)) - EH ** 2, min=0)).item())
        eE = torch.linalg.eigvalsh(Ed).cpu().numpy()
        Bpsi = float(torch.real(Bsec[psi, psi]).item())
        vpsi = V[psi, :]
        wPi = float(torch.sum(torch.abs(vpsi[torch.as_tensor(inP, device=dev)]) ** 2).item()) if inP.any() else 0.0
        if inP.all() or (~inP).all():
            lamPi = 0.0
        else:
            Vi, Vo = V[:, torch.as_tensor(inP, device=dev)], V[:, torch.as_tensor(~inP, device=dev)]
            K = R2.adj(Vo) @ (Ed @ Vi)
            lamPi = float(torch.linalg.svdvals(K)[0].item())
            del K, Vi, Vo
        t_dense = time.time() - ts + t_eig
        r.update(parts=parts, t_B_dense_s=t_b, t_eigh_s=t_eig, eigh_residual_low=res, eigh_residual_high=res_hi, B_min=float(lam_np[0]), B_max=float(lam_np[-1]),
                 B_hartree=Bpsi, dim_Pi=int(inP.sum()), hartree_weight_in_Pi=wPi, lambda_Pi=lamPi, E_hartree=EH, E_spread_on_hartree=spread,
                 E_spectrum=[float(eE[0]), float(eE[-1])], E_half_range=float(0.5 * (eE[-1] - eE[0])), E_ground_minus_hartree=float(eE[0] - EH),
                 gpu_peak_GB=torch.cuda.max_memory_allocated(dev) / 1024 ** 3, rss_peak_GB=rss_gb(), dense_bytes_one_matrix=16 * sec.D ** 2)
        log('  n %d: D %d; B_sec in [%.3e, %.3e], <B>_H %.3e, dim Pi %d/%d, Hartree weight in Pi %.6f; lambda_Pi %.3e; E_H %.6f spread %.3e half-range %.3e; '
            'eigh %.1f s (res %.1e/%.1e); sparse apply %.2e s; GPU peak %.2f GB; RSS %.2f GB' % (
                n, sec.D, lam_np[0], lam_np[-1], Bpsi, int(inP.sum()), sec.D, wPi, lamPi, EH, spread, r['E_half_range'], t_eig, res, res_hi,
                r['t_apply_sparse_s'], r['gpu_peak_GB'], r['rss_peak_GB']))
        for l in order:
            log('     %s: mean resid %.1e, <(M_hat - M - <Q>)^2>_act %.4e (Var L_act %.4e), herm %.1e, nnz %d%s' % (
                l, parts[l]['mean_resid'], parts[l]['second_moment_act'], parts[l]['varL_act'], parts[l]['herm_defect'], parts[l]['nnz'],
                ('' if parts[l]['beta_vs_phi_rel'] is None else '; |beta - phi|/|phi| %.1e' % parts[l]['beta_vs_phi_rel'])))
        del Bsec, V, Ed, Id
        torch.cuda.empty_cache()
        results.append(r)
        json.dump(dict(script=VERSION, code_md5=CODE_MD5, slice=path, ops=a.ops, kappa=a.kappa, results=results, args=vars(a), seconds=time.time() - T0),
                  open(os.path.join(a.out, '%s_leak_pilot_v1.json' % tag), 'w'), indent=1, default=str)
    log('done; %.0f s' % (time.time() - T0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--slice', nargs=2)
    ap.add_argument('--ops')
    ap.add_argument('--n', type=lambda s: [int(x) for x in s.split(',')], default=[12, 14, 16])
    ap.add_argument('--kappa', type=int, default=1)
    ap.add_argument('--half', action='store_true', help='N = n/2 (default)')
    ap.add_argument('--k', type=int, default=0)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default='<internal>/leak')
    ap.add_argument('--threads', default='8')
    a = ap.parse_args()
    if not a.half and a.k == 0:
        a.half = True
    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(min(int(a.threads), 8))
    if a.selftest:
        return selftest(a)
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
