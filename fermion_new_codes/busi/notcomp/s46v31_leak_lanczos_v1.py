#!/usr/bin/env python3
"""s46v31_leak_lanczos_v1.py (busi, inbox #469: leakage option (b), box only): the largest eigenvalue of the finite-sector branch operator
B_sec = 4 sum_j (M_hat_j/M_j - I)^2 by Lanczos (scipy eigsh on the sparse sector operators), in larger native number-conserving sectors
than the dense pilot (s46v31_leak_pilot_v1.py, whose operator construction, active space and conventions this reuses unchanged):
lambda_Pi = 0 exactly whenever B_max <= 1 (then Pi = 1_[0,1](B_sec) is the whole sector), so no dense step is needed.

Sectors (each a product of blocks; every block is one copy of one channel kappa, its modes chosen as in the pilot: the copy's dust columns
and the top sea Ritz vectors occupied, the lowest h_m Ritz vectors of (1 - C) empty; occupation N_b = max(n_b // 2, n_dust) per block; the
operators conserve each block's number, so the sector fixes every N_b):
  single block, kappa 1: n = 16 (the dense pilot's check), 18, 20;
  two copies of kappa 1: 10 + 10 modes;  two channels, kappa 1 and kappa 2: 10 + 10 modes.
Operators: the vertices are block diagonal (sum over copies and channels with unit weight per copy), Y_j block diagonal from each block's own
D = Cbar - P_ref, the two-body term antisymmetrized over (a < c, b < d) and assembled in chunks (memory). Constants as in the pilot.
--selftest: on n = 10 random data, the chunked antisymmetrized assembly against the pilot's build_sector_op (max rel), and Lanczos against dense.
Box only: nice 10, at most 8 threads, GPU 1 (cuda:1) for the 4096-dimensional one-body steps, never cuda:0; sparse work on the CPU.
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
import resource  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import s46v31_leak_pilot_v1 as LP  # noqa: E402
from s46v31_leak_pilot_v1 import R2, superop  # noqa: E402

VERSION = 's46v31_leak_lanczos_v1'
CT, RT = torch.complex128, torch.float64
T0 = time.time()
CODE_MD5 = hashlib.md5(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8]


def log(*a):
    print('[%8.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2


class BlockSector(LP.Sector):
    """the states with exactly N_b particles in each block of modes (blocks consecutive)."""

    def __init__(self, blocks):
        n = sum(nb for nb, _ in blocks)
        self.n, self.N = n, sum(Nb for _, Nb in blocks)
        per, off = [], 0
        for nb, Nb in blocks:
            per.append([sum(1 << (off + i) for i in c) for c in itertools.combinations(range(nb), Nb)])
            off += nb
        st = [sum(x) for x in itertools.product(*per)]
        self.states = np.array(sorted(st), dtype=np.int64)
        self.D = len(self.states)
        self.index = -np.ones(1 << n, dtype=np.int64)
        self.index[self.states] = np.arange(self.D)
        self.pc = np.array([bin(i).count('1') for i in range(1 << n)], dtype=np.int64)


def coefficients(Wt, a_lin, Y, Cb, const_mean):
    """the pilot's build_sector_op constants and one-body block, and the antisymmetrized two-body coefficient V[a, b, c, d] for
    c_a^+ c_c^+ c_d c_b, a < c, b < d (from (1/2) W)."""
    n = Cb.shape[0]
    W4 = Wt.reshape(n, n, n, n)
    S_of = lambda X: np.einsum('abcd,bc->ad', W4, X)  # noqa: E731
    aHt = np.einsum('abcd,ba->cd', W4, Cb)
    SC = S_of(Cb)
    tHt = np.einsum('abcd,ba,dc->', W4, Cb, Cb)
    t1 = (a_lin - Y) + SC - aHt
    c0 = const_mean - np.trace((a_lin - Y) @ Cb) + 0.5 * tHt - 0.5 * np.trace(SC @ Cb)
    V = 0.5 * (W4 - W4.transpose(2, 1, 0, 3) - W4.transpose(0, 3, 2, 1) + W4.transpose(2, 3, 0, 1))
    return complex(c0), t1, V


def assemble_many(sec, coefs, log_every=True):
    """csr operators for a list of (c0, t1, V), sharing the structure; the two-body part chunked by the first annihilated mode b."""
    n, D = sec.n, sec.D
    R, Cc, S, P = sec.one_body_structure()
    keep = R >= 0                                        # (block sectors: a hop across blocks leaves the sector)
    R, Cc, S, P = R[keep], Cc[keep], S[keep], P[keep]
    mats = []
    for c0, t1, V in coefs:
        A = sp.coo_matrix((np.concatenate([np.full(D, c0, dtype=complex), S * t1.reshape(-1)[P]]),
                           (np.concatenate([np.arange(D), R]), np.concatenate([np.arange(D), Cc]))), shape=(D, D)).tocsr()
        A.eliminate_zeros()
        mats.append(A)
    del R, Cc, S, P
    s0 = sec.states
    cols0 = np.arange(D)
    Vf = [V.reshape(-1) for _, _, V in coefs]
    n_entries = 0
    for b in range(n):
        s1, g1, o1 = sec._ann(s0, b)
        rows, cols, sg, qq = [], [], [], []
        for d in range(b + 1, n):
            s2, g2, o2 = sec._ann(s1, d)
            m = o1 & o2
            if not m.any():
                continue
            t2, h2, c2 = s2[m], (g1 * g2)[m], cols0[m]
            for c in range(n):
                s3, g3, o3 = sec._cre(t2, c)
                for a in range(c):
                    s4, g4, o4 = sec._cre(s3, a)
                    ok = o3 & o4
                    if not ok.any():
                        continue
                    idx = sec.index[s4[ok]]
                    good = idx >= 0                      # (block sectors: a move across blocks leaves the sector)
                    if not good.all():
                        okn = np.nonzero(ok)[0][good]
                        idx = idx[good]
                    else:
                        okn = np.nonzero(ok)[0]
                    if len(okn) == 0:
                        continue
                    rows.append(idx); cols.append(c2[okn]); sg.append((h2 * g3 * g4)[okn])
                    qq.append(np.full(len(okn), ((a * n + b) * n + c) * n + d))
        if not rows:
            continue
        rows, cols, sg, qq = np.concatenate(rows), np.concatenate(cols), np.concatenate(sg), np.concatenate(qq)
        n_entries += len(rows)
        for i, vf in enumerate(Vf):
            X = sp.coo_matrix((sg * vf[qq], (rows, cols)), shape=(D, D)).tocsr()
            X.eliminate_zeros()
            mats[i] = mats[i] + X
            del X
        del rows, cols, sg, qq
    return mats, n_entries


def lanczos_B(Xs, D, psi, tol=1e-10):
    """largest eigenvalue of B = 4 sum X_j^2 (X_j = M_hat_j/M_j - I, Hermitian csr) by eigsh; its residual; <psi|B|psi>."""
    nmv = [0]

    def mv(v):
        nmv[0] += 1
        v = np.asarray(v).reshape(-1)
        out = np.zeros(D, complex)
        for X in Xs:
            out += 4.0 * (X @ (X @ v))
        return out
    Lop = spl.LinearOperator((D, D), matvec=mv, dtype=complex)
    t = time.time()
    v0 = np.random.default_rng(5).standard_normal(D) + 0j
    w, V = spl.eigsh(Lop, k=1, which='LA', tol=tol, v0=v0, ncv=min(40, D - 1), maxiter=20000)
    t = time.time() - t
    v = V[:, 0]
    res = float(np.linalg.norm(mv(v) - w[0] * v))
    x = np.zeros(D, complex); x[psi] = 1
    Bpsi = float(sum(4.0 * np.vdot(X @ x, X @ x).real for X in Xs))
    return float(w[0]), res, Bpsi, t, nmv[0]


def selftest(a):
    rng = np.random.default_rng(7)
    n, N, m = 10, 5, 6
    sec = LP.Sector(n, N)
    ob, tb = sec.one_body_structure(), sec.two_body_structure()
    Cb = np.diag([1.0] * N + [0.0] * (n - N))
    As = []
    for _ in range(m):
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        As.append(0.5 * (X + X.conj().T))
    Hm = rng.standard_normal((m, m)); Hm = 0.5 * (Hm + Hm.T)
    Wt = sum(Hm[i, j] * np.outer(As[i].reshape(-1), As[j].reshape(-1)) for i in range(m) for j in range(m))
    al = sum(rng.standard_normal() * A for A in As)
    Yb = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); Yb = 0.5 * (Yb + Yb.conj().T)
    ref = LP.build_sector_op(sec, ob, tb, Wt, al, Yb, Cb, 2.3).toarray()
    mats, _ = assemble_many(sec, [coefficients(Wt, al, Yb, Cb, 2.3)])
    e1 = float(np.abs(mats[0].toarray() - ref).max() / np.abs(ref).max())
    X = (mats[0] - 2.3 * sp.identity(sec.D, format='csr')) * 0.1
    X = 0.5 * (X + X.getH())
    Bd = 4.0 * (X @ X).toarray()
    wd = float(np.linalg.eigvalsh(Bd)[-1])
    wl, res, _, _, _ = lanczos_B([X.tocsr()], sec.D, 0)
    e2 = abs(wl - wd) / abs(wd)
    out = [dict(name='chunked antisymmetrized assembly vs the pilot build_sector_op (n 10, max rel)', value=e1, tol=1e-12, status='PASS' if e1 < 1e-12 else 'FAIL'),
           dict(name='Lanczos max eig vs dense (n 10)', value=e2, tol=1e-9, status='PASS' if e2 < 1e-9 else 'FAIL', residual=res)]
    # the block sector: a 2-block random check that BlockSector states are those of the full sector with the block numbers
    bs = BlockSector([(4, 2), (4, 2)])
    full = LP.Sector(8, 4)
    cnt = sum(1 for s in full.states if bin(s & 0xF).count('1') == 2)
    out.append(dict(name='BlockSector state count vs filter (4+4, 2+2)', value=float(abs(bs.D - cnt)), tol=0.0, status='PASS' if bs.D == cnt else 'FAIL'))
    for o in out:
        log(json.dumps(o))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, 'selftest_%s.json' % VERSION), 'w'), indent=1)
    return 0 if all(o['status'] == 'PASS' for o in out) else 1


def block_space(ch, n_b, dev):
    nd = ch['n_dust']
    Nb = max(n_b // 2, nd)
    k = Nb - nd
    n_e = n_b - Nb
    U, info = LP.active_space(ch, n_e, k, dev)
    return U, Nb, info


def run(a):
    dev = torch.device(a.device)
    assert a.device != 'cuda:0', 'cuda:0 is reserved'
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t)
    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, '%s_%s.json' % (tag, VERSION))
    path = R2.find_slice(hist, t)
    rec = R2.load_record(hist, t)
    Z = np.load(path, allow_pickle=False)
    R = np.asarray(Z['R'], float).reshape(-1)
    ns = len(R)
    Nf = np.asarray(Z['N_applied'], float).reshape(-1)
    v = np.asarray(Z['v_applied'], float).reshape(-1)
    O = np.load(a.ops)
    order = [str(x) for x in O['order'] if str(x) != 'R_out']
    log('=== %s %s T/r_s %g; ops %s (%s); device %s ===' % (VERSION, hist, t, a.ops, ','.join(order), dev))
    chans = {}
    for kap in sorted(set([1] + ([2] if 'chan2' in a.sectors else []))):
        ch = R2.build_channel(Z, kap, ns, rec['m'], Nf, v, dev, want_D_eig=False, keep_raw=False)
        Phi = ch['Phi']
        ch['Dfull'] = R2.herm(Phi @ R2.adj(Phi)) - R2.herm(ch['R2'] @ R2.adj(ch['R2']))
        chans[kap] = ch
        log('  channel kappa %d: n_sea %d n_dust %d gap %.4f' % (kap, ch['n_sea'], ch['n_dust'], ch['gap']))
    specs = []
    for s in a.sectors:
        if s.startswith('single'):
            specs.append((s, [(1, int(s[6:]))]))
        elif s == 'copy2':
            specs.append((s, [(1, 10), (1, 10)]))
        elif s == 'chan2':
            specs.append((s, [(1, 10), (2, 10)]))
    results = json.load(open(pj))['results'] if (os.path.exists(pj) and not a.force) else []
    done = {r['sector'] for r in results}
    for name, blocks in specs:
        if name in done:
            log('  %s done; skipped' % name)
            continue
        ts = time.time()
        Us, Nbs, infos = [], [], []
        for kap, nb in blocks:
            U, Nb, inf = block_space(chans[kap], nb, dev)
            Us.append((kap, U)); Nbs.append(Nb); infos.append(dict(kappa=kap, n_b=nb, N_b=Nb, **inf))
        n = sum(nb for _, nb in blocks)
        offs = np.cumsum([0] + [nb for _, nb in blocks])
        Ablk = [LP.vertex_blocks(chans[kap], U, ns) for kap, U in Us]
        Aact = torch.zeros((2 * ns, n, n), dtype=CT, device=dev)
        for i, A in enumerate(Ablk):
            Aact[:, offs[i]:offs[i + 1], offs[i]:offs[i + 1]] = A
        Cb = np.diag(np.concatenate([[1.0] * Nb + [0.0] * (nb - Nb) for (_, nb), Nb in zip(blocks, Nbs)]))
        sec = BlockSector([(nb, Nb) for (_, nb), Nb in zip(blocks, Nbs)])
        psi = sec.index[int(sum(sum(1 << (offs[i] + j) for j in range(Nb)) for i, Nb in enumerate(Nbs)))]
        At = Aact.reshape(2 * ns, n * n)
        coefs, parts = [], {}
        for l in order:
            H = torch.as_tensor(O['H_' + l], dtype=RT, device=dev)
            Wt = (At.T @ (H.to(CT) @ At)).cpu().numpy()
            beta = O['beta_' + l]
            a_lin = torch.einsum('m,mab->ab', torch.as_tensor(beta, dtype=CT, device=dev), Aact).cpu().numpy()
            Yb = np.zeros((n, n), complex)
            for i, (kap, U) in enumerate(Us):
                Yf = R2.herm(superop((chans[kap]['hb'], chans[kap]['qb']), H, chans[kap]['Dfull'], ns))
                Yb[offs[i]:offs[i + 1], offs[i]:offs[i + 1]] = (R2.adj(U) @ Yf @ U).cpu().numpy()
                del Yf
            const = float(O['M_' + l]) + float(O['meanQ_' + l])
            coefs.append(coefficients(Wt, a_lin, Yb, Cb, const))
            parts[l] = dict(M=float(O['M_' + l]), meanQ=float(O['meanQ_' + l]), const=const,
                            varL_act=float(np.real(np.trace(Cb @ a_lin @ (np.eye(n) - Cb) @ a_lin))))
            del H
        t_coef = time.time() - ts
        ts = time.time()
        mats, n_entries = assemble_many(sec, coefs)
        t_asm = time.time() - ts
        Xs = []
        for l, Mt in zip(order, mats):
            X = Mt / parts[l]['M'] - sp.identity(sec.D, format='csr', dtype=complex)
            hd = float(abs(X - X.getH()).max()) if sec.D <= 60000 else None
            X = (0.5 * (X + X.getH())).tocsr()
            x = np.zeros(sec.D, complex); x[psi] = 1
            y = Mt @ x
            parts[l].update(nnz=int(Mt.nnz), herm_defect=hd, mean_resid=float(abs(y[psi] - parts[l]['const'])),
                            second_moment_act=float(np.vdot(y - parts[l]['const'] * x, y - parts[l]['const'] * x).real))
            Xs.append(X)
        del mats
        xr = np.random.default_rng(2).standard_normal(sec.D) + 0j
        t1_ = time.time()
        _ = Xs[0] @ xr
        t_apply = time.time() - t1_
        Bmax, res, Bpsi, t_lz, nmv = lanczos_B(Xs, sec.D, psi, a.tol)
        r = dict(sector=name, blocks=infos, n=n, N=int(sum(Nbs)), D=sec.D, two_body_entries=int(n_entries), nnz=[parts[l]['nnz'] for l in order],
                 B_max=Bmax, lanczos_residual=res, B_hartree=Bpsi, lambda_Pi=(0.0 if Bmax <= 1.0 else None), Pi_is_whole_sector=bool(Bmax <= 1.0),
                 t_coef_s=t_coef, t_assemble_s=t_asm, t_apply_one_op_s=t_apply, t_lanczos_s=t_lz, n_matvec=nmv, rss_peak_GB=rss_gb(), parts=parts)
        log('  %s: n %d N %d D %d; entries %d; nnz %s; B_max %.4e (residual %.1e, %d matvecs, %.0f s); <B>_H %.3e; assemble %.0f s; apply %.2e s; RSS %.1f GB' % (
            name, n, r['N'], sec.D, n_entries, r['nnz'], Bmax, res, nmv, t_lz, Bpsi, t_asm, t_apply, r['rss_peak_GB']))
        for l in order:
            log('     %s: mean resid %.1e, second moment act %.4e (Var L_act %.4e), herm %s' % (
                l, parts[l]['mean_resid'], parts[l]['second_moment_act'], parts[l]['varL_act'], parts[l]['herm_defect']))
        results.append(r)
        del Xs
        json.dump(dict(script=VERSION, code_md5=CODE_MD5, pilot_md5=LP.CODE_MD5, slice=path, ops=a.ops, results=results, args=vars(a), seconds=time.time() - T0),
                  open(pj, 'w'), indent=1, default=str)
    log('done; %.0f s' % (time.time() - T0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--slice', nargs=2)
    ap.add_argument('--ops')
    ap.add_argument('--sectors', type=lambda s: s.split(','), default=['single16', 'single18', 'single20', 'copy2', 'chan2'])
    ap.add_argument('--tol', type=float, default=1e-8)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default='<internal>/lanczos')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--threads', default='8')
    a = ap.parse_args()
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
