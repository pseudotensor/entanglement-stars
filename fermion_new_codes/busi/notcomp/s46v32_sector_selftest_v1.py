#!/usr/bin/env python3
"""s46v32_sector_selftest_v1.py (busi #472, the sector self-test of the state-leakage bound s46v32_stateleak_v1).

On the leak pilot's native number-conserving sectors (s46v31_leak_pilot_v1: one copy of channel kappa 1, n active modes, N = n/2,
the rest of the Hartree state frozen), with the declared v3 operators compressed exactly as the pilot does, it checks the production
formulas of s46v32_stateleak_v1 against brute force in the sector, with production (not random) one-body and two-body data:
  (a) the decomposition (E_hat_2 - E_H) psi = N_z psi + T_E psi, z = h_m - Y_E (the active block), T_E = (1/2) V :N N:;
  (b) ||N_z psi||^2 = g_z = ||z_PH||_F^2;
  (c) the exact leading part a_j^2 = ||(N_x + c_j) N_z psi||^2 from lead_parts() on the active blocks, against the sparse sector product;
  (d) the sector value X_sec = <psi|(E_hat_2 - E_H) B_sec (E_hat_2 - E_H)|psi> = 4 sum_j ||(M_hat_j - M_j) phi||^2 / M_j^2, its leading
      part X_lead,sec = 4 sum_j a_j^2 / M_j^2, and ||phi||^2_sec, so the size of the non-leading remainder is seen on real data.
Sparse only (no dense matrices). Modes: --slice HIST T --ops NPZ --n 12,14,16. Box only: nice 10, at most 8 threads, cuda:1, never cuda:0.
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
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, '..', 'route2'), '<internal>/code'):
    if p not in sys.path:
        sys.path.append(p)
import route2_core_b29_v1 as R2  # noqa: E402
from s46v3_mass2_v1 import superop  # noqa: E402
import s46v31_leak_pilot_v1 as LP  # noqa: E402
import s46v32_stateleak_v1 as SL  # noqa: E402

VERSION = 's46v32_sector_selftest_v1'
CT, RT = torch.complex128, torch.float64
T0 = time.time()
CODE_MD5 = hashlib.md5(open(os.path.abspath(__file__), 'rb').read()).hexdigest()[:8]


def log(*a):
    print('[%8.1f s] ' % (time.time() - T0) + ' '.join(str(x) for x in a), flush=True)


def onebody(sec, ob, x, Cb):
    """N_x = c^+ x c - Tr(x Cb) on the sector."""
    return LP.assemble(sec, ob, None, complex(-np.trace(x @ Cb)), x, None)


def run(a):
    dev = torch.device(a.device)
    assert a.device != 'cuda:0', 'cuda:0 is reserved'
    hist, t = a.slice[0], float(a.slice[1])
    tag = '%s_T%07.2f' % (hist, t)
    os.makedirs(a.out, exist_ok=True)
    path = R2.find_slice(hist, t)
    rec = R2.load_record(hist, t)
    Z = np.load(path, allow_pickle=False)
    ns = len(np.asarray(Z['R'], float).reshape(-1))
    Nf = np.asarray(Z['N_applied'], float).reshape(-1)
    v = np.asarray(Z['v_applied'], float).reshape(-1)
    O = np.load(a.ops)
    order = [str(x) for x in O['order'] if str(x) != 'R_out']
    if 'H_R_out' in O.files:
        Vm = 0.5 * (O['H_R_out'] + O['H_R_out'].T)
        v_src = 'H(R_out) saved'
    else:
        Vm = np.linalg.solve(np.eye(2 * ns) + O['B'] @ O['PA'], O['B'])
        Vm = 0.5 * (Vm + Vm.T)
        v_src = '(I + B P^A)^-1 B from the saved B, P^A'
    log('=== %s %s T/r_s %g; ops %s; V from %s; kappa %d; device %s ===' % (VERSION, hist, t, a.ops, v_src, a.kappa, dev))
    ch = R2.build_channel(Z, a.kappa, ns, rec['m'], Nf, v, dev, want_D_eig=False, keep_raw=False)
    Phi = ch['Phi']
    C = R2.herm(Phi @ R2.adj(Phi))
    D = C - R2.herm(ch['R2'] @ R2.adj(ch['R2']))
    hs = (ch['hb'], ch['qb'])
    Vt = torch.as_tensor(Vm, dtype=RT, device=dev)
    YEf = R2.herm(superop(hs, Vt, D, ns))
    Yf = {l: R2.herm(superop(hs, torch.as_tensor(0.5 * (O['H_' + l] + O['H_' + l].T), dtype=RT, device=dev), D, ns)) for l in order}
    results = []
    for n in a.n:
        ts = time.time()
        Nn = n // 2
        nd = ch['n_dust']
        k = Nn - nd
        assert k >= 0
        U, ainfo = LP.active_space(ch, n - Nn, k, dev)
        Aact = LP.vertex_blocks(ch, U, ns)
        At = Aact.reshape(2 * ns, n * n)
        hm = (R2.adj(U) @ ch['hmb'].left(U)).cpu().numpy()
        YE = (R2.adj(U) @ YEf @ U).cpu().numpy()
        Cb = np.diag([1.0] * Nn + [0.0] * (n - Nn))
        H_, P_ = slice(0, Nn), slice(Nn, n)
        sec = LP.Sector(n, Nn)
        ob = sec.one_body_structure()
        tb = sec.two_body_structure()
        psi_i = sec.index[sum(1 << i for i in range(Nn))]
        psi = np.zeros(sec.D, complex); psi[psi_i] = 1.0
        # E_hat_2 as the pilot builds it (linear part A_phi = h_m on the active block), and its two parts
        WV = (At.T @ (Vt.to(CT) @ At)).cpu().numpy()
        Eop = LP.build_sector_op(sec, ob, tb, WV, hm, YE, Cb, 0.0)
        EH = complex(np.vdot(psi, Eop @ psi))
        phi = Eop @ psi - EH * psi
        z = hm - YE
        TE = LP.build_sector_op(sec, ob, tb, WV, np.zeros_like(hm), np.zeros_like(hm), Cb, 0.0)
        phi1 = onebody(sec, ob, z, Cb) @ psi
        phi2 = TE @ psi
        dec = float(np.linalg.norm(phi - phi1 - phi2) / np.linalg.norm(phi))
        TE_mean = float(abs(np.vdot(psi, phi2)))
        gz = float(np.sum(np.abs(z[P_, H_]) ** 2))
        r = dict(n=n, N=Nn, D=sec.D, active=dict(n_dust=nd, k_sea=k, orthonormality=ainfo['orthonormality']),
                 decomposition_rel=dec, TE_mean_on_psi=TE_mean, E_H_imag=abs(EH.imag),
                 phi_sq=float(np.vdot(phi, phi).real), phi1_sq_brute=float(np.vdot(phi1, phi1).real), g_z=gz,
                 phi2_sq_brute=float(np.vdot(phi2, phi2).real), per_radius={})
        r['g_z_vs_brute_rel'] = abs(gz - r['phi1_sq_brute']) / r['phi1_sq_brute']
        X_sec = X_lead = 0.0
        for l in order:
            M = float(O['M_' + l])
            c = float(O['meanQ_' + l])
            H = torch.as_tensor(0.5 * (O['H_' + l] + O['H_' + l].T), dtype=RT, device=dev)
            Wt = (At.T @ (H.to(CT) @ At)).cpu().numpy()
            alin = torch.einsum('m,mab->ab', torch.as_tensor(O['beta_' + l], dtype=CT, device=dev), Aact).cpu().numpy()
            Yb = (R2.adj(U) @ Yf[l] @ U).cpu().numpy()
            Mop = LP.build_sector_op(sec, ob, tb, Wt, alin, Yb, Cb, M + c)
            y = Mop @ phi - M * phi
            full = float(np.vdot(y, y).real)
            x = alin - Yb
            yl = onebody(sec, ob, x, Cb) @ phi1 + c * phi1
            lead_brute = float(np.vdot(yl, yl).real)
            tz = lambda m_: torch.as_tensor(np.ascontiguousarray(m_), dtype=CT)  # noqa: E731
            Lw = SL.lead_parts([(1.0, tz(x[P_, H_]), tz(z[P_, H_]), tz(x[P_, P_]), tz(x[H_, H_]))], c)
            X_sec += 4 * full / M ** 2
            X_lead += 4 * Lw['total'] / M ** 2
            r['per_radius'][l] = dict(M=M, c=c, a_sq_formula=Lw['total'], a_sq_brute=lead_brute,
                                      a_sq_rel=abs(Lw['total'] - lead_brute) / max(lead_brute, 1e-300),
                                      parts=dict(zero=Lw['zero'], one=Lw['one'], two=Lw['two']),
                                      full_sq=full, full_over_lead=full / max(lead_brute, 1e-300))
        r.update(X_sec=X_sec, X_lead_sec=X_lead, X_sec_over_lead=X_sec / max(X_lead, 1e-300), X_sec_over_phi_sq=X_sec / r['phi_sq'],
                 seconds=time.time() - ts)
        log('  n %d (D %d): decomposition %.1e, g_z vs brute %.1e, <psi|T_E psi> %.1e; ||phi||^2 %.4e = %.4e (1ph) + %.4e (2ph); '
            'X_sec %.4e, X_lead %.4e (ratio %.3f); max a^2 rel %.1e; %.0f s' % (
                n, sec.D, dec, r['g_z_vs_brute_rel'], TE_mean, r['phi_sq'], r['phi1_sq_brute'], r['phi2_sq_brute'], X_sec, X_lead,
                r['X_sec_over_lead'], max(p['a_sq_rel'] for p in r['per_radius'].values()), r['seconds']))
        results.append(r)
        out = dict(script=VERSION, code_md5=CODE_MD5, slice=path, ops=a.ops, V_source=v_src, kappa=a.kappa, results=results,
                   max_a_sq_rel=max(p['a_sq_rel'] for rr in results for p in rr['per_radius'].values()),
                   max_decomposition_rel=max(rr['decomposition_rel'] for rr in results),
                   max_g_z_rel=max(rr['g_z_vs_brute_rel'] for rr in results), seconds=time.time() - T0)
        json.dump(out, open(os.path.join(a.out, '%s_sector_selftest_v1.json' % tag), 'w'), indent=1, default=str)
    log('done; %.0f s' % (time.time() - T0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slice', nargs=2, required=True)
    ap.add_argument('--ops', required=True)
    ap.add_argument('--n', type=lambda s: [int(x) for x in s.split(',')], default=[12, 14, 16])
    ap.add_argument('--kappa', type=int, default=1)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default='<internal>/s46v32')
    ap.add_argument('--threads', default='8')
    a = ap.parse_args()
    try:
        os.nice(10)
    except OSError:
        pass
    torch.set_num_threads(min(int(a.threads), 8))
    return run(a)


if __name__ == '__main__':
    sys.exit(main())
