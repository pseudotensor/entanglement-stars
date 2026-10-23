"""cw62: #498 (1)-(2) computations, busi. Per form at n = 5:
(1) BALL-CERTIFIED upper bound on ||T_ext||_2, T_ext = the HH -> LL block of V_ext = H_ext(g = 1) - H_ext(g = 0) on the empty
    4-site extension (n + 4 = 9 sites, heavy-pair sector of the extended chain -> light-pair sector). Sector restriction:
    e0 = psi0_ext - J psi0 lies in the heavy-pair sector (B is heavy and H0 preserves species), so V_ext e0 = T_ext e0.
    Method: G = T^dag T in ball arithmetic (128 bit); s = lambda_max_float(G) + 2 eta; certify s I - G = B B^dag + eta I + E
    with ||E||_F < eta (Weyl), eta = 1e-10; then ||T||_2 <= sqrt(s) (rounded up) + e_T, e_T = ||T_float - T_target||_F bound
    from the exact kernel deltas (filtered) plus an a priori 1e-13 ||T||_F allowance.
(2) Side-by-side first-order wall bounds on psi1 at t = 1, 2, 3, same P as executed (cw55):
    (A)      the executed bound E1_A (from cw55) with normalization 2 E / l;
    (D on A) Dunkl-Williams: 2 E1_A / (2 l - E1_A);
    (D on C) Dunkl-Williams on C: E1_C = sum d (||S0 psi1 + S_V psi0|| + C0 eps1 + ||S_V|| eps0) + quadrature (La + Lc) d^2/4
             + ||T_ext||_cert sum d E0(right end), normalized 2 E1_C / (2 l - E1_C),
    with l = lower ||psi1(t)|| (ball) and l - E > 0 certified at every clock where the normalization is used.
Output: <internal>/cw62_tnorm_dw.json"""
import json, os, sys, time, resource, math
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import cw55_yardstick as Y55
import flint
from flint import acb_mat, acb, arb
flint.ctx.prec = 128
U = 2.0 ** -53
R = '<internal>/n5run'; Y = f'{R}/yardstick_v1'
ETA = 1e-10

def tnorm_cert(T):
    G = T.conj().T @ T
    Ta = Y55.acb_from(T); Ga = Ta.conjugate().transpose() * Ta
    lam = float(np.linalg.eigvalsh((G + G.conj().T) / 2).max())
    s = lam + 2 * ETA
    Mf = s * np.eye(G.shape[0]) - G - ETA * np.eye(G.shape[0])
    w, V = np.linalg.eigh((Mf + Mf.conj().T) / 2)
    Bm = Y55.acb_from(V * np.sqrt(np.clip(w, 0, None)))
    E = Y55.acb_from(s * np.eye(G.shape[0])) - Ga - Bm * Bm.conjugate().transpose() - Y55.acb_from(ETA * np.eye(G.shape[0]))
    eF = Y55.frob_upper(E)
    ok = eF < ETA
    return math.sqrt(s) * (1 + 1e-15), ok, eF, math.sqrt(lam)

def main():
    a = json.load(open(f'{Y}/cw55_yardstick.json'))
    res = {'eta': ETA, 'forms': {}}
    for form in ('onsite', 'filtered'):
        T0 = time.process_time()
        N9 = M.profile(5) + [1.0] * 4
        H1e, Be = M.assemble(9, 1.0, form, N=N9); H0e, _ = M.assemble(9, 0.0, form, N=N9)
        Ve = H1e - H0e; del H1e, H0e
        spe = np.array([(x % 8) // 4 + (y % 8) // 4 for x, y in Be])
        HHe = spe == 0; LLe = spe == 2
        T = Ve[np.ix_(LLe, HHe)]
        other = float(np.abs(Ve[np.ix_(~LLe, HHe)]).max())
        # target error of T (kernel deltas, filtered) + a priori allowance
        eT = 1e-13 * float(np.linalg.norm(T))
        if form == 'filtered':
            Gx = {0: Fr(388627261303, 549873254400), 1: Fr(456290907571, 3115948441600), 2: Fr(180865117, 983983718400)}
            Gs = dict(M.G); dT = np.zeros_like(T)
            for r in (0, 1, 2):
                M.G.clear(); M.G.update({0: 0.0, 1: 0.0, 2: 0.0}); M.G[r] = 1.0
                Hr, _ = M.assemble(9, 1.0, form, N=N9); Hr0, _ = M.assemble(9, 0.0, form, N=N9)
                dT = dT + float(Fr(Gs[r]) - Gx[r]) * (Hr - Hr0)[np.ix_(LLe, HHe)]
            M.G.clear(); M.G.update(Gs)
            eT += float(np.linalg.norm(dT)) * (1 + 1e-6)
        tn, ok, eF, tfloat = tnorm_cert(T)
        tn_up = tn + eT
        schur_Vext = float(np.abs(Ve).sum(axis=1).max())
        cert_cpu = time.process_time() - T0
        print(form, 'T dims', T.shape, 'float ||T||_2', tfloat, 'certified upper', tn_up, 'psd cert', ok, 'Schur V_ext', schur_Vext, 'cpu', round(cert_cpu, 1), flush=True)
        # side by side
        x = a['forms'][form]; P = x['P']
        A_rows = []
        for c in x['clocks']:
            l = c['psi1_norm_lower']; E = c['E1']
            A_rows.append(dict(t=c['t'], l=l, E1_A=E, A=c['Bwall1'], l_minus_E_A=l - E, DW_A=(2 * E / (2 * l - E)) * (1 + 1e-12) if l - E > 0 else None))
        # recompute C with the combined source and the certified T norm (same P, same constants as cw55)
        T1 = time.process_time()
        C = Y55_recompute(form, P, tn_up)
        for r_, cE in zip(A_rows, C):
            l = r_['l']
            r_.update(E1_C=cE, l_minus_E_C=l - cE, DW_C=(2 * cE / (2 * l - cE)) * (1 + 1e-12) if l - cE > 0 else None)
        res['forms'][form] = dict(T_shape=list(T.shape), T_other_blocks_from_HH_max_abs=other, T_norm_float=tfloat, T_norm_certified_upper=tn_up, e_T=eT,
                                  psd_certificate=ok, E_frob_upper=eF, Schur_V_ext=schur_Vext, P=P, rows=A_rows,
                                  cert_cpu_s=cert_cpu, recompute_cpu_s=time.process_time() - T1)
        print(form, [(r_['t'], round(r_['A'], 6), r_['DW_A'] and round(r_['DW_A'], 6), r_['DW_C'] and round(r_['DW_C'], 6)) for r_ in A_rows], flush=True)
    res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; res['cpu_s_total'] = time.process_time()
    json.dump(res, open(f'{Y}/cw62_tnorm_dw.json', 'w'), indent=1)

def Y55_recompute(form, P, tnorm):
    """E1_C at t = 1, 2, 3: combined source ||S0 psi1 + S_V psi0||; all other constants and error terms as in cw55"""
    import hashlib
    B = M.basis(5); idx = {p: i for i, p in enumerate(B)}; D = len(B)
    Bv = np.zeros(D, dtype=complex); Bv[idx[(0, 1)]] = 1
    d0, d1 = f'{R}/{form}_g0', f'{R}/{form}_g1'
    H0 = np.load(f'{d0}/hamiltonian.npz')['H']; H1 = np.load(f'{d1}/hamiltonian.npz')['H']
    E = np.load(f'{d0}/eigenbasis.npz'); w, V0 = E['w'], E['V']
    S0 = np.load(f'{d0}/joining_source_S.npz')['S']; S1 = np.load(f'{d1}/joining_source_S.npz')['S']
    V = H1 - H0; SV = S1 - S0
    r0 = json.load(open(f'{d0}/result.json'))
    x = json.load(open(f'{Y}/cw55_yardstick.json'))['forms'][form]
    R0f, Gf, d0c = x['R0_frob'], x['G_frob'], x['d0']
    c = V0.conj().T @ Bv; cn = float(np.linalg.norm(c)) * (1 + 1e-12); V0F = float(np.linalg.norm(V0))
    Vb, H0b, C0, SVb = x['V_norm'], x['H0_norm'], x['C0'], x['S_V_norm']
    Vt = V0.conj().T @ V @ V0
    dw = w[:, None] - w[None, :]; DEL = 1e-2; far = np.abs(dw) > DEL
    Mf = np.where(far, Vt / np.where(far, 1j * dw, 1.0), 0.0); Mfc = Mf @ c
    ni, nk = np.nonzero(~far); Vnear = Vt[ni, nk] * c[nk]; dwn = dw[ni, nk]
    afl = 8 * U / DEL * float(np.linalg.norm(Vt)) * cn + 8 * U * float(np.linalg.norm(Vt)) * cn * 3
    def psi1(t):
        yv = np.exp(1j * w * t) * (Mf @ (np.exp(-1j * w * t) * c)) - Mfc
        near = np.zeros(D, dtype=complex)
        np.add.at(near, ni, Vnear * t * np.exp(1j * dwn * t / 2) * np.sinc(dwn * t / (2 * np.pi)))
        y = -1j * np.exp(-1j * w * t) * (yv + near)
        return V0 @ y, y
    eps0max = d0c + 3 * cn * R0f + 2 * D * U * V0F * cn
    eps0 = lambda t: d0c + t * cn * R0f + 2 * D * U * V0F * cn
    ymax = 3 * Vb * cn * (1 + 1e-12)
    eps1 = lambda t, yn: t * (Gf * Vb * cn + R0f * ymax) + Vb * t * eps0max + 2 * D * U * V0F * yn + afl
    La, Lc, L0 = x['La'], x['Lc'], x['L0']
    dd = 1.0 / P; E0 = 0.0; E1 = 0.0; pref = {}
    for k in range(3 * P):
        tm = (k + 0.5) * dd
        ph0 = V0 @ (np.exp(-1j * w * tm) * c); p1, y = psi1(tm)
        comb = float(np.linalg.norm(S0 @ p1 + SV @ ph0)) * (1 + 1e-12) + C0 * eps1(tm, float(np.linalg.norm(y))) + SVb * eps0(tm)
        f0 = float(np.linalg.norm(S0 @ ph0)) * (1 + 1e-12) + C0 * eps0(tm)
        E0 += dd * f0 + L0 * dd * dd / 4
        E1 += dd * comb + (La + Lc) * dd * dd / 4 + tnorm * dd * E0
        if (k + 1) % P == 0:
            pref[(k + 1) // P] = E1 * (1 + 1e-12)
    return [pref[1], pref[2], pref[3]]

if __name__ == '__main__':
    main()
