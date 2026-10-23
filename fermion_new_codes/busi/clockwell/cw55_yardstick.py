"""cw55: #497 STEP 1, the free-phase YARDSTICK on the saved n = 5 g = 1 arms (busi). No new diagonalization.
Per form: V = H1 - H0 from the saved hamiltonian.npz (g = 1, g = 0; recheck H2 - H1 = H1 - H0 with g = 2);
psi1(t) = -i int_0^t U0(t-s) V U0(s) B ds evaluated in closed form in the saved g = 0 eigenbasis (V0, w):
  y_m(t) = -i sum_k Vt_mk c_k e^{-i w_m t} (e^{i (w_m - w_k) t} - 1) / (i (w_m - w_k))   (t e^{-i w_m t} if w_m = w_k),
  Vt = V0^dag V V0, c = V0^dag B, psi1 = V0 y; evaluated as a separable sum for |w_m - w_k| > 1e-2 (floating allowance
  8u/1e-2 ||Vt||_F ||c|| + 24 u ||Vt||_F ||c|| added to eps1) and by t e^{i dw t/2} sinc(dw t/2pi) for the near pairs.
Certified state error of psi1 (absolute): with R0 = H0 V0 - V0 W (ball, saved residual of the g = 0 arm recomputed here),
  G = ||V0 V0^dag - I||_F (ball): eps1(t) <= t (G ||V|| ||c|| + ||R0|| y_max) + ||V|| t eps0max + 2 D u ||V0||_F ||y||,
  eps0(t) = ||V0 c - B|| + t ||c|| ||R0|| + 2 D u ||V0||_F ||c||.
Wall control of psi1 (S-source Duhamel, first order in g): e1 = psi1_ext - J psi1 obeys
  i e1' = H0_ext e1 + S0 psi1 + V_ext e0 + S_V psi0,  S0 = H0_ext J - J H0 (saved g = 0 S), S_V = S1 - S0 (saved S's),
  e0 = psi0_ext - J psi0 with ||e0(s)|| <= E0(s) (the g = 0 Duhamel bound recomputed at P),
so ||e1(t)|| <= E1(t) = sum over midpoint panels of d (a_mid + c_mid) + d^2/4 (La + Lc) per panel + ||V_ext|| sum_panels d E0(right end),
  a(s) = ||S0 psi1(s)|| + C0 eps1(s), La = C0 ||V|| (3 ||H0|| + 1);  c(s) = ||S_V psi0(s)|| + ||S_V|| eps0(s), Lc = ||S_V|| ||H0||;
all operator norms by Schur (absolute row-sum) bounds. Normalized wall bound Bwall1(t) = 2 E1(t) / lower ||psi1(t)||.
P rule (existing): P = smallest power of two with the quadrature part of Bwall1 at the last clock (t = 3) <= 0.01;
allowance: Bwall1(t) <= 0.1 for t = 1, 2, 3 (the wall must control t = 3 on psi1, else VOID).
Normalizers (ball-certified positive before any division): ||V B||; ||psi1(t)||^2 at t = 1, 2, 3; N_L[psi1(t)] / ||psi1||^2
(psi1 lies in the LL sector, checked; = 2) with lower bound >= 0.05. No heavy fraction is formed for psi1 (N_H[psi1] = 0
structurally), so no heavy denominator is used. At quadrature midpoints no normalizer is used (integrands unnormalized).
f_L1(t) = <N_L on sites 1-4>[psi1] / N_L[psi1] = (light particles on sites 1-4)/2 on the LL sector, interval +- 2 eps_n1,
eps_n1 = 2 eps1 / lower ||psi1||. Verdict on I = f_L(3) - f_L1(3) (f_L(3) from the saved v1 g = 1 result): PASS (free-phase)
if I within [-0.1, 0.1]; FAIL (conversion-specific shift) if I misses [-0.1, 0.1] (sign printed); VOID otherwise or if any
prerequisite fails. Output: <internal>/cw55_yardstick.json"""
import json, os, sys, time, resource, math, hashlib
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import flint
from flint import acb_mat, acb, arb
flint.ctx.prec = 128
U = 2.0 ** -53
R = '<internal>/n5run'; OUT = f'{R}/yardstick_v1'; os.makedirs(OUT, exist_ok=True)

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def acb_from(Mx):
    return acb_mat([[acb(float(x.real), float(x.imag)) for x in row] for row in np.atleast_2d(Mx)])

def frob_upper(A):
    s = arb(0)
    for row in A.tolist():
        for e in row:
            s += e.real * e.real + e.imag * e.imag
    return float(s.sqrt().upper())

def norm2_ball(v):
    s = arb(0)
    for x in v:
        s += arb(float(x.real)) * arb(float(x.real)) + arb(float(x.imag)) * arb(float(x.imag))
    return s

def schur(A):
    return float(np.abs(A).sum(axis=1).max()) * (1 + 1e-12)

def main():
    res = {'forms': {}}
    B = M.basis(5); idx = {p: i for i, p in enumerate(B)}; D = len(B)
    Bv = np.zeros(D, dtype=complex); Bv[idx[(0, 1)]] = 1
    sp = np.array([(a % 8) // 4 + (b % 8) // 4 for a, b in B])
    nLlead = np.array([sum(1 for m in (a, b) if (m % 8) // 4 == 1 and m >= 8) for a, b in B], dtype=float)
    for form in ('onsite', 'filtered'):
        T0, W0 = time.process_time(), time.time()
        d0, d1, d2 = (f'{R}/{form}_g{g}' for g in ('0', '1', '2'))
        r0 = json.load(open(f'{d0}/result.json')); r1 = json.load(open(f'{d1}/result.json'))
        for d_, r_, fns in ((d0, r0, ('hamiltonian.npz', 'eigenbasis.npz', 'joining_source_S.npz')), (d1, r1, ('hamiltonian.npz', 'joining_source_S.npz'))):
            for fn in fns:
                assert md5(f'{d_}/{fn}') == r_['files'][fn], (d_, fn)
        H0 = np.load(f'{d0}/hamiltonian.npz')['H']; H1 = np.load(f'{d1}/hamiltonian.npz')['H']; H2 = np.load(f'{d2}/hamiltonian.npz')['H']
        E = np.load(f'{d0}/eigenbasis.npz'); w, V0 = E['w'], E['V']
        S0 = np.load(f'{d0}/joining_source_S.npz')['S']; S1 = np.load(f'{d1}/joining_source_S.npz')['S']
        V = H1 - H0; SV = S1 - S0
        lin = float(np.abs((H2 - H1) - (H1 - H0)).max())
        # sector structure: V maps HH -> LL only, H0 preserves species
        HH = sp == 0; LL = sp == 2
        struct = dict(V_from_HH_outside_LL=float(np.abs(V[np.ix_(~LL, HH)]).max()), H0_mixing=float(max(np.abs(H0[np.ix_(sp == 0, sp != 0)]).max(), np.abs(H0[np.ix_(sp == 2, sp != 2)]).max())))
        # certification constants
        R0f = frob_upper(acb_from(H0) * acb_from(V0) - acb_from(V0 * w))
        Va = acb_from(V0)
        Gf = frob_upper(Va * Va.conjugate().transpose() - acb_from(np.eye(D)))
        c = V0.conj().T @ Bv
        d0c = frob_upper(Va * acb_from(c.reshape(-1, 1)) - acb_from(Bv.reshape(-1, 1)))
        cn = float(np.linalg.norm(c)) * (1 + 1e-12); V0F = float(np.linalg.norm(V0))
        Vb = schur(V); H0b = schur(H0); C0 = math.sqrt(schur(S0.conj().T @ S0)); SVb = math.sqrt(schur(SV.conj().T @ SV))
        Hext1, Bext = M.assemble(9, 1.0, form, N=M.profile(5) + [1.0] * 4); Hext0, _ = M.assemble(9, 0.0, form, N=M.profile(5) + [1.0] * 4)
        Vext = schur(Hext1 - Hext0); del Hext1, Hext0
        VB = norm2_ball(V @ Bv); VBlo = float(VB.lower())
        Vt = V0.conj().T @ V @ V0
        dw = w[:, None] - w[None, :]
        DEL = 1e-2
        far = np.abs(dw) > DEL
        Mf = np.where(far, Vt / np.where(far, 1j * dw, 1.0), 0.0)
        Mfc = Mf @ c
        ni, nk = np.nonzero(~far)
        Vnear = Vt[ni, nk] * c[nk]; dwn = dw[ni, nk]
        afl = 8 * U / DEL * float(np.linalg.norm(Vt)) * cn + 8 * U * float(np.linalg.norm(Vt)) * cn * 3
        def psi1(t):
            # separable far part: sum_k M_mk c_k (e^{i w_m t} e^{-i w_k t} - 1); near part: t e^{i dw t/2} sinc(dw t / 2pi)
            yv = np.exp(1j * w * t) * (Mf @ (np.exp(-1j * w * t) * c)) - Mfc
            near = np.zeros(D, dtype=complex)
            np.add.at(near, ni, Vnear * t * np.exp(1j * dwn * t / 2) * np.sinc(dwn * t / (2 * np.pi)))
            y = -1j * np.exp(-1j * w * t) * (yv + near)
            return V0 @ y, y
        eps0max = d0c + 3 * cn * R0f + 2 * D * U * V0F * cn
        eps0 = lambda t: d0c + t * cn * R0f + 2 * D * U * V0F * cn
        ymax = 3 * Vb * cn * (1 + 1e-12)
        eps1 = lambda t, ynorm: t * (Gf * Vb * cn + R0f * ymax) + Vb * t * eps0max + 2 * D * U * V0F * ynorm + afl
        setup_cpu = time.process_time() - T0
        # P rule and E0, E1
        La = C0 * Vb * (3 * H0b + 1); Lc = SVb * H0b; L0 = r0['wall']['L']
        rows = {}
        for t in (1.0, 2.0, 3.0):
            p1, y = psi1(t); n2 = norm2_ball(p1); e1 = eps1(t, float(np.linalg.norm(y)))
            n_lo = math.sqrt(max(float(n2.lower()), 0.0)) - e1
            rows[t] = dict(psi1=p1, norm_lower=n_lo, norm2_ball=[float(n2.lower()), float(n2.upper())], eps1=e1)
        nlo3 = rows[3.0]['norm_lower']
        P = 64
        while 2 * (3 * (La + Lc) / (4 * P)) / nlo3 > 0.01 and P < 2 ** 16:
            P *= 2
        t1c = time.process_time()
        dd = 1.0 / P; E0 = 0.0; E1 = 0.0; pref = {}; E0pref = {}
        for k in range(3 * P):
            tm = (k + 0.5) * dd
            ph0 = V0 @ (np.exp(-1j * w * tm) * c)
            p1, y = psi1(tm)
            a = float(np.linalg.norm(S0 @ p1)) * (1 + 1e-12) + C0 * eps1(tm, float(np.linalg.norm(y)))
            cc = float(np.linalg.norm(SV @ ph0)) * (1 + 1e-12) + SVb * eps0(tm)
            f0 = float(np.linalg.norm(S0 @ ph0)) * (1 + 1e-12) + C0 * eps0(tm)
            E0 += dd * f0 + L0 * dd * dd / 4
            E1 += dd * (a + cc) + (La + Lc) * dd * dd / 4 + Vext * dd * E0
            if (k + 1) % P == 0:
                pref[(k + 1) // P] = E1 * (1 + 1e-12); E0pref[(k + 1) // P] = E0
        mid_cpu = time.process_time() - t1c
        out = dict(P=P, La=La, Lc=Lc, L0=L0, C0=C0, S_V_norm=SVb, V_norm=Vb, H0_norm=H0b, V_ext_norm=Vext, R0_frob=R0f, G_frob=Gf, d0=d0c,
                   linearity_recheck_max_abs_H2_minus_2H1_plus_H0=lin, structure=struct, VB_norm2_lower=VBlo,
                   quadrature_part_at_3=2 * (3 * (La + Lc) / (4 * P)) / nlo3, clocks=[])
        ok = VBlo > 0 and lin < 1e-12 and struct['V_from_HH_outside_LL'] == 0 and struct['H0_mixing'] == 0
        for t in (1.0, 2.0, 3.0):
            r = rows[t]; p1 = r['psi1']
            nn = float(np.real(np.vdot(p1, p1)))
            e_n = 2 * r['eps1'] / r['norm_lower'] if r['norm_lower'] > 0 else float('inf')
            fL1 = float(np.dot(np.abs(p1) ** 2, nLlead)) / (2 * nn)
            LLw = float(np.sum(np.abs(p1[LL]) ** 2)) / nn
            NLnorm = [2 * LLw - 4 * e_n, 2 * LLw + 4 * e_n]
            bw = 2 * pref[int(t)] / r['norm_lower'] if r['norm_lower'] > 0 else float('inf')
            okc = r['norm_lower'] > 0 and NLnorm[0] >= 0.05 and bw <= 0.1
            ok = ok and okc
            out['clocks'].append(dict(t=t, psi1_norm2_ball=r['norm2_ball'], psi1_norm_lower=r['norm_lower'], eps1=r['eps1'], eps_n1=e_n,
                                      N_L_over_norm2=NLnorm, f_L1=[fL1 - 2 * e_n - 1e-13, fL1 + 2 * e_n + 1e-13], E1=pref[int(t)], E0=E0pref[int(t)],
                                      Bwall1=bw, prerequisites_ok=okc))
        fL = [c_ for c_ in r1['clocks'] if c_['t'] == 3.0][0]['f_L']
        fL1_3 = out['clocks'][2]['f_L1']
        I = [fL[0] - fL1_3[1], fL[1] - fL1_3[0]]
        if not ok:
            verdict = 'VOID'
        elif I[0] >= -0.1 and I[1] <= 0.1:
            verdict = 'PASS (free-phase)'
        elif I[0] > 0.1 or I[1] < -0.1:
            verdict = 'FAIL (conversion-specific shift), sign ' + ('+' if I[0] > 0.1 else '-')
        else:
            verdict = 'VOID'
        out.update(f_L_g1_t3=fL, Delta_interval=I, prerequisites_all=ok, verdict=verdict,
                   saved_g1_NL_NH=[dict(t=c_['t'], N_L=c_['N_L'], N_H=c_['N_H']) for c_ in r1['clocks']],
                   setup_cpu_s=setup_cpu, midpoint_cpu_s=mid_cpu, midpoints=3 * P, total_cpu_s=time.process_time() - T0, wall_s=time.time() - W0)
        res['forms'][form] = out
        print(form, 'P', P, 'Bwall1', [round(x['Bwall1'], 5) for x in out['clocks']], 'f_L1(3)', fL1_3, 'I', I, verdict, 'cpu', round(out['total_cpu_s'], 1), flush=True)
    res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json.dump(res, open(f'{OUT}/cw55_yardstick.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
