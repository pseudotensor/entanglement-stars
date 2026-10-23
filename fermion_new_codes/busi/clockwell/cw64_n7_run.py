"""cw64: the n = 7 transport-only test, ONE SHOT per form, exactly per deliveries/lead_gaps/MOCK_n7_rule_agreed_v1.md (#500).
Usage: cw64_n7_run.py FORM CPU_ALLOWANCE_S   (FORM onsite|filtered). Output (small): <internal>/n7run<form>/
Steps (busi builder cw32; no icloud code):
 0. H0, H1, H2 at n = 7 (complete N = 2 sector, D = 1540); linearity max|H2 - 2 H1 + H0|; e_H target errors.
 1. eigendecompositions of H0 and H1; eigen-residuals ||H V - V Lambda||_F ball-certified (128 bit), must be <= 1e-10.
 2. g = 1 pure B: psi(t) = V1 e^{-i w1 t} V1^dag B, t = 1..4, eps(t) as in cw45; N2 ball-certified positive; lower
    N_L(t)/||psi||^2 >= 0.05 at t = 1..4; f_L(t) = <N_L on sites 1-6>/<N_L> after the N_L lower bound is certified positive;
    p_LL(3) printed unjudged; no heavy normalizer is formed (no f_H), so none is required.
 3. psi1(t) (cw55 closed form in the g = 0 eigenbasis), t = 1..4: ||VB|| > 0, ||psi1(t)||^2 > 0 (ball), lower N_L/||psi1||^2
    >= 0.05; f_L1(t) = (light particles on sites 1-6)/2 on the LL-only psi1.
 4. Walls at t = 4 (n + 4 = 11-site empty extension): (a) g = 1 S-source Duhamel bound E for pure B as frozen (plan v2), and
    its Dunkl-Williams form 2E/(2l - E) with l = lower ||psi(4)||, l - E > 0 certified; both printed, the verdict uses the
    Dunkl-Williams value (the rule's literal normalization; it is the larger); (b) first-order bound on psi1 by construction C
    (combined source, ball-certified upper bound on the HH -> LL block norm of V_ext at 11 sites), Dunkl-Williams 2E1/(2l1 - E1),
    l1 - E1 > 0 certified. P per wall = the first power of two from 64 with quadrature share <= 0.01 at t = 4 (share of (a):
    4 L/(4P); of (b): 2 (4 (La + Lc)/(4P)) / l1(4)), SET AT RUN START before any midpoint is evaluated.
 5. Budget: after setup, the projected midpoint cost (measured per-midpoint CPU on 8 test midpoints x the midpoint count)
    must fit the remaining allowance; else STOP, VOID, no reduced-P fallback.
 6. Verdict: I = f_L(4) - f_L1(4); PASS (free-phase) if I inside [-0.1, 0.1]; FAIL (signed) if disjoint; VOID otherwise or
    on any prerequisite failure. 1/2 printed alongside, unjudged.
Retained: result.json and the 8 state vectors (psi(t), psi1(t), t = 1..4) as one npz (about 0.2 MB)."""
import json, os, sys, time, resource, math, hashlib
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import cw45_n5_run as K45
import cw62_tnorm_dw as C62
import flint
from flint import acb_mat, acb, arb
flint.ctx.prec = 128
U = 2.0 ** -53

def schur(A):
    return float(np.abs(A).sum(axis=1).max()) * (1 + 1e-12)

def main():
    form = sys.argv[1]; allowance = float(sys.argv[2]); n = 7
    OUT = f'<internal>/{form}'; os.makedirs(OUT, exist_ok=True)
    T0 = time.process_time(); W0 = time.time(); cost = []; res = dict(n=n, form=form, rule='deliveries/lead_gaps/MOCK_n7_rule_agreed_v1.md', allowance_cpu_s=allowance)
    def mark(name, t0):
        cost.append(dict(stage=name, cpu_s=time.process_time() - t0, cum_cpu_s=time.process_time() - T0, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        res['cost'] = cost; json.dump(res, open(f'{OUT}/result.json', 'w'), indent=1, default=float)
        print(f"{name:40s} cpu {cost[-1]['cpu_s']:8.1f} cum {cost[-1]['cum_cpu_s']:8.1f}", flush=True)
    def void(reason):
        res['verdict'] = 'VOID'; res['void_reason'] = reason
        json.dump(res, open(f'{OUT}/result.json', 'w'), indent=1, default=float); print('VOID', reason, flush=True); sys.exit(0)
    # 0
    t0 = time.process_time()
    H0, B = M.assemble(n, 0.0, form); H1, _ = M.assemble(n, 1.0, form); H2, _ = M.assemble(n, 2.0, form)
    D = len(B); lin = float(np.abs(H2 - 2 * H1 + H0).max()); del H2
    eH0 = K45.target_error(n, 0.0, form, H0); eH1 = K45.target_error(n, 1.0, form, H1)
    V = H1 - H0
    res.update(dim=D, linearity=lin, e_H0=eH0, e_H1=eH1)
    mark('assembly_linearity_target_errors', t0)
    if lin > 1e-12:
        void('linearity')
    # 1
    t0 = time.process_time()
    w0, V0, R0 = K45.cert_eig(H0); w1, V1, R1 = K45.cert_eig(H1)
    res.update(residual_H0=R0, residual_H1=R1)
    mark('eigendecompositions_and_ball_residuals', t0)
    if R0 > 1e-10 or R1 > 1e-10:
        void('residual above 1e-10')
    idx = {p: i for i, p in enumerate(B)}
    Bv = np.zeros(D, dtype=complex); Bv[idx[(0, 1)]] = 1
    sp = np.array([(a % 8) // 4 + (b % 8) // 4 for a, b in B]); LL = sp == 2
    nL = sp.astype(float)
    nLlead = np.array([sum(1 for m in (a, b) if (m % 8) // 4 == 1 and m >= 8) for a, b in B], dtype=float)
    # 2: g = 1 pure B
    t0 = time.process_time()
    Va1 = K45.acb_from(V1)
    c1 = V1.conj().T @ Bv
    d01 = K45.frob_upper(Va1 * K45.acb_from(c1.reshape(-1, 1)) - K45.acb_from(Bv.reshape(-1, 1)))
    cn1 = float(np.linalg.norm(c1)) * (1 + 1e-12); fl1 = 2 * D * U * float(np.linalg.norm(V1)) * cn1
    eps = lambda t: d01 + t * cn1 * (R1 + eH1) + fl1
    g1rows = []; keep = {}
    for t in (1.0, 2.0, 3.0, 4.0):
        phi = V1 @ (np.exp(-1j * w1 * t) * c1); keep[f'psi_g1_t{t:g}'] = phi
        N2 = K45.norm2_ball(phi); lo2, hi2 = float(N2.lower()), float(N2.upper())
        if not lo2 > 0:
            void(f'N2 not certified positive at t = {t}')
        en = 2 * eps(t) / math.sqrt(lo2)
        p = np.abs(phi) ** 2 * 2 / (lo2 + hi2)
        NL = float(np.dot(p, nL)); NLl = float(np.dot(p, nLlead)); pLL = float(p[LL].sum())
        NLi = [NL - 4 * en - 1e-13, NL + 4 * en + 1e-13]; NLli = [NLl - 4 * en - 1e-13, NLl + 4 * en + 1e-13]
        row = dict(t=t, N2=[lo2, hi2], eps=eps(t), eps_n=en, N_L=NLi, N_L_lead=NLli, p_LL=[pLL - 2 * en - 1e-13, pLL + 2 * en + 1e-13], l_lower=math.sqrt(lo2) - eps(t))
        row['N_L_prereq'] = NLi[0] >= 0.05
        if NLi[0] > 0:
            row['f_L'] = [max(0.0, NLli[0]) / NLi[1], NLli[1] / NLi[0]]
        g1rows.append(row)
    res['g1'] = g1rows
    mark('g1_states_admission_readouts', t0)
    # 3: psi1
    t0 = time.process_time()
    Va0 = K45.acb_from(V0)
    Gf = K45.frob_upper(Va0 * Va0.conjugate().transpose() - K45.acb_from(np.eye(D)))
    c0 = V0.conj().T @ Bv
    d00 = K45.frob_upper(Va0 * K45.acb_from(c0.reshape(-1, 1)) - K45.acb_from(Bv.reshape(-1, 1)))
    cn0 = float(np.linalg.norm(c0)) * (1 + 1e-12); V0F = float(np.linalg.norm(V0))
    Vb = schur(V); H0b = schur(H0)
    VB = K45.norm2_ball(V @ Bv); VBlo = float(VB.lower())
    if not VBlo > 0:
        void('||VB|| not certified positive')
    Vt = V0.conj().T @ V @ V0
    dw = w0[:, None] - w0[None, :]; DEL = 1e-2; far = np.abs(dw) > DEL
    Mf = np.where(far, Vt / np.where(far, 1j * dw, 1.0), 0.0); Mfc = Mf @ c0
    ni, nk = np.nonzero(~far); Vnear = Vt[ni, nk] * c0[nk]; dwn = dw[ni, nk]
    afl = 8 * U / DEL * float(np.linalg.norm(Vt)) * cn0 + 8 * U * float(np.linalg.norm(Vt)) * cn0 * 4
    del Vt
    def psi1(t):
        yv = np.exp(1j * w0 * t) * (Mf @ (np.exp(-1j * w0 * t) * c0)) - Mfc
        near = np.zeros(D, dtype=complex)
        np.add.at(near, ni, Vnear * t * np.exp(1j * dwn * t / 2) * np.sinc(dwn * t / (2 * np.pi)))
        y = -1j * np.exp(-1j * w0 * t) * (yv + near)
        return V0 @ y, y
    eps0max = d00 + 4 * cn0 * R0 + 2 * D * U * V0F * cn0
    eps0 = lambda t: d00 + t * cn0 * R0 + 2 * D * U * V0F * cn0
    ymax = 4 * Vb * cn0 * (1 + 1e-12)
    eps1 = lambda t, yn: t * (Gf * Vb * cn0 + R0 * ymax) + Vb * t * eps0max + 2 * D * U * V0F * yn + afl + t * eH0 * ymax
    p1rows = []
    for t in (1.0, 2.0, 3.0, 4.0):
        p1, y = psi1(t); keep[f'psi1_t{t:g}'] = p1
        n2 = K45.norm2_ball(p1); e1 = eps1(t, float(np.linalg.norm(y)))
        l1 = math.sqrt(max(float(n2.lower()), 0.0)) - e1
        if not l1 > 0:
            void(f'||psi1|| not certified positive at t = {t}')
        nn = float(np.real(np.vdot(p1, p1))); en1 = 2 * e1 / l1
        LLw = float(np.sum(np.abs(p1[LL]) ** 2)) / nn
        NLn = [2 * LLw - 4 * en1, 2 * LLw + 4 * en1]
        fL1 = float(np.dot(np.abs(p1) ** 2, nLlead)) / (2 * nn)
        p1rows.append(dict(t=t, norm2=[float(n2.lower()), float(n2.upper())], eps1=e1, l1=l1, eps_n1=en1, N_L_over_norm2=NLn, N_L_prereq=NLn[0] >= 0.05,
                           f_L1=[fL1 - 2 * en1 - 1e-13, fL1 + 2 * en1 + 1e-13], weight_outside_LL=1 - LLw))
    res['psi1'] = p1rows; res['VB_norm2_lower'] = VBlo
    mark('psi1_states_admission_readouts', t0)
    # 4: walls, setup
    t0 = time.process_time()
    Next = M.profile(n) + [1.0] * 4
    H1e, Be = M.assemble(n + 4, 1.0, form, N=Next); H0e, _ = M.assemble(n + 4, 0.0, form, N=Next)
    ie = {p_: i for i, p_ in enumerate(Be)}; emb = np.array([ie[p_] for p_ in B])
    S1 = H1e[:, emb].copy(); S1[emb, :] -= H1
    S0 = H0e[:, emb].copy(); S0[emb, :] -= H0
    Ve = H1e - H0e
    eH1e = K45.target_error(n + 4, 1.0, form, H1e); eH0e = K45.target_error(n + 4, 0.0, form, H0e)
    del H1e, H0e
    spe = np.array([(a % 8) // 4 + (b % 8) // 4 for a, b in Be])
    T = Ve[np.ix_(spe == 2, spe == 0)]; other = float(np.abs(Ve[np.ix_(spe != 2, spe == 0)]).max()); del Ve
    tn, tok, tE, tfl = C62.tnorm_cert(T)
    tn_up = tn + 1e-13 * float(np.linalg.norm(T)) + (eH1e + eH0e)
    del T
    SV = S1 - S0
    Cj1 = math.sqrt(schur(S1.conj().T @ S1)) + (eH1 + eH1e); H1bound = schur(H1) + eH1; L1 = Cj1 * H1bound
    C0 = math.sqrt(schur(S0.conj().T @ S0)) + (eH0 + eH0e); SVb = math.sqrt(schur(SV.conj().T @ SV)) + (eH1 + eH1e + eH0 + eH0e)
    L0 = C0 * (H0b + eH0)
    La = C0 * Vb * (4 * H0b + 1); Lc = SVb * H0b
    l1_4 = p1rows[3]['l1']
    Pa = 64
    while 4 * L1 / (4 * Pa) > 0.01:
        Pa *= 2
    Pb = 64
    while 2 * (4 * (La + Lc) / (4 * Pb)) / l1_4 > 0.01:
        Pb *= 2
    res['wall_setup'] = dict(T_norm_float=tfl, T_norm_certified_upper=tn_up, T_psd_certificate=tok, T_E_frob=tE, T_other_blocks_from_HH=other,
                             C_join_g1=Cj1, H1_bound=H1bound, L_g1=L1, C0=C0, S_V_norm=SVb, L0=L0, La=La, Lc=Lc, V_norm=Vb, H0_norm=H0b,
                             P_g1=Pa, P_psi1=Pb, quadrature_share_g1=L1 / Pa, quadrature_share_psi1=2 * (La + Lc) / Pb / l1_4)
    mark('wall_setup_and_P_rule', t0)
    if not tok:
        void('HH->LL norm certificate failed')
    # 5: budget projection from 8 test midpoints each
    t0 = time.process_time()
    for k in range(8):
        tm = (k + 0.5) / Pa; ph = V1 @ (np.exp(-1j * w1 * tm) * c1); _ = np.linalg.norm(S1 @ ph)
    per_a = (time.process_time() - t0) / 8
    t0b = time.process_time()
    for k in range(8):
        tm = (k + 0.5) / Pb; ph0 = V0 @ (np.exp(-1j * w0 * tm) * c0); p1, y = psi1(tm)
        _ = np.linalg.norm(S0 @ p1 + SV @ ph0); _ = np.linalg.norm(S0 @ ph0)
    per_b = (time.process_time() - t0b) / 8
    proj = 4 * Pa * per_a + 4 * Pb * per_b
    remaining = allowance - (time.process_time() - T0)
    res['budget'] = dict(per_midpoint_g1_s=per_a, per_midpoint_psi1_s=per_b, midpoints_g1=4 * Pa, midpoints_psi1=4 * Pb, projected_midpoint_cpu_s=proj, remaining_allowance_s=remaining)
    mark('budget_projection', t0)
    if proj > 0.9 * remaining:
        void(f'projected midpoint cost {proj:.0f} s exceeds the remaining allowance {remaining:.0f} s (no reduced-P fallback)')
    # 6a: g = 1 wall
    t0 = time.process_time()
    d = 1.0 / Pa; acc = 0.0; pref = {}
    for k in range(4 * Pa):
        tm = (k + 0.5) * d
        ph = V1 @ (np.exp(-1j * w1 * tm) * c1)
        N2lo = float(np.sum(ph.real ** 2 + ph.imag ** 2)) * (1 - 2 * D * U / (1 - 2 * D * U)) * (1 - U)
        fm = float(np.linalg.norm(S1 @ ph)) * (1 + 1e-12) / math.sqrt(N2lo) + 1e-13 + Cj1 * 2 * eps(tm) / math.sqrt(N2lo)
        acc += d * fm + L1 * d * d / 4
        if (k + 1) % Pa == 0:
            pref[(k + 1) // Pa] = acc * (1 + 1e-12)
    l4 = g1rows[3]['l_lower']
    E4 = pref[4]
    res['wall_g1'] = dict(P=Pa, E=pref, l_lower_t4=l4, l_minus_E_t4=l4 - E4, DW_t4=(2 * E4 / (2 * l4 - E4)) * (1 + 1e-12) if l4 - E4 > 0 else None)
    mark('wall_g1_midpoints', t0)
    # 6b: psi1 wall (construction C, DW)
    t0 = time.process_time()
    d = 1.0 / Pb; E0 = 0.0; E1 = 0.0; pref1 = {}
    for k in range(4 * Pb):
        tm = (k + 0.5) * d
        ph0 = V0 @ (np.exp(-1j * w0 * tm) * c0); p1, y = psi1(tm)
        comb = float(np.linalg.norm(S0 @ p1 + SV @ ph0)) * (1 + 1e-12) + C0 * eps1(tm, float(np.linalg.norm(y))) + SVb * eps0(tm)
        f0 = float(np.linalg.norm(S0 @ ph0)) * (1 + 1e-12) + C0 * eps0(tm)
        E0 += d * f0 + L0 * d * d / 4
        E1 += d * comb + (La + Lc) * d * d / 4 + tn_up * d * E0
        if (k + 1) % Pb == 0:
            pref1[(k + 1) // Pb] = E1 * (1 + 1e-12)
    res['wall_psi1'] = dict(P=Pb, E1=pref1, DW={t_: ((2 * pref1[t_] / (2 * r['l1'] - pref1[t_])) * (1 + 1e-12) if r['l1'] - pref1[t_] > 0 else None) for t_, r in zip((1, 2, 3, 4), p1rows)},
                            l_minus_E={t_: r['l1'] - pref1[t_] for t_, r in zip((1, 2, 3, 4), p1rows)})
    mark('wall_psi1_midpoints', t0)
    # 7: verdict
    pre = dict(linearity=lin <= 1e-12, residuals=R0 <= 1e-10 and R1 <= 1e-10, VB=VBlo > 0,
               N_L_g1_all_clocks=all(r['N_L_prereq'] for r in g1rows), N_L_psi1_all_clocks=all(r['N_L_prereq'] for r in p1rows),
               wall_g1_t4=res['wall_g1']['DW_t4'] is not None and res['wall_g1']['DW_t4'] <= 0.1,
               wall_psi1_t4=res['wall_psi1']['DW'][4] is not None and res['wall_psi1']['DW'][4] <= 0.1,
               l_minus_E_certified=(res['wall_g1']['l_minus_E_t4'] > 0 and res['wall_psi1']['l_minus_E'][4] > 0))
    fL = g1rows[3].get('f_L'); fL1 = p1rows[3]['f_L1']
    Iv = [fL[0] - fL1[1], fL[1] - fL1[0]] if fL else None
    if not all(pre.values()) or Iv is None:
        verdict = 'VOID'
    elif Iv[0] >= -0.1 and Iv[1] <= 0.1:
        verdict = 'PASS (free-phase)'
    elif Iv[0] > 0.1 or Iv[1] < -0.1:
        verdict = 'FAIL (conversion-specific shift), sign ' + ('+' if Iv[0] > 0.1 else '-')
    else:
        verdict = 'VOID'
    res.update(prerequisites=pre, I=Iv, f_L_t4=fL, f_L1_t4=fL1, half_printed_alongside=0.5, verdict=verdict)
    np.savez(f'{OUT}/states_small.npz', **keep)
    res['retained_files'] = {fn: dict(bytes=os.path.getsize(f'{OUT}/{fn}')) for fn in os.listdir(OUT)}
    res['cpu_s_total'] = time.process_time() - T0; res['wall_s_total'] = time.time() - W0; res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json.dump(res, open(f'{OUT}/result.json', 'w'), indent=1, default=float)
    print(form, 'VERDICT', verdict, 'I', Iv, 'walls', res['wall_g1']['DW_t4'], res['wall_psi1']['DW'][4], 'cpu', res['cpu_s_total'], flush=True)

if __name__ == '__main__':
    main()
