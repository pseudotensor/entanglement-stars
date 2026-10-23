"""cw45: the n = 5 mock run, ONE ARM per process (form, g), per MOCK_n5_plan_v2.md. busi's own pipeline (cw32 builder).
Usage: cw45_n5_run.py FORM G   (FORM in onsite|filtered, G in 0|1|2). Outputs: <internal>/n5run<form>_g<G>/
Gated readouts (g = 1 only) are for the normalized PURE input B; registered-state readouts are separate and ungated.
Certification (sec. 4 of the plan):
  e_H   = ||H_binary64 - H_target||_F (exact target m = 3/5, exact G_r): computed from the exact parameter deltas times the
          unit-parameter matrices (dyadic), plus an a priori assembly allowance 1e-13 ||H||_F;
  Rres  = H V - V Lambda in ball arithmetic (128 bit), ||Rres||_F upper;
  eps(t)= ||V c - psi0|| + t ||c|| (||Rres||_F + e_H) + 2 D u ||V||_F ||c||, c = V^dagger psi0 (ball arithmetic for the first);
  N2    = ||phi(t)||^2 in ball arithmetic from the stored binary64 vector; admitted iff lower N2 > 0 and |N2 - 1| <= (1 + eps)^2 - 1;
  eps_n = 2 eps / sqrt(lower N2) (normalized-vector error); expectations of A: +- 2 ||A|| eps_n + 1e-13.
"""
import json, os, sys, time, resource, math, hashlib
from fractions import Fraction as Fr
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import flint
from flint import acb_mat, acb, arb
flint.ctx.prec = 128
U = 2.0 ** -53

def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()

def acb_from(Mx):
    Mx = np.atleast_2d(Mx)
    return acb_mat([[acb(float(x.real), float(x.imag)) for x in row] for row in Mx])

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

def h_bin(x):
    return 0.0 if x <= 0 or x >= 1 else -x * math.log(x) - (1 - x) * math.log(1 - x)

def audenaert(T, d):
    T = min(1.0, T)
    return math.log(d) if T >= 1 - 1 / d else T * math.log(d - 1) + h_bin(T)

def route_B(n, g, form, B):
    """independent assembly: apply every one-body and quartic monomial to every basis bitmask with Jordan-Wigner signs"""
    idx = {p: i for i, p in enumerate(B)}; D = len(B)
    h = M.one_body(n); Hb = np.zeros((D, D), dtype=complex)
    def apply(ops, state):  # ops: list of (mode, dagger) applied right to left; state: frozenset bits as int
        sign = 1; s = state
        for m, dag in reversed(ops):
            occ = (s >> m) & 1
            if dag == occ:
                return None, 0
            sign *= (-1) ** bin(s & ((1 << m) - 1)).count('1')
            s ^= (1 << m)
        return s, sign
    for j, (a, b2) in enumerate(B):
        st = (1 << a) | (1 << b2)
        for p in range(h.shape[0]):
            for q in range(h.shape[0]):
                if h[p, q] != 0:
                    s, sg = apply([(p, True), (q, False)], st)
                    if s is not None:
                        ms = [k for k in range(8 * n) if (s >> k) & 1]
                        Hb[idx[(ms[0], ms[1])], j] += sg * h[p, q]
    if g:
        C = M.onsite_quartic(n) if form == 'onsite' else M.filtered_quartic(n)
        for j, (a, b2) in enumerate(B):
            st = (1 << a) | (1 << b2)
            for (p, q, s_, r), cf in C.items():
                if cf == 0:
                    continue
                s, sg = apply([(p, True), (q, True), (s_, False), (r, False)], st)
                if s is not None:
                    ms = [k for k in range(8 * n) if (s >> k) & 1]
                    i = idx[(ms[0], ms[1])]
                    Hb[i, j] += g * sg * cf
                    Hb[j, i] += g * sg * np.conj(cf)
    return Hb

def target_error(n, g, form, H):
    """||H_float - H_target||_F from exact parameter deltas (mass, G_r) times unit-parameter matrices + a priori allowance"""
    dm = float(Fr(M.MH) - Fr(3, 5))
    save = M.MH
    M.MH = 1.0; Hm1, _ = M.assemble(n, 0.0, form); M.MH = 0.0; Hm0, _ = M.assemble(n, 0.0, form); M.MH = save
    dH = dm * (Hm1 - Hm0)
    if g and form == 'filtered':
        Gexact = {0: Fr(388627261303, 549873254400), 1: Fr(456290907571, 3115948441600), 2: Fr(180865117, 983983718400)}
        Gsave = dict(M.G)
        for r in (0, 1, 2):
            M.G.clear(); M.G.update({0: 0.0, 1: 0.0, 2: 0.0}); M.G[r] = 1.0
            Hr, _ = M.assemble(n, g, form); H0r, _ = M.assemble(n, 0.0, form)
            dH = dH + float(Fr(Gsave[r]) - Gexact[r]) * (Hr - H0r)
        M.G.clear(); M.G.update(Gsave)
    return float(np.linalg.norm(dH)) * (1 + 1e-6) + 1e-13 * float(np.linalg.norm(H))

def cert_eig(H):
    w, V = np.linalg.eigh(H)
    R = acb_from(H) * acb_from(V) - acb_from(V * w)
    return w, V, frob_upper(R)

def prop(V, w, c, t):
    return V @ (np.exp(-1j * w * t) * c)

def main():
    form, gs = sys.argv[1], sys.argv[2]; g = float(gs)
    n = 5; T0 = time.process_time(); W0 = time.time()
    OUT = f'<internal>/{form}_g{gs}'; os.makedirs(OUT, exist_ok=True)
    cost = []
    def mark(name, t0, w0):
        cost.append(dict(stage=name, cpu_s=time.process_time() - t0, wall_s=time.time() - w0, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if time.process_time() - T0 > 7000:
            raise SystemExit('CPU cap reached')
    res = dict(n=n, form=form, g=g, header=('endpoint-copy halo: N_-1 = N_0 = 1/2, N_5 = N_4 = 1 (declared convention)' if form == 'filtered' else 'on-site converter'))
    t0, w0 = time.process_time(), time.time()
    H, B = M.assemble(n, g, form); D = len(B)
    Hb = route_B(n, g, form, B)
    res['assembly'] = dict(dim=D, nnz=int(np.count_nonzero(H)), max_abs_routeA_minus_routeB=float(np.abs(H - Hb).max()), hermitian_defect=float(np.abs(H - H.conj().T).max()))
    del Hb
    eH = target_error(n, g, form, H); res['assembly']['e_H'] = eH
    mark('assembly_two_routes_and_target_error', t0, w0)
    t0, w0 = time.process_time(), time.time()
    w, V, Rf = cert_eig(H); res['eig'] = dict(residual_frob_upper=Rf, normV_F=float(np.linalg.norm(V)))
    mark('eigendecomposition_and_certified_residual', t0, w0)
    psi0, reg0 = M.initial_states(B)
    clocks = [1.0, 2.0, 3.0, 4.0, 5.0]
    def certify_traj(v0):
        c = V.conj().T @ v0
        Va = acb_from(V); ca = acb_from(c.reshape(-1, 1)); v0a = acb_from(v0.reshape(-1, 1))
        d0 = frob_upper(Va * ca - v0a)
        cn = float(np.linalg.norm(c)) * (1 + 1e-12)
        fl = 2 * D * U * float(np.linalg.norm(V)) * cn
        return c, (lambda t: d0 + t * cn * (Rf + eH) + fl), dict(d0=d0, c_norm=cn, float_allowance=fl)
    t0, w0 = time.process_time(), time.time()
    cp, epsp, ep_info = certify_traj(psi0)
    creg = []; epsr = []
    for r in range(2):
        c_, e_, _ = certify_traj(reg0[r]); creg.append(c_); epsr.append(e_)
    res['state_error_terms'] = ep_info
    sp = np.array([(a % 8) // 4 + (b % 8) // 4 for a, b in B])
    nL = np.array([((a % 8) // 4) + ((b % 8) // 4) for a, b in B], dtype=float)
    nH = 2 - nL
    lead = lambda m: m >= 8
    nLlead = np.array([sum(1 for m in (a, b) if (m % 8) // 4 == 1 and lead(m)) for a, b in B], dtype=float)
    nHlead = np.array([sum(1 for m in (a, b) if (m % 8) // 4 == 0 and lead(m)) for a, b in B], dtype=float)
    pure = {}; regs = {}; rows = []
    for t in clocks:
        phi = prop(V, w, cp, t); pure[t] = phi
        N2 = norm2_ball(phi); eps = epsp(t)
        N2lo = float(N2.lower()); N2hi = float(N2.upper())
        admit = N2lo > 0 and max(abs(N2lo - 1), abs(N2hi - 1)) <= (1 + eps) ** 2 - 1
        epsn = 2 * eps / math.sqrt(N2lo) if N2lo > 0 else float('inf')
        p = np.abs(phi) ** 2 / (N2lo + N2hi) * 2
        def ex(vals, normA):
            v = float(np.dot(p, vals)); a = 2 * normA * epsn + 1e-13
            return [v - a, v + a]
        pLL = ex((sp == 2).astype(float), 1); NL = ex(nL, 2); NH = ex(nH, 2); NLl = ex(nLlead, 2); NHl = ex(nHlead, 2)
        fL = [max(0.0, NLl[0]) / NL[1], NLl[1] / NL[0]] if NL[0] > 0 else None
        fH = [max(0.0, NHl[0]) / NH[1], NHl[1] / NH[0]] if NH[0] > 0 else None
        rr = np.vstack([prop(V, w, creg[0], t), prop(V, w, creg[1], t)]); regs[t] = rr
        G = 2 * (rr.conj() @ rr.T)
        Nr = float(np.real(np.trace(G))) / 2
        epsg = max(epsr[0](t), epsr[1](t))
        gram = float(np.linalg.norm(G / Nr - np.eye(2)))
        sq2 = math.sqrt(2) * (1 + 1e-15)
        gram_ok = gram <= 2 * (2 * sq2 * epsg + 2 * epsg ** 2) + 1e-14
        rows.append(dict(t=t, N2=[N2lo, N2hi], eps=eps, eps_n=epsn, admitted=admit, p_LL=pLL, N_L=NL, N_H=NH, N_L_lead=NLl, N_H_lead=NHl,
                         f_L=fL, f_H=fH, registered_gram_distance=gram, registered_eps=epsg, registered_admitted=gram_ok))
    res['clocks'] = rows
    mark('actions_admission_gated_readouts', t0, w0)
    # Bwall
    t0, w0 = time.process_time(), time.time()
    Next = M.profile(n) + [1.0] * 4
    Hext, Bext = M.assemble(n + 4, g, form, N=Next)
    iext = {p_: i for i, p_ in enumerate(Bext)}; emb = np.array([iext[p_] for p_ in B])
    S = Hext[:, emb].copy(); S[emb, :] -= H
    eHext = target_error(n + 4, g, form, Hext); del Hext
    Cj2 = float(np.abs(S.conj().T @ S).sum(axis=1).max()) * (1 + 1e-12) + 1e-12
    Cj = math.sqrt(Cj2) * (1 + 1e-15) + (eH + eHext)
    Hbound = float(np.abs(H).sum(axis=1).max()) * (1 + 1e-12) + eH
    L = Cj * Hbound; d = 1 / 64
    pref = {}; acc = 0.0; mids = []
    for k in range(64 * 5):
        tm = (k + 0.5) * d
        phi = prop(V, w, cp, tm); N2 = float(norm2_ball(phi).lower())
        epsn = 2 * epsp(tm) / math.sqrt(N2)
        fm = float(np.linalg.norm(S @ phi)) * (1 + 1e-12) / math.sqrt(N2) + 1e-13 + Cj * epsn
        acc += d * fm + L * d * d / 4; mids.append(fm)
        if abs((k + 1) * d - round((k + 1) * d)) < 1e-12:
            pref[round((k + 1) * d)] = acc * (1 + 1e-12)
    res['wall'] = dict(C_join=Cj, H_bound=Hbound, L=L, e_H_ext=eHext, panels_per_unit=64, prefix={str(k): v for k, v in pref.items()},
                       S_shape=list(S.shape), meaning='pure-input Duhamel state-distance bound to the empty 4-site extension; not a probability')
    np.savez(f'{OUT}/joining_source_S.npz', S=S); del S
    mark('wall_source_and_320_midpoints', t0, w0)
    # gate (pure B), g = 1 only
    pr = {r['t']: r for r in rows}
    tstar = max([t for t in clocks if pref[int(t)] <= 0.1], default=None)
    gate = None
    if g == 1.0:
        if tstar is None:
            gate = dict(verdict='VOID', reason='no clock with certified Bwall <= 0.1')
        else:
            prev = max([t for t in clocks if t < tstar], default=None)
            r = pr[tstar]; rp = pr[prev] if prev else None
            if tstar < 3 or rp is None:
                gate = dict(verdict='VOID', reason='t* < 3')
            elif not (r['admitted'] and rp['admitted']):
                gate = dict(verdict='UNRESOLVED', reason='state admission failed')
            elif rp['p_LL'][1] < 0.1 or r['N_L'][1] < 0.05:
                gate = dict(verdict='VOID', reason='prerequisite certified below threshold')
            elif rp['p_LL'][0] < 0.1 or r['N_L'][0] < 0.05 or not (r['N_H'][0] > 0):
                gate = dict(verdict='UNRESOLVED', reason='prerequisite interval straddles its threshold or heavy divisor not certified positive')
            else:
                fL, fH = r['f_L'], r['f_H']
                if fL[0] >= 0.5 and fL[0] > fH[1]:
                    gate = dict(verdict='PASS', meaning='a wall-controlled converted-light transport witness for the pure bright input at n = 5; not release, purification or a Page turnover')
                elif fL[1] < 0.5 and rp['p_LL'][0] >= 0.1:
                    gate = dict(verdict='FAIL')
                else:
                    gate = dict(verdict='UNRESOLVED', reason='f_L or f_L vs f_H not decided by the certified intervals')
            gate.update(t_star=tstar, t_prev=prev)
    res['gate'] = gate if gate else dict(verdict='ungated (g != 1)', t_star=tstar)
    # LL band
    t0, w0 = time.process_time(), time.time()
    LLi = np.where(sp == 2)[0]
    ev = np.linalg.eigvalsh(H[np.ix_(LLi, LLi)])
    EB = float(np.real(np.vdot(psi0, H @ psi0)))
    res['LL_band'] = dict(dim=len(LLi), min=float(ev.min()), max=float(ev.max()), heavy_input_energy=EB, inside=bool(ev.min() <= EB <= ev.max()),
                          spectrum_file='LL_spectrum.npz')
    np.savez(f'{OUT}/LL_spectrum.npz', eig=ev)
    mark('LL_band', t0, w0)
    # ride-along: registered D_RK, Cq at site 4
    t0, w0 = time.process_time(), time.time()
    out_light = [m for x in range(1, 5) for m in range(8 * x + 4, 8 * x + 8)]; out_all = list(range(8, 40))
    dr = []
    for t in clocks:
        rr = regs[t] / math.sqrt(float(np.real(np.sum(np.abs(regs[t]) ** 2))))
        e3 = 3 * 2 * max(epsr[0](t), epsr[1](t))
        dr.append(dict(t=t, D_light=M.D_RK(rr, B, out_light, n), D_all=M.D_RK(rr, B, out_all, n), allowance=e3 + 1e-12))
    res['register_ungated'] = dr
    site4 = set(range(32, 40))
    keep = lambda modes: (set(modes) <= site4) or not (set(modes) & site4)
    Hq, _ = M.assemble(n, g, form, keep=keep)
    wq, Vq, Rq = cert_eig(Hq)
    p4 = pure[4.0] / math.sqrt(float(np.real(np.vdot(pure[4.0], pure[4.0]))))
    cq = Vq.conj().T @ p4
    q5 = prop(Vq, wq, cq, 1.0)
    epsq = 2 * epsp(4.0) + float(np.linalg.norm(cq)) * (Rq + eH) + 2 * D * U * float(np.linalg.norm(Vq)) * float(np.linalg.norm(cq))
    q5n = q5 / np.linalg.norm(q5)
    rA2, d2 = M.reduce_rho(q5n.reshape(1, -1), B, list(range(24, 40)))
    rA1, d1 = M.reduce_rho(q5n.reshape(1, -1), B, list(range(32, 40)))
    Cq = M.vn(rA2) - M.vn(rA1); Tq = min(1.0, 2 * epsq)
    res['Cq_ungated'] = dict(t1=4.0, t2=5.0, B1='site 4', B2='site 3', value=Cq, interval=[Cq - audenaert(Tq, d2) - audenaert(Tq, d1), Cq + audenaert(Tq, d2) + audenaert(Tq, d1)],
                             protected_residual=Rq, eps_protected=epsq, dims=[d2, d1])
    mark('ride_along_register_and_Cq', t0, w0)
    # save
    t0, w0 = time.process_time(), time.time()
    np.savez(f'{OUT}/hamiltonian.npz', H=H); np.savez(f'{OUT}/eigenbasis.npz', w=w, V=V)
    np.savez(f'{OUT}/states_pure.npz', **{f't{t:g}': pure[t] for t in clocks}); np.savez(f'{OUT}/states_registered.npz', **{f't{t:g}': regs[t] for t in clocks})
    np.savez(f'{OUT}/protected.npz', Hq=Hq, wq=wq, Vq=Vq, state_t5=q5)
    mark('serialization', t0, w0)
    res['cost'] = cost; res['cpu_s_total'] = time.process_time() - T0; res['wall_s_total'] = time.time() - W0
    res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    res['files'] = {fn: md5(f'{OUT}/{fn}') for fn in sorted(os.listdir(OUT)) if fn.endswith('.npz')}
    res['bytes'] = {fn: os.path.getsize(f'{OUT}/{fn}') for fn in sorted(os.listdir(OUT)) if fn.endswith('.npz')}
    json.dump(res, open(f'{OUT}/result.json', 'w'), indent=1)
    print(form, gs, 'gate', res['gate'], 'cpu', round(res['cpu_s_total'], 1), 'rss', res['peak_rss_kib'], flush=True)

if __name__ == '__main__':
    main()
