"""cw50: #495 (1): ONE-SHOT recertification of the two g = 1 pure-B wall bounds at n = 5, from the SAVED matrices
(hamiltonian.npz, eigenbasis.npz, joining_source_S.npz of n5run/<form>_g1, md5-checked against result.json).
Rule stated in advance (#495): P = the smallest power of two with quadrature term 5 L / (4 P) <= 0.01 for both
g = 1 arms; the midpoint contributions are RECOMPUTED at P (not obtained by subtracting the old remainder); the
sampled-integrand sums are printed at P = 64, 256, 1024; no further refinement. Gate text, thresholds, clocks
and caps unchanged; the original VOIDs (P = 64) are preserved and reported.
Certification of each midpoint: f_mid = ||S phi|| (1 + 1e-12) / sqrt(lower N2) + 1e-13 + C_join eps_n, with
lower N2 = fl(sum |phi_i|^2) (1 - gamma_D), gamma_D = D u / (1 - D u) (a priori bound of a floating sum of
nonnegative terms); eps(t) recomputed from the saved eigenbasis: ||V c - psi0|| (ball arithmetic) + t ||c||
(||H V - V Lambda||_F (ball arithmetic, recomputed) + e_H (saved)) + 2 D u ||V||_F ||c||.
Output: <internal>/cw50_recert.json"""
import json, os, sys, time, resource, math, hashlib
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import flint
from flint import acb_mat, acb, arb
flint.ctx.prec = 128
U = 2.0 ** -53
R = '<internal>/n5run'; OUT = f'{R}/recert_v1'; os.makedirs(OUT, exist_ok=True)

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

def main():
    res = {'arms': {}, 'rule': '5 L / (4 P) <= 0.01 for both g = 1 arms; P the smallest power of two; one shot'}
    loaded = {}
    for form in ('onsite', 'filtered'):
        t0, w0 = time.process_time(), time.time()
        d = f'{R}/{form}_g1'; rj = json.load(open(f'{d}/result.json'))
        for fn in ('hamiltonian.npz', 'eigenbasis.npz', 'joining_source_S.npz'):
            assert md5(f'{d}/{fn}') == rj['files'][fn], fn
        H = np.load(f'{d}/hamiltonian.npz')['H']; E = np.load(f'{d}/eigenbasis.npz'); w, V = E['w'], E['V']; S = np.load(f'{d}/joining_source_S.npz')['S']
        D = H.shape[0]
        Rres = acb_from(H) * acb_from(V) - acb_from(V * w); Rf = frob_upper(Rres); del Rres
        B = M.basis(5); psi0, _ = M.initial_states(B)
        c = V.conj().T @ psi0
        d0 = frob_upper(acb_from(V) * acb_from(c.reshape(-1, 1)) - acb_from(psi0.reshape(-1, 1)))
        cn = float(np.linalg.norm(c)) * (1 + 1e-12); fl = 2 * D * U * float(np.linalg.norm(V)) * cn
        eH = rj['assembly']['e_H']
        eps = lambda t: d0 + t * cn * (Rf + eH) + fl
        Cj = rj['wall']['C_join']; L = rj['wall']['L']
        loaded[form] = (w, V, S, c, eps, Cj, L, D, rj)
        res['arms'][form] = dict(setup_cpu_s=time.process_time() - t0, setup_wall_s=time.time() - w0, residual_frob_upper=Rf, d0=d0, c_norm=cn, e_H=eH,
                                 C_join=Cj, L=L, saved_md5_verified=True)
    # rule
    P = 1
    while not all(5 * loaded[f_][6] / (4 * P) <= 0.01 for f_ in loaded):
        P *= 2
    res['P_rule'] = P
    res['quadrature_term_at_5'] = {f_: 5 * loaded[f_][6] / (4 * P) for f_ in loaded}
    gam = D * U / (1 - D * U)
    for form, (w, V, S, c, eps, Cj, L, D, rj) in loaded.items():
        sums = {}
        for Pp in (64, 256, 1024):
            t0, w0 = time.process_time(), time.time()
            dd = 1.0 / Pp; acc = 0.0; pref = {}
            for k in range(5 * Pp):
                tm = (k + 0.5) * dd
                phi = V @ (np.exp(-1j * w * tm) * c)
                N2lo = float(np.sum(phi.real ** 2 + phi.imag ** 2)) * (1 - gam)
                epsn = 2 * eps(tm) / math.sqrt(N2lo)
                fm = float(np.linalg.norm(S @ phi)) * (1 + 1e-12) / math.sqrt(N2lo) + 1e-13 + Cj * epsn
                acc += dd * fm
                if (k + 1) % Pp == 0:
                    pref[(k + 1) // Pp] = acc
            sums[Pp] = dict(sampled_sum={str(t): v * (1 + 1e-12) for t, v in pref.items()},
                            quadrature_term={str(t): t * L / (4 * Pp) for t in pref},
                            Bwall={str(t): (pref[t] + t * L / (4 * Pp)) * (1 + 1e-12) for t in pref},
                            cpu_s=time.process_time() - t0, wall_s=time.time() - w0)
            print(form, Pp, {t: round(v, 6) for t, v in sums[Pp]['Bwall'].items()}, 'cpu', round(sums[Pp]['cpu_s'], 1), flush=True)
        res['arms'][form]['ladder'] = {str(k): v for k, v in sums.items()}
        # gate with the rule's P (frozen gate text, thresholds)
        Bw = sums[P]['Bwall']; clocks = [1, 2, 3, 4, 5]
        pr = {int(r['t']): r for r in rj['clocks']}
        tstar = max([t for t in clocks if Bw[str(t)] <= 0.1], default=None)
        if tstar is None:
            gate = dict(verdict='VOID', reason='no clock with certified Bwall <= 0.1')
        else:
            prev = max([t for t in clocks if t < tstar], default=None); r = pr[tstar]; rp = pr[prev] if prev else None
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
                    gate = dict(verdict='FAIL', f_L=fL, f_H=fH, p_LL_prev=rp['p_LL'], N_L=r['N_L'], N_H=r['N_H'])
                else:
                    gate = dict(verdict='UNRESOLVED', reason='f_L or f_L vs f_H not decided by the certified intervals')
            gate.update(t_star=tstar, t_prev=prev)
        res['arms'][form]['gate_recertified'] = gate
        res['arms'][form]['gate_original_P64'] = rj['gate']
        print(form, 'original', rj['gate']['verdict'], 'recertified', gate, flush=True)
    res['cpu_s_total'] = time.process_time(); res['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    json.dump(res, open(f'{OUT}/cw50_recert.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
