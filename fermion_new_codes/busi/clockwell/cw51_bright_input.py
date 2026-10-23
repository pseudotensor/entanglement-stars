"""cw51: #495 (2): the bright registered input Psi_bright = (|0_R>|01>_H + |1_R> B')/sqrt 2, B' = (|03>_H + |12>_H)/sqrt 2
at site 0, n = 5, both forms, g = 0, 1, 2, clocks 1..5, propagated with the SAVED eigenbases (n5run/<form>_g<G>/eigenbasis.npz,
md5-checked); UNGATED. Admission per clock: N2 (both rows) certified positive in ball arithmetic, |N2 - 1| within
(1 + 2 eps)^2 - 1 (eps = the larger row error; the two rows each carry up to eps), Gram distance of the two evolved branches from I_2 within 2 (2 sqrt 2 eps + 2 eps^2). Ride-along:
D_RK for light-only whole-lead output and complete whole-lead output (+-3 eps_n). The (B, D) code stays as the dark
control (its D_RK from the v1 run is printed beside). No decoder SDP. States saved with md5.
Output: <internal>/bright_v1<form>_g<G>.json and states npz"""
import json, os, sys, time, resource, math, hashlib
import numpy as np
sys.path.insert(0, '<internal>/code')
import cw32_mockmodel as M
import flint
from flint import acb_mat, acb, arb
flint.ctx.prec = 128
U = 2.0 ** -53
R = '<internal>/n5run'; OUT = f'{R}/bright_v1'; os.makedirs(OUT, exist_ok=True)

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

def main():
    B = M.basis(5); idx = {p: i for i, p in enumerate(B)}; D = len(B)
    b0 = np.zeros(D, dtype=complex); b0[idx[(0, 1)]] = 1
    bp = np.zeros(D, dtype=complex); bp[idx[(0, 3)]] = 1 / math.sqrt(2); bp[idx[(1, 2)]] = 1 / math.sqrt(2)
    reg0 = np.vstack([b0 / math.sqrt(2), bp / math.sqrt(2)])
    out_light = [m for x in range(1, 5) for m in range(8 * x + 4, 8 * x + 8)]; out_all = list(range(8, 40))
    summary = {}
    for form in ('onsite', 'filtered'):
        for gs in ('0', '1', '2'):
            t0, w0 = time.process_time(), time.time()
            d = f'{R}/{form}_g{gs}'; rj = json.load(open(f'{d}/result.json'))
            assert md5(f'{d}/eigenbasis.npz') == rj['files']['eigenbasis.npz']
            E = np.load(f'{d}/eigenbasis.npz'); w, V = E['w'], E['V']
            Rf = rj['eig']['residual_frob_upper']; eH = rj['assembly']['e_H']
            Va = acb_from(V)
            cs = []; eps_f = []
            for r in range(2):
                c = V.conj().T @ reg0[r]
                d0 = frob_upper(Va * acb_from(c.reshape(-1, 1)) - acb_from(reg0[r].reshape(-1, 1)))
                cn = float(np.linalg.norm(c)) * (1 + 1e-12); fl = 2 * D * U * float(np.linalg.norm(V)) * cn
                cs.append(c); eps_f.append((d0, cn, fl))
            rows = []; states = {}
            for t in (1.0, 2.0, 3.0, 4.0, 5.0):
                rr = np.vstack([V @ (np.exp(-1j * w * t) * cs[r]) for r in range(2)]); states[f't{t:g}'] = rr
                eps = max(d0 + t * cn * (Rf + eH) + fl for d0, cn, fl in eps_f)
                N2 = norm2_ball(rr.ravel()); lo, hi = float(N2.lower()), float(N2.upper())
                admit = lo > 0 and max(abs(lo - 1), abs(hi - 1)) <= (1 + 2 * eps) ** 2 - 1
                G = 2 * (rr.conj() @ rr.T); Nr = float(np.real(np.trace(G))) / 2
                gram = float(np.linalg.norm(G / Nr - np.eye(2)))
                gram_ok = gram <= 2 * (2 * math.sqrt(2) * (1 + 1e-15) * eps + 2 * eps ** 2) + 1e-14
                rn = rr / math.sqrt(float(np.real(np.sum(np.abs(rr) ** 2))))
                epsn = 2 * eps / math.sqrt(lo)
                rows.append(dict(t=t, N2=[lo, hi], eps=eps, admitted=bool(admit), gram_distance=gram, gram_admitted=bool(gram_ok),
                                 D_light=M.D_RK(rn, B, out_light, 5), D_all=M.D_RK(rn, B, out_all, 5), allowance=3 * epsn + 1e-12,
                                 dark_control_D_light=[x for x in rj['register_ungated'] if x['t'] == t][0]['D_light'],
                                 dark_control_D_all=[x for x in rj['register_ungated'] if x['t'] == t][0]['D_all']))
            sp = f'{OUT}/{form}_g{gs}_states_bright.npz'; np.savez(sp, **states)
            info = dict(form=form, g=gs, eigenbasis_md5=rj['files']['eigenbasis.npz'], rows=rows, states_file=sp, states_md5=md5(sp),
                        cpu_s=time.process_time() - t0, wall_s=time.time() - w0, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            json.dump(info, open(f'{OUT}/{form}_g{gs}.json', 'w'), indent=1)
            summary[f'{form}_g{gs}'] = info
            print(form, gs, [(r['t'], round(r['D_light'], 6), round(r['D_all'], 6), r['admitted'], r['gram_admitted']) for r in rows], 'cpu', round(info['cpu_s'], 1), flush=True)
    summary['cpu_s_total'] = time.process_time()
    json.dump(summary, open(f'{OUT}/cw51_bright.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
