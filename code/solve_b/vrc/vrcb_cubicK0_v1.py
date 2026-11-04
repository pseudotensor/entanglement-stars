# vrcb_cubicK0_v1.py (box CPU) -- VRC amend7 (961b41ef), lead #484: the MASSLESS cubic strain tensor (reviewer B w2fulla eq. (5)) at m = 0 on shifted
# midpoint grids N = 128, 256, 512 (+1024 if argv[2] == '1'), reduced algebraic form (literal 4-vector check at N = 128), symmetric map,
# Richardson in 1/N with fitted exponent and the amend7 enclosure; ma -> 0 cross-check m = 0.2, 0.1, 0.05, 0.025 at N = 512.
# Usage: <out.json> <do1024 0|1>
import sys, json, itertools, time
import numpy as np
outp, do1024 = sys.argv[1], sys.argv[2] == '1'
def K_reduced(m, L):
    p = -np.pi + (np.arange(L) + 0.5) * 2 * np.pi / L
    s, c, q = np.sin(p), np.cos(p), 0.5 * np.sin(2 * p)
    S2, S3 = np.meshgrid(s, s, indexing='ij'); C2, C3 = np.meshgrid(c, c, indexing='ij'); Q2, Q3 = np.meshgrid(q, q, indexing='ij')
    K = np.zeros((3, 3, 3, 3))
    for i1 in range(L):
        sv = [np.full_like(S2, s[i1]), S2, S3]; cv = [np.full_like(C2, c[i1]), C2, C3]; qv = [np.full_like(Q2, q[i1]), Q2, Q3]
        e2 = sv[0] ** 2 + sv[1] ** 2 + sv[2] ** 2 + m * m; e = np.sqrt(e2)
        M = {}
        for a in range(3):
            for b in range(a, 3):
                M[(a, b)] = cv[a] * cv[b] * ((1.0 if a == b else 0.0) / e2 - sv[a] * sv[b] / e2 ** 2)
        for i in range(3):
            for j in range(i, 3):
                w = qv[i] * qv[j] / e
                for (a, b), Mab in M.items():
                    val = float(np.sum(w * Mab))
                    for (aa, bb) in {(a, b), (b, a)}:
                        for (ii, jj) in {(i, j), (j, i)}:
                            K[aa, ii, bb, jj] += val
    return K * 0.25 / L ** 3
def K_literal(m, L):
    p = -np.pi + (np.arange(L) + 0.5) * 2 * np.pi / L
    s, c, q = np.sin(p), np.cos(p), 0.5 * np.sin(2 * p)
    S2, S3 = np.meshgrid(s, s, indexing='ij'); C2, C3 = np.meshgrid(c, c, indexing='ij'); Q2, Q3 = np.meshgrid(q, q, indexing='ij')
    K = np.zeros((3, 3, 3, 3))
    for i1 in range(L):
        sv = [np.full_like(S2, s[i1]), S2, S3]; cv = [np.full_like(C2, c[i1]), C2, C3]; qv = [np.full_like(Q2, q[i1]), Q2, Q3]
        e2 = sv[0] ** 2 + sv[1] ** 2 + sv[2] ** 2 + m * m; e = np.sqrt(e2)
        d = np.stack([sv[0], sv[1], sv[2], np.full_like(e, m)])
        dn = []
        for a in range(3):
            dd = np.zeros_like(d); dd[a] = cv[a]
            dn.append(dd / e - d * (sv[a] * cv[a] / e) / e2)
        for a, i, b, j in itertools.product(range(3), repeat=4):
            K[a, i, b, j] += np.sum(qv[i] * qv[j] / e * np.sum(dn[a] * dn[b], axis=0))
    return K * 0.25 / L ** 3
def smap(K):
    c11 = [K[a, a, a, a] for a in range(3)]; c12 = [K[a, a, b, b] for a in range(3) for b in range(3) if a != b]
    k1212 = [K[a, b, a, b] for a in range(3) for b in range(3) if a != b]; k1221 = [K[a, b, b, a] for a in range(3) for b in range(3) if a != b]
    return dict(c11=float(np.mean(c11)), c12=float(np.mean(c12)), k1212=float(np.mean(k1212)), k1221=float(np.mean(k1221)),
                c44_sym=float(0.5 * (np.mean(k1212) + np.mean(k1221))),
                spread=float(max(np.ptp(c11), np.ptp(c12), np.ptp(k1212), np.ptp(k1221))))
out = dict(manifest='VRC_MANIFEST_route2_v1_amend7.md 961b41ef', grid='p_n = -pi + (n + 1/2) 2 pi / N (shifted midpoint, off the doublers)')
T0 = time.time()
Ns = [128, 256, 512] + ([1024] if do1024 else [])
seq = {}
for N in Ns:
    seq[N] = smap(K_reduced(0.0, N)); print(N, json.dumps(seq[N]), f'{time.time() - T0:.0f}s', flush=True)
out['m0_sequence'] = {str(N): v for N, v in seq.items()}
lit = smap(K_literal(0.0, 128)); out['literal_check_N128'] = {k: abs(lit[k] - seq[128][k]) for k in ('c11', 'c12', 'k1212', 'k1221')}
def extrap(a, b, c):
    num, den = a - b, b - c
    if den == 0 or num / den <= 0:
        return None, None
    p = float(np.log2(num / den)); return c + (c - b) / (2 ** p - 1), p
enc = {}
for k in ('c11', 'c12', 'k1212', 'k1221', 'c44_sym'):
    v = [seq[N][k] for N in Ns]
    Kinf, p = extrap(v[-3], v[-2], v[-1])
    row = dict(values=v, p_fit=p, K_inf=Kinf)
    if Kinf is None:
        lo, hi = min(v[-2:]), max(v[-2:]); w_ = hi - lo; row.update(enclosure=[lo - w_, hi + w_], rule='non-monotone: last two widened')
    else:
        if len(v) >= 4:
            Kp, pp = extrap(v[-4], v[-3], v[-2]); alt = Kp; row['p_fit_prev'] = pp
            unstable = pp is None or abs(pp - p) > 0.3
        else:
            pr = round(p); alt = v[-1] + (v[-1] - v[-2]) / (2 ** pr - 1) if pr > 0 else Kinf; unstable = False; row['p_int'] = pr
        delta = abs(Kinf - v[-1]) + (abs(alt - Kinf) if alt is not None else abs(Kinf - v[-1])) + seq[Ns[-1]]['spread']
        if unstable:
            lo, hi = min(v[-2:]), max(v[-2:]); w_ = hi - lo; row.update(enclosure=[lo - w_, hi + w_], rule='p unstable between triples')
        else:
            row.update(enclosure=[Kinf - delta, Kinf + delta], delta=delta, alt=alt, rule='amend7')
    enc[k] = row
    print(k, json.dumps(row), flush=True)
out['enclosures'] = enc
ms = [0.2, 0.1, 0.05, 0.025]
mseq = {m: smap(K_reduced(m, 512)) for m in ms}
out['m_sequence_N512'] = {str(m): v for m, v in mseq.items()}
mex = {}
for k in ('c11', 'c12', 'c44_sym'):
    v = [mseq[m][k] for m in ms]
    Kl, pm = extrap(v[-3], v[-2], v[-1])
    mex[k] = dict(values=v, p_fit_m=pm, m_to_0=Kl, m0_N512=seq[512][k])
out['m_to_0_crosscheck'] = mex
print('m->0', json.dumps(mex), flush=True)
out['wall_s'] = time.time() - T0
json.dump(out, open(outp, 'w'), indent=1, default=float)
print('END', out['wall_s'], flush=True)
