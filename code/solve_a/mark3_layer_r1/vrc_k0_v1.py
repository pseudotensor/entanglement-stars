"""vrc_k0_v1 (#684, ADDENDUM_A10_vrc620_cubicK0_v1): massless limit of the cubic K. kappa = vrc_kmid_v1's kappa_formula (e2a8db58) on the
midpoint grid p_i = -pi + (2i + 1) pi / N (off the doubler nodes), summed in p1-slabs; components c11 = K[0,0,0,0], c12 = K[0,0,1,1],
kappa_12_12 = K[0,1,0,1], kappa_12_21 = K[0,1,1,0], c44_sym = (K[0,1,0,1] + K[0,1,1,0])/2. (A) m = 0 at N = 128..1024 with fitted exponents and
Richardson; (B) m = 0.2, 0.1, 0.05, 0.025 at N = 512, 1024 and m -> 0 extrapolations. Usage: python vrc_k0_v1.py <out.json>"""
import sys, json, time
import numpy as np
T0 = time.time()


def grid(n): return -np.pi + (2 * np.arange(n) + 1) * np.pi / n


def comps(m, n, slab=16):
    p = grid(n); acc = dict(c11=0.0, c12=0.0, kappa_12_12=0.0, kappa_12_21=0.0)
    P2, P3 = np.meshgrid(p, p, indexing='ij'); s2, s3 = np.sin(P2), np.sin(P3); c2 = np.cos(P2); q2 = np.sin(2 * P2) / 2
    for i0 in range(0, n, slab):
        p1 = p[i0:i0 + slab][:, None, None]
        s1, c1, q1 = np.sin(p1), np.cos(p1), np.sin(2 * p1) / 2
        e2 = s1 ** 2 + s2[None] ** 2 + s3[None] ** 2 + m * m; w = 1.0 / (4 * e2 * np.sqrt(e2))
        g00 = c1 * c1 * (1.0 - s1 * s1 / e2) * w          # a = b = 1
        g01 = c1 * c2[None] * (0.0 - s1 * s2[None] / e2) * w   # a = 1, b = 2
        # K[a,i,b,j] = sum q_i q_j g_ab:  c11 = K[0,0,0,0], c12 = K[0,0,1,1], kappa_12_12 = K[0,1,0,1], kappa_12_21 = K[0,1,1,0]
        acc['c11'] += float((q1 * q1 * g00).sum())
        acc['c12'] += float((q1 * q2[None] * g01).sum())
        acc['kappa_12_12'] += float((q2[None] * q2[None] * g00).sum())
        acc['kappa_12_21'] += float((q2[None] * q1 * g01).sum())
    out = {k: v / n ** 3 for k, v in acc.items()}
    out['c44_sym'] = (out['kappa_12_12'] + out['kappa_12_21']) / 2
    return out


KEYS = ('c11', 'c12', 'kappa_12_12', 'kappa_12_21', 'c44_sym')
res = dict(addendum='ADDENDUM_A10_vrc620_cubicK0_v1', grid='midpoint p_i = -pi + (2i+1) pi/N', A={}, B={})
# self-check against vrc_kmid_v1 at m = 0.2, n = 128 (formula_128 there)
chk = comps(0.2, 128); res['selfcheck_m0p2_n128'] = chk; print('selfcheck', json.dumps(chk), flush=True)
for n in (128, 256, 512, 1024):
    res['A'][str(n)] = comps(0.0, n); print('A m 0 N', n, json.dumps(res['A'][str(n)]), '%.0f s' % (time.time() - T0), flush=True)
A = res['A']; encl = {}
for k in KEYS:
    K = [A[str(n)][k] for n in (128, 256, 512, 1024)]
    p = float(np.log2(abs(K[0] - K[1]) / abs(K[1] - K[2]))); pp = float(np.log2(abs(K[1] - K[2]) / abs(K[2] - K[3])))
    Kinf = K[3] + (K[3] - K[2]) / (2 ** pp - 1); Kinf2 = K[3] + (K[3] - K[2]) / (2 ** p - 1)
    half = abs(Kinf - K[3]) + abs(Kinf - Kinf2); widened = abs(p - pp) > 0.3
    if widened:
        lo = min(K[3], Kinf, Kinf2) - abs(K[3] - K[2]); hi = max(K[3], Kinf, Kinf2) + abs(K[3] - K[2])
    else:
        lo, hi = Kinf - half, Kinf + half
    encl[k] = dict(levels=K, p_128_256_512=p, p_256_512_1024=pp, K_inf=Kinf, K_inf_alt=Kinf2, enclosure=[lo, hi], widened=widened)
res['A_enclosure'] = encl; print('A enclosure', json.dumps(encl), flush=True)
for m in (0.2, 0.1, 0.05, 0.025):
    r5, r10 = comps(m, 512), comps(m, 1024)
    res['B'][str(m)] = {'512': r5, '1024': r10, 'converged': {k: bool(abs(r5[k] - r10[k]) <= 1e-3 * abs(r10[k])) for k in KEYS}}
    print('B m', m, json.dumps(res['B'][str(m)]), '%.0f s' % (time.time() - T0), flush=True)
mx = {}
for k in KEYS:
    ms = [0.1, 0.05, 0.025]; Ks = [res['B'][str(m)]['1024'][k] for m in ms]
    lin = Ks[2] - (Ks[1] - Ks[2]) / (ms[1] - ms[2]) * ms[2]
    quad = float(np.polyval(np.polyfit(ms, Ks, 2), 0.0))
    e = encl[k]['enclosure']; cons = bool(e[0] <= lin <= e[1] and e[0] <= quad <= e[1])
    final = [min(e[0], lin, quad), max(e[1], lin, quad)]
    mx[k] = dict(K_m=dict(zip([str(m) for m in ms], Ks)), m_to_0_linear=lin, m_to_0_quadratic=quad, consistent_with_A=cons, final_enclosure=final,
                 converged_flags={str(m): res['B'][str(m)]['converged'][k] for m in ms})
res['B_extrapolation'] = mx; print('B extrapolation', json.dumps(mx), flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
print('done %.0f s' % (time.time() - T0), flush=True)
