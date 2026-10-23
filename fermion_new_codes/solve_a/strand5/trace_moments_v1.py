import json, numpy as np, sys
sys.path.insert(0, '<internal>/code')
import strand5_lattice_star_v3 as S
D = json.load(open('<internal>/kernel_J4_L256.json'))
rows = sorted([r for r in D['equal'] if abs(r['aH'] - 0.1) < 1e-12 and abs(r.get('m_light', 0.1) - 0.01) < 1e-12], key=lambda r: r['n'])
q = np.array([r['q'] for r in rows]); k = np.array([r['k_a'] for r in rows]); L = 256
kk = k.copy(); kk[-1] /= 2
Cq = np.zeros(L); ql2 = 2 * (1 - np.cos(q))
for j, qq in enumerate(q):
    n = int(round(qq * L / (2 * np.pi))); Cq[n % L] = -kk[j] * ql2[j]; Cq[(-n) % L] = -kk[j] * ql2[j]
K = np.real(np.fft.ifft(Cq))
nn = np.arange(L); nn = np.where(nn > L // 2, nn - L, nn)
t3k = 0.5 * np.sum(nn ** 2 * K)
st, kmom, R = S.line_kernel(q, k)
print('T3k route %.10e  J4 line_kernel %.10e  diff %.4e  R %d' % (t3k, kmom, t3k - kmom, R))
print('ring n = L/2 term (1/2)(L/2)^2 K_{L/2} = %.4e   (K_{L/2}/K_0 = %.3e)' % (0.5 * (L // 2) ** 2 * K[L // 2], K[L // 2] / K[0]))
print('J4 stencil moment from ring K over |n| <= 127 = %.10e' % (0.5 * sum(n * n * K[n % L] for n in range(-127, 128))))
