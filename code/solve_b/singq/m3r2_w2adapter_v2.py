# m3r2_w2adapter_v2.py -- adapter of vref_r1's FULL-FLOW W_2 (verbatim copy m3r2_w2chain_copy_v2.py; reviewer B w2fulla eqs. (2)-(4))
# to route 2's chains for the amendment-33 Stage 3. W_2[N, v] = -sum F_n / N_n per copy, degree one, exact in the flow amplitude
# at second-gradient order; clock source included.
#   dW2(chain, Nbar, vbar) -> (tN, tv) = chain weight x (dW_2/dN_n, dW_2/dv_n) / V_n   (route 2's source units)
#   heavy chains: kappa = k, m = m; the light chain (massless sector chain = the 2-component m = 0, k = 1 channel): kappa = 1,
#   m = 0 -- EXCLUDED (low-k massless, eta_M = 1/k at all r); computed and reported separately.
#   regions(chain): masks for the chain-uncontrolled core (r < 2/m for k <= 3; reviewer A coref), r < 32a, eta_M > 0.05 (reviewer B), light.
import numpy as np
import m3r2_w2chain_copy_v2 as W

_C = {}


def _coef(c):
    key = (c.kind, int(c.kch), c.ns, float(c.m))
    if key not in _C:
        m = float(c.m) if c.kind == 'heavy' else 0.0
        _C[key] = W.coeffs(c.ns, float(c.kch), m)
    return _C[key]


def dW2(c, Nb, vb):
    Cc = _coef(c)
    r = np.arange(1, c.ns + 1, dtype=float)
    V = 4 * np.pi * r ** 2
    Nb = np.asarray(Nb, float); vb = np.asarray(vb, float)
    return c.weight * W.grad_N(Nb, vb, Cc) / V, c.weight * W.grad_v(Nb, vb, Cc) / V


def W2(c, Nb, vb):
    return c.weight * W.W2(np.asarray(Nb, float), np.asarray(vb, float), _coef(c))


def eta(c):
    r = np.arange(1, c.ns + 1, dtype=float)
    m = float(c.m) if c.kind == 'heavy' else 0.0
    return W.eta_M(r, float(c.kch), m)


def regions(c):
    r = np.arange(1, c.ns + 1, dtype=float)
    light = np.full(c.ns, c.kind != 'heavy')
    m = float(c.m) if c.kind == 'heavy' else 0.0
    unc = (c.kind == 'heavy') & (int(c.kch) <= 3) & (r < 2.0 / m) if m > 0 else np.zeros(c.ns, bool)
    return dict(uncontrolled_core=np.asarray(unc, bool), r_lt_32=r < 32.0, eta_gt_005=eta(c) > 0.05, light=light)
