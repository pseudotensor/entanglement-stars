# obs_plateau_pgs_probe_v1.py (builder analyst): the probe that found the analytic interior continuation for J-PL (run once on the box from obs/JPL; printed output only; superseded by obs_plateau_v2.py profile pgs).
import sys, math, numpy as np
sys.path.insert(0, "<internal>/code")
import obs_plateau_v1 as P
from scipy.optimize import brentq

def profile_pgs(rs, R_lead, R1, w):
    R0 = R1 - 40 * w
    n = int(math.ceil(R_lead - R0)) + 1; n += n % 2
    R = R0 + np.arange(n, dtype=float)
    t = (R - R1) / w
    rho = R1 + w * np.logaddexp(0.0, t)
    v = -np.sqrt(rs / rho)
    v[:4] = v[4]; v[-4:] = v[-5]
    Rh = brentq(lambda x: R1 + w * np.logaddexp(0.0, (x - R1) / w) - rs, R1, rs + 1)
    kap = (1.0 / (2 * rs)) / (1.0 + math.exp(-(Rh - R1) / w))
    return R, v, np.ones_like(v), kap

for rs, R1, w in ((64, 16, 4), (64, 8, 4), (64, 16, 6), (32, 4, 2.5), (32, 8, 2), (128, 32, 8), (128, 16, 6)):
    R, v, N, kap = profile_pgs(rs, 32 * rs, R1, w)
    out = []
    for r in (0.25, 1.0):
        rd = P.run_point(v, N, kap, r, 0.0)
        out.append("r %g theta %.9f c %.4f unit %.1e" % (r, rd["theta"], (1 - rd["theta"]) / (1 / (2 * rs)) ** 2, rd["unitarity"]))
    print("rs %d R1 %g w %g kappa*2rs %.9f |" % (rs, R1, w, kap * 2 * rs), " | ".join(out), flush=True)
