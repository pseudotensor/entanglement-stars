#!/usr/bin/env python3
"""obs_escape_demo_v3.py (builder analyst; version 3 of obs_escape_demo_v2.py: the far region's content at each printed time split by the
sign of the site momentum k on the sx = +1 component (an FFT of the part beyond 1.5 r_s: k > 0 the positive-energy outgoing branch, k < 0 the
negative-energy one, the sea's right-movers at v = 0), and the run to --tmax-over-rs (default 16): version 2 found the projected lower
packet (negative Killing frequency) escaping too, which the operator allows only on the k < 0 branch. Version 2's docstring follows.)

obs_escape_demo_v2.py (builder analyst; SPEC_final4b_observables.md sec. 10.3; version 2 of obs_escape_demo_v1.py: each packet is
projected on the sign of its Killing frequency before it is launched (the upper on h > 0, the lower on h < 0, by the Chebyshev expansion of
the erf step of the static operator at width --ws), because version 1's packets, narrow enough in space, spanned a region where the root
arccos(N/|v|) moves by more than dk, so each carried both signs of the Killing frequency and part of the lower one escaped (its record:
escape_demo_v1). A mode of positive free-fall and negative Killing frequency cannot exist where |v| < N, so the projected lower packet must
stay inside; the projected norms are printed. Version 1's docstring follows.

obs_escape_demo_v1.py (builder analyst; SPEC_final4b_observables.md sec. 10.3, the demonstration of the inner escape branch).

On the 1 + 1 axial chain of the model's operator (obs_plateau_v4's conventions; m = 0, y = 0, N = 1) on the exact Painleve-Gullstrand
profile continued analytically inside (obs_plateau_v2's pgs map, R_1 = r_s/16, w = 4; the control's interior, labelled): (1) per radius
inside the surface the two roots k = +-arccos(N/|v|) of omega = 0 on the + component (omega = sin k (1 + v cos k)) and their group velocity
N cos k + v cos 2k = |v| - N^2/|v|, printed; (2) two Gaussian packets launched at R_0 inside the surface on the + component (the sx = +1
component, which carries the outgoing long-wave mode outside) with central momenta k_0 = arccos(N/|v(R_0)|) +- dk (the Killing frequency
positive above the root, negative below), width sigma_x, propagated forward by Chebyshev exponentials; printed against time: the norm beyond
R_far = 1.5 r_s, beyond the surface, and inside R_0 / 2, and the packet's centroid. The reading (the lead's): the upper packet escapes to the far
region, the lower does not. Args: --rs --R0-over-rs --dk --sigma-x --tmax-over-rs --nt --nsites --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import math
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_plateau_v2 as P2  # noqa: E402
import obs_raymap_solved_v1 as J3P  # noqa: E402


def cheb_apply_step(B, X, ws):
    """f(h) X with f(E) = erfc(E/(sqrt 2 ws))/2 by the Chebyshev expansion on [-r, r] (obs_raymap_solved_v1's coefficients)."""
    from scipy.fft import dct
    from scipy.special import erfc
    r = B.bound
    M = 1
    while M < 8 * r / ws:
        M *= 2
    th = (np.arange(M) + 0.5) * math.pi / M
    c = dct(0.5 * erfc(r * np.cos(th) / (math.sqrt(2.0) * ws)), type=2) / M
    c[0] /= 2.0
    K = int(np.nonzero(np.abs(c) > 1e-13)[0][-1]) + 1
    T0, T1 = X, B.apply(X) / r
    Y = c[0] * T0 + c[1] * T1
    for k in range(2, K):
        T2 = 2 * B.apply(T1) / r - T0
        Y = Y + c[k] * T2
        T0, T1 = T1, T2
    return Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--R0-over-rs", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--dk", type=float, default=0.1)
    ap.add_argument("--sigma-x", type=float, default=12.0)
    ap.add_argument("--tmax-over-rs", type=float, default=16.0)
    ap.add_argument("--nt", type=int, default=16)
    ap.add_argument("--nsites", type=int, default=None, help="default the pgs chain to R_lead = 8 r_s")
    ap.add_argument("--ws", type=float, default=0.01, help="the width of the Killing-frequency step")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rs = a.rs
    R, v, N, info = P2.profile_pgs(rs, 8 * rs, rs / 16, 4.0, None)
    n = len(R)
    L = ["## The inner escape branch (obs_escape_demo_v3.py, the packets projected on the sign of the Killing frequency; the exact profile with the analytic interior, r_s = %g, axial chain, m = 0)" % rs, "",
         "### (1) The roots k = +-arccos(N/|v|) inside the surface and their group velocity |v| - N^2/|v|", "", "| R/r_s | v | k_+ | group velocity |", "|---|---|---|---|"]
    for q in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95):
        i = int(np.argmin(np.abs(R - q * rs)))
        vv = abs(float(v[i])); kb = math.acos(min(1.0, float(N[i]) / vv))
        L.append("| %g | %.5f | %.5f | %.5f |" % (q, float(v[i]), kb, vv - float(N[i]) ** 2 / vv))
    L += ["", "### (2) The packets: the norm beyond 1.5 r_s, beyond the surface and inside R_0/2 against time", "",
          "| R_0/r_s | k_0 | omega at launch | t/r_s | norm beyond 1.5 r_s | beyond R_t | inside R_0/2 | centroid R/r_s | far share k in (0, pi/2) | far share k in (-pi/2, 0) |", "|---|---|---|---|---|---|---|---|---|---|"]
    B = J3P.Diags(N, v, 0.0, np.zeros(n))
    rec = {"script": "obs_escape_demo_v3.py", "args": vars(a), "rows": []}
    Rt = info["R_h"]
    for q in a.R0_over_rs:
        i0 = int(np.argmin(np.abs(R - q * rs)))
        vv = abs(float(v[i0])); kb = math.acos(min(1.0, float(N[i0]) / vv))
        for sgn in (+1, -1):
            k0 = kb + sgn * a.dk
            om0 = math.sin(k0) * (1.0 - vv * math.cos(k0))
            x = R - R[i0]
            env = np.exp(-x ** 2 / (4 * a.sigma_x ** 2)) * np.exp(1j * k0 * x)
            psi = np.zeros(2 * n, complex)
            psi[0::2] = env / math.sqrt(2.0); psi[1::2] = env / math.sqrt(2.0)          # the sx = +1 component
            psi /= np.linalg.norm(psi)
            # version 2: the projection on the sign of the Killing frequency (f(h) = erfc(h/(sqrt 2 ws))/2 is the projector on h < 0)
            fpsi = cheb_apply_step(B, psi, a.ws)
            proj = (psi - fpsi) if sgn > 0 else fpsi
            kept = float(np.vdot(proj, proj).real)
            psi = proj / math.sqrt(kept)
            ts = np.linspace(0.0, a.tmax_over_rs * rs, a.nt + 1)
            cur, tcur = psi.copy(), 0.0
            series = []
            for t in ts:
                if t > tcur:
                    cur = J3P.cheb_expm(B, cur, t - tcur)
                    tcur = t
                rho = (np.abs(cur.reshape(-1, 2)) ** 2).sum(axis=1)
                far = float(rho[R > 1.5 * rs].sum()); out = float(rho[R > Rt].sum()); inn = float(rho[R < 0.5 * q * rs].sum())
                cen = float((rho * R).sum() / rho.sum())
                comp = cur.reshape(-1, 2)
                plus = (comp[:, 0] + comp[:, 1]) / math.sqrt(2.0)                       # the sx = +1 component
                fsel = R > 1.5 * rs
                fp = np.fft.fft(np.where(fsel, plus, 0.0))
                kk = np.fft.fftfreq(len(plus)) * 2 * math.pi
                pw = np.abs(fp) ** 2
                kpos = float(pw[(kk > 0) & (kk < math.pi / 2)].sum()); kneg = float(pw[(kk < 0) & (kk > -math.pi / 2)].sum()); tot_ = float(pw.sum())
                fpos = kpos / tot_ if tot_ > 0 else 0.0; fneg = kneg / tot_ if tot_ > 0 else 0.0
                series.append({"t_over_rs": t / rs, "far": far, "beyond_Rt": out, "inside_half_R0": inn, "centroid_over_rs": cen / rs,
                               "far_share_k_pos": fpos, "far_share_k_neg": fneg})
                L.append("| %g | %.4f | %+.5f | %.2f | %.6f | %.6f | %.6f | %.3f | %.4f | %.4f |" % (q, k0, om0, t / rs, far, out, inn, cen / rs, fpos, fneg))
            rec["rows"].append({"R0_over_rs": q, "k0": k0, "omega_launch": om0, "k_root": kb, "projected_norm": kept, "series": series})
            L.append("| %g | %.4f | projected norm kept %.4f | | | | | | | |" % (q, k0, kept))
    L += ["", "- every number above is printed by the script; R_t = %.4f (the continued profile's surface)." % Rt]
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-12:]))


if __name__ == "__main__":
    main()
