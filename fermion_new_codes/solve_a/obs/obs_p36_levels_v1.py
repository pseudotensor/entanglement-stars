#!/usr/bin/env python3
"""obs_p36_levels_v1.py (builder analyst; SPEC_final4b_observables.md sec. 9.2, task 5 (3): P36's levels (i) and (ii) on the radial chain;
the exact profile with the analytic interior as the control now, a collapse slice (obs_plateau_v4's import) when it exists; level (iii),
D_det, needs two collapse runs and is not here).

(i) The cross-cut anticommutator: for x in A_in (the sites R_t - d_in .. R_t - 1) and y in A_out (R_t + 1 .. R_t + d_out), the one-body
propagator's element |[U(t)]_{yx}|^2 = sum over the spinor components of |<y a| e^(-i h t) |x b>|^2 (the anticommutator {c_y(t), c_x^dagger}
of the free field), printed against t: its small-t form (t |h_yx|)^2 for the nearest pairs, its maximum over A_in x A_out, and the escaping share
of a particle started at x (the norm beyond R_far = 2 r_s) at late times. (ii) The packet channel: a Gaussian packet on the outward
positive-frequency branch launched at R_t - 4 (projected on h > 0 by the Chebyshev erf step, obs_escape_demo_v2's route) at omega/kappa in
{1/2, 1, 2}, central momentum from the local root of omega = sin k (1 + v cos k) on the + branch; tau = its norm beyond R_far at the late time
(the channel E_out(rho) = tau rho + (1 - tau) |vac><vac| for a spin-encoded packet, the two spin states being the channels kappa_j = +-1, whose
taus are printed side by side). Args: --rs --profile (pgs|solved) --fields --records --slice --j --tmax-over-rs --nt --dmax --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import math
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import obs_escape_demo_v2 as ED  # noqa: E402
import obs_plateau_v4 as P  # noqa: E402
import obs_raymap_solved_v1 as J3P  # noqa: E402


def chain_fields(a):
    """the radial chain's sites R = 1 .. n, (N, v) on them, the surface R_t and kappa_T."""
    rs = a.rs
    n = int(a.nsites)
    R = np.arange(1, n + 1, dtype=float)
    if a.profile == "pgs":
        R1, w = rs / 16, 4.0
        rho = R1 + w * np.logaddexp(0.0, (R - R1) / w)
        v = -np.sqrt(rs / rho); N = np.ones(n)
        Rt = brentq(lambda x: R1 + w * math.log1p(math.exp((x - R1) / w)) - rs if (x - R1) / w < 700 else x - rs, R1, 2 * rs)
        kT = 1.0 / (2 * rs)
        label = "the exact profile with the analytic interior (R_1 = r_s/16, w = 4)"
    else:
        fn, sinfo = P.slice_fields(a.fields, a.records, a.slice, a.g)
        N, v = fn(R)
        N = np.asarray(N, float); v = np.asarray(v, float)
        fs = N + v
        i = int(np.nonzero(fs[:-1] * fs[1:] <= 0)[0][-1])
        Rt = float(R[i] - fs[i] * (R[i + 1] - R[i]) / (fs[i + 1] - fs[i]))
        kT = float(fs[i + 1] - fs[i])
        rs = 2 * a.g * sinfo["E"]
        label = "collapse slice %d (%s)" % (a.slice, sinfo["label"])
    return R, N, v, Rt, kT, rs, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--profile", choices=["pgs", "solved"], default="pgs")
    ap.add_argument("--fields", default=None)
    ap.add_argument("--records", default=None)
    ap.add_argument("--slice", type=int, default=-1)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--j", type=float, default=0.5)
    ap.add_argument("--nsites", type=int, default=1024)
    ap.add_argument("--tmax-over-rs", type=float, default=6.0)
    ap.add_argument("--nt", type=int, default=12)
    ap.add_argument("--dmax", type=int, default=4)
    ap.add_argument("--omega-over-kappa", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    R, N, v, Rt, kT, rs, label = chain_fields(a)
    n = len(R)
    Rfar = 2.0 * rs
    rec = {"script": "obs_p36_levels_v1.py", "args": vars(a), "label": label, "R_t": Rt, "kappa_T": kT, "level_i": [], "level_ii": []}
    L = ["## P36's levels (i) and (ii) on the radial chain (obs_p36_levels_v1.py): %s; r_s %g, R_t %.4f, kappa_T %.6e, j %g" % (label, rs, Rt, kT, a.j), ""]
    ts = np.linspace(0.0, a.tmax_over_rs * rs, a.nt + 1)
    it = int(math.floor(Rt)) - 1                                   # the site index of the last site inside the surface
    L += ["### (i) The cross-cut propagator element |[U(t)]_{yx}|^2 (x inside, y outside the surface) and the escaping share", "",
          "| kappa_j | t/r_s | max over x, y | x, y of the max | nearest pair (x = R_t - 1, y = R_t + 1) | share beyond 2 r_s from x = R_t - 1 |", "|---|---|---|---|---|---|"]
    for sgn in (+1.0, -1.0):
        kj = sgn * (a.j + 0.5)
        B = J3P.Diags(N, v, 0.0, kj / R)
        cols = []
        xs = [it - d for d in range(a.dmax)]
        for x in xs:
            for comp in (0, 1):
                e = np.zeros(2 * n, complex); e[2 * x + comp] = 1.0
                cols.append(e)
        X = np.array(cols).T                                           # 2n x (2 dmax)
        cur, tcur = X.copy(), 0.0
        for t in ts:
            if t > tcur:
                cur = np.column_stack([J3P.cheb_expm(B, cur[:, c], t - tcur) for c in range(cur.shape[1])])
                tcur = t
            amp = np.abs(cur) ** 2
            best, bxy = 0.0, None
            for xi, x in enumerate(xs):
                blk = amp[:, 2 * xi:2 * xi + 2].sum(axis=1).reshape(-1, 2).sum(axis=1) / 2.0
                for d in range(a.dmax):
                    y = it + 1 + d
                    if blk[y] > best:
                        best, bxy = float(blk[y]), (float(R[x]), float(R[y]))
            near = float(amp[:, 0:2].sum(axis=1).reshape(-1, 2).sum(axis=1)[it + 1] / 2.0)
            share = float(amp[:, 0:2].sum(axis=1).reshape(-1, 2).sum(axis=1)[R > Rfar].sum() / 2.0)
            rec["level_i"].append({"kappa_j": kj, "t_over_rs": t / rs, "max": best, "argmax": bxy, "nearest": near, "share_far": share})
            L.append("| %+g | %.2f | %.4e | %s | %.4e | %.4e |" % (kj, t / rs, best, ("%.0f, %.0f" % bxy) if bxy else "-", near, share))
    L += ["", "### (ii) The packet channel: tau = the norm beyond 2 r_s at the late time, the packet launched at R_t - 4 on h > 0", "",
          "| omega/kappa | k_0 | tau (kappa_j = +1) | tau (kappa_j = -1) | projected norm kept (+1, -1) |", "|---|---|---|---|---|"]
    i0 = int(round(Rt - 4)) - 1
    vv = abs(float(v[i0])); Nl = float(N[i0])
    for r_ in a.omega_over_kappa:
        om = r_ * kT
        f = lambda k: Nl * math.sin(k) * (1.0 - (vv / Nl) * math.cos(k)) - om
        kb = math.acos(min(1.0, Nl / vv))
        try:
            k0 = brentq(f, kb + 1e-9, math.pi / 2)
        except ValueError:
            k0 = kb + 0.1
        taus, kept = [], []
        for sgn in (+1.0, -1.0):
            kj = sgn * (a.j + 0.5)
            B = J3P.Diags(N, v, 0.0, kj / R)
            x = R - R[i0]
            env = np.exp(-x ** 2 / (4 * 12.0 ** 2)) * np.exp(1j * k0 * x)
            psi = np.zeros(2 * n, complex); psi[0::2] = env / math.sqrt(2.0); psi[1::2] = env / math.sqrt(2.0)
            psi /= np.linalg.norm(psi)
            fpsi = ED.cheb_apply_step(B, psi, 0.01)
            proj = psi - fpsi
            kp = float(np.vdot(proj, proj).real)
            proj /= math.sqrt(kp)
            fin = J3P.cheb_expm(B, proj, a.tmax_over_rs * rs)
            rho = (np.abs(fin.reshape(-1, 2)) ** 2).sum(axis=1)
            taus.append(float(rho[R > Rfar].sum())); kept.append(kp)
        rec["level_ii"].append({"omega_over_kappa": r_, "k0": k0, "tau": taus, "projected_norm": kept})
        L.append("| %g | %.4f | %.6f | %.6f | %.4f, %.4f |" % (r_, k0, taus[0], taus[1], kept[0], kept[1]))
    L += ["", "- every number above is printed by the script; the chain is open at R = %d (a packet reaching the end by t_max reflects: t_max is chosen below it)." % n]
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-10:]))


if __name__ == "__main__":
    main()
