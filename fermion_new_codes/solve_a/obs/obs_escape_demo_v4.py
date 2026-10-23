#!/usr/bin/env python3
"""obs_escape_demo_v4.py (builder analyst; B8 of SPEC_final4b_observables.md sec. 11.6): (1) the hole reading: the lower packet of
obs_escape_demo_v3 (k_0 = arccos(N/|v|) - dk at R_0 = r_s/2, projected on h < 0) propagated to the clean time, and the weight of its far part
(R > 1.5 r_s) on the far sea's occupied modes under the flat projector (the negative-energy projector of the chain with v = 0, by the Chebyshev
erf step at width --ws), against the upper packet's (which must be near 0); (2) the stationary route's conversion row at omega < 0 against the
omega > 0 row (obs_plateau_v2's scattering on its pgs profile at one size): the outer lead's outgoing long-wave channel at -omega is the
negative-energy right-mover (k < 0); its decomposition over the inner in-channels, n_H, |alpha|^2, |beta|^2 and T_conv per sign, printed
side by side (by charge conjugation the -omega row is expected to carry the omega row with the occupations exchanged: B8's check).
Args: --rs --dk --sigma-x --t-over-rs --ws --kappa-a --r --out-json --md."""
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
import obs_escape_demo_v2 as ED  # noqa: E402
import obs_plateau_v2 as P2  # noqa: E402
import obs_raymap_solved_v1 as J3P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--dk", type=float, default=0.1)
    ap.add_argument("--sigma-x", type=float, default=12.0)
    ap.add_argument("--t-over-rs", type=float, default=8.0)
    ap.add_argument("--ws", type=float, default=0.01)
    ap.add_argument("--kappa-a", type=float, default=1.0 / 128)
    ap.add_argument("--r", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rs = a.rs
    R, v, N, info = P2.profile_pgs(rs, 8 * rs, rs / 16, 4.0, None)
    n = len(R)
    B = J3P.Diags(N, v, 0.0, np.zeros(n))
    Bflat = J3P.Diags(N, np.zeros(n), 0.0, np.zeros(n))
    i0 = int(np.argmin(np.abs(R - 0.5 * rs)))
    vv = abs(float(v[i0])); kb = math.acos(min(1.0, float(N[i0]) / vv))
    rec = {"script": "obs_escape_demo_v4.py", "args": vars(a), "part1": [], "part2": []}
    L = ["## B8 (obs_escape_demo_v4.py): the hole reading and the stationary row at negative Killing frequency (the exact profile with the analytic interior, r_s %g)" % rs, "",
         "### (1) The far part's weight on the flat sea's occupied modes (R_0 = r_s/2, t = %g r_s)" % a.t_over_rs, "",
         "| packet | k_0 | norm beyond 1.5 r_s | its weight on the flat sea's occupied modes |", "|---|---|---|---|"]
    for sgn, lab in ((+1, "upper (h > 0)"), (-1, "lower (h < 0)")):
        k0 = kb + sgn * a.dk
        x = R - R[i0]
        env = np.exp(-x ** 2 / (4 * a.sigma_x ** 2)) * np.exp(1j * k0 * x)
        psi = np.zeros(2 * n, complex); psi[0::2] = env / math.sqrt(2.0); psi[1::2] = env / math.sqrt(2.0)
        psi /= np.linalg.norm(psi)
        fpsi = ED.cheb_apply_step(B, psi, a.ws)
        proj = (psi - fpsi) if sgn > 0 else fpsi
        proj /= np.linalg.norm(proj)
        fin = J3P.cheb_expm(B, proj, a.t_over_rs * rs)
        far = np.where(np.repeat(R > 1.5 * rs, 2), fin, 0.0)
        nf = float(np.vdot(far, far).real)
        occ = ED.cheb_apply_step(Bflat, far, a.ws)
        w_occ = float(np.vdot(far, occ).real) / nf if nf > 0 else None
        rec["part1"].append({"packet": lab, "k0": k0, "far_norm": nf, "flat_sea_weight": w_occ})
        L.append("| %s | %.4f | %.6f | %s |" % (lab, k0, nf, ("%.6f" % w_occ) if w_occ is not None else "-"))
    L += ["", "### (2) The stationary row at +omega and -omega (kappa a = %g, the pgs profile R_1 = r_s/16, w = 4, R_lead = 32 r_s)" % a.kappa_a, "",
          "| omega/kappa | sign | n_H | |alpha|^2 | |beta|^2 | Gamma | T_conv/T_H | k_H | the inner channels' weights (k: weight, occupied) |", "|---|---|---|---|---|---|---|---|---|"]
    rs2 = 1.0 / (2 * a.kappa_a)
    R2, v2, N2, info2 = P2.profile_pgs(rs2, 32 * rs2, rs2 / 16, 4.0, None)
    kap = info2["kappa"]
    for r in a.r:
        for sg in (+1, -1):
            om = sg * r * kap
            sc = P2.scatter(v2, N2, 0.0, 0.0, om)
            # the outgoing long-wave channel on the outer lead (|k| < pi/2) at either sign of omega
            Hs = [o for o, (side, i, mo) in enumerate(sc["outs"]) if side == "R" and abs(mo["k"]) < math.pi / 2]
            if len(Hs) != 1:
                L.append("| %g | %+d | %d long-wave outgoing channels | | | | | | |" % (r, sg, len(Hs)))
                continue
            h = Hs[0]
            row = np.abs(sc["S"][h]) ** 2
            occ_w = sum(row[c] for c, (side, md) in enumerate(sc["chans"]) if md["wff"] < 0)
            a2 = sum(row[c] for c, (side, md) in enumerate(sc["chans"]) if side == "L" and md["wff"] >= 0)
            b2 = sum(row[c] for c, (side, md) in enumerate(sc["chans"]) if side == "L" and md["wff"] < 0)
            Gam = a2 + b2
            Tc = (abs(om) / math.log(max(a2, b2) / min(a2, b2))) if (a2 > 0 and b2 > 0 and a2 != b2) else None
            inner = ", ".join("%.3f: %.3e%s" % (md["k"], row[c], " occ" if md["wff"] < 0 else "") for c, (side, md) in enumerate(sc["chans"]) if side == "L" and row[c] > 1e-12)
            rec["part2"].append({"omega_over_kappa": r, "sign": sg, "n_H": float(occ_w), "alpha2": float(a2), "beta2": float(b2), "Gamma": float(Gam),
                                 "T_conv_over_TH": (Tc / (kap / (2 * math.pi))) if Tc else None, "k_H": sc["outs"][h][2]["k"], "unitarity": sc["unitarity"]})
            L.append("| %g | %+d | %.6e | %.6e | %.6e | %.9f | %s | %.5f | %s |" % (r, sg, occ_w, a2, b2, Gam, ("%.6f" % (Tc / (kap / (2 * math.pi)))) if Tc else "-",
                     sc["outs"][h][2]["k"], inner))
    L += ["", "- every number above is printed by the script; at -omega the empty and occupied labels follow the free-fall frequency's sign as at +omega;",
          "  T_conv uses the larger of the two inner weights over the smaller (the conversion ratio's magnitude)."]
    json.dump(rec, open(a.out_json, "w"), indent=1, default=float)
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
