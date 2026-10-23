#!/usr/bin/env python3
"""obs_greybody_radial_v1.py (builder analyst; SPEC_final4b_observables.md sec. 8.1, "Gamma. The primary transmission is a stationary
scattering calculation on the frozen exterior", for J3 version 3; the scattering core is obs_plateau_v2's, imported).

Per radial channel (m, kappa_j = sign (j + 1/2)) on the radial chain's operator (on-site N (m sz + (kappa_j/R) sy), the sine hop, the range-two
flow hop; flow_chart_v4.radial_fields' convention) on the exact Painleve-Gullstrand exterior: Gamma_j(omega) = the share of the outgoing
long-wave mode at the outer lead that comes from inside the trapping surface (1 minus the reflection R from the outer incoming channels), on a
grid of 64 frequencies per decade over [kappa/4, 8 kappa], kappa = 1/(2 r_s).  BUILDER choices, labelled, carried over from J-PL (the record
results/obs/JPL/RESULT_JPL_v1.md): the interior is continued analytically, the same map for the flow and the centrifugal term:
rho(R) = R_1 + w log(1 + e^((R - R_1)/w)), v = -sqrt(r_s/rho), y = kappa_j/rho (rho = R outside up to w e^(-(R - R_1)/w)); the outer lead joins
by the smooth saturation rho -> R_lead - w_L log(1 + e^((R_lead - rho)/w_L)), w_L = 0.05 R_lead.  The spec's "inner cut at r_s - 2 and r_s - 4
agreeing within 5 percent" is replaced by the continuation ladder (R_1, w) (the inner branch mixing of J-PL's finding 3 makes a cut at a fixed
site a non-analytic boundary); the lead ladder R_lead is printed.  Checks per row: unitarity of the scattering matrix (to 1e-10), Gamma -> 0
as omega -> 0 with the printed power (the log-log slope over the lowest decade).  Args: --rs (list) --j (list) --m-over-kappa (list) --signs
--R-lead (list, units of r_s) --cont (list R_1/r_s:w) --npd --out-json --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import json
import math
import sys
import time

import numpy as np
from scipy.linalg import solve_banded

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_plateau_v2 as P  # noqa: E402


def band_matrix_y(v, N, m, y):
    """obs_plateau_v2.band_matrix with a site-dependent y (the centrifugal term)."""
    n = len(v)
    dim = 2 * n
    ab = np.zeros((2 * P.BW + 1, dim), complex)

    def put(r, c, val):
        np.add.at(ab, (P.BW + r - c, c), val)
    idx = np.arange(n)
    on = N[:, None, None] * (m * P.SZ[None] + y[:, None, None] * P.SY[None])
    for a_ in range(2):
        for b_ in range(2):
            put(2 * idx + a_, 2 * idx + b_, on[:, a_, b_])
    i1 = np.arange(n - 1)
    B = -0.5j * 0.5 * (N[:-1] + N[1:])
    for a_ in range(2):
        for b_ in range(2):
            if P.SX[a_, b_] != 0:
                put(2 * i1 + a_, 2 * i1 + 2 + b_, B * P.SX[a_, b_])
                put(2 * i1 + 2 + b_, 2 * i1 + a_, np.conj(B * P.SX[a_, b_]))
    i2 = np.arange(n - 2)
    C = -0.25j * 0.5 * (v[:-2] + v[2:])
    for a_ in range(2):
        put(2 * i2 + a_, 2 * i2 + 4 + a_, C)
        put(2 * i2 + 4 + a_, 2 * i2 + a_, np.conj(C))
    return ab


def scatter_y(v, N, m, y, omega):
    """obs_plateau_v2.scatter with the site-dependent y; the leads take the end values (the first and last 4 sites uniform)."""
    n = len(v)
    LL = P.lead_setup(v[0], N[0], m, y[0], omega, "L")
    LR = P.lead_setup(v[-1], N[-1], m, y[-1], omega, "R")
    ab = band_matrix_y(v, N, m, y)
    ab[P.BW, :] -= omega
    P.add_block(ab, 0, LL["Sig"])
    P.add_block(ab, 2 * n - 4, LR["Sig"])
    chans = [("L", md) for md in LL["inc"]] + [("R", md) for md in LR["inc"]]
    rhs = np.zeros((2 * n, len(chans)), complex)
    for c, (side, md) in enumerate(chans):
        Ub = np.concatenate([md["u"], md["lam"] * md["u"]])
        if side == "L":
            rhs[0:4, c] = -(LL["H01"].conj().T @ (Ub / md["lam"] ** 2 - LL["T"] @ Ub))
        else:
            rhs[2 * n - 4:2 * n, c] = -(LR["H01"] @ (md["lam"] ** 2 * Ub - LR["T"] @ Ub))
    psi = solve_banded((P.BW, P.BW), ab, rhs, check_finite=False)
    outs = [("L", i, md) for i, md in enumerate(LL["keep"]) if md["prop"]] + [("R", i, md) for i, md in enumerate(LR["keep"]) if md["prop"]]
    S = np.zeros((len(outs), len(chans)), complex)
    for c, (side, md) in enumerate(chans):
        Ub = np.concatenate([md["u"], md["lam"] * md["u"]])
        cL = LL["Ui"] @ (psi[0:4, c] - (Ub if side == "L" else 0))
        cR = LR["Ui"] @ (psi[2 * n - 4:2 * n, c] - (Ub if side == "R" else 0))
        for o, (oside, i, mo) in enumerate(outs):
            S[o, c] = (cL[i] if oside == "L" else cR[i]) * math.sqrt(abs(mo["vg"]) / abs(md["vg"]))
    unit = float(np.max(np.abs(np.sum(np.abs(S) ** 2, axis=0) - 1.0))) if S.size else None
    return {"S": S, "outs": outs, "chans": chans, "unitarity": unit}


def radial_profile(rs, kj, R_lead, R1, w, wL_frac=0.05):
    wL = wL_frac * R_lead
    R0 = R1 - 40.0 * w
    nsites = int(math.ceil(R_lead + 40.0 * wL - R0)) + 1
    nsites += nsites % 2
    R = R0 + np.arange(nsites, dtype=float)
    rho = R1 + w * np.logaddexp(0.0, (R - R1) / w)
    rho = R_lead - wL * np.logaddexp(0.0, (R_lead - rho) / wL)
    v = -np.sqrt(rs / rho)
    y = kj / rho
    for arr in (v, y):
        arr[:4] = arr[4]; arr[-4:] = arr[-5]
    return R, v, np.ones_like(v), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, nargs="+", default=[256.0, 512.0])
    ap.add_argument("--j", type=float, nargs="+", default=[0.5, 1.5])
    ap.add_argument("--m-over-kappa", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--signs", default="+1", help="comma list of the signs of kappa_j, e.g. +1,-1")
    ap.add_argument("--R-lead", type=float, nargs="+", default=[32.0, 64.0])
    ap.add_argument("--cont", nargs="+", default=["0.0625:4", "0.125:4"])
    ap.add_argument("--npd", type=int, default=64, help="frequencies per decade over [kappa/4, 8 kappa]")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    signs = [float(s) for s in a.signs.split(",")]
    res = []
    for rs in a.rs:
        kappa = 1.0 / (2 * rs)
        om = kappa * np.logspace(math.log10(0.25), math.log10(8.0), int(round(a.npd * math.log10(32.0))) + 1)
        for j in a.j:
            for sg in signs:
                kj = sg * (j + 0.5)
                for mk in a.m_over_kappa:
                    m = mk * kappa
                    for RL in a.R_lead:
                        for c in a.cont:
                            R1, w = float(c.split(":")[0]) * rs, float(c.split(":")[1])
                            R, v, N, y = radial_profile(rs, kj, RL * rs, R1, w)
                            t0 = time.time()
                            G, U, Rf = [], [], []
                            for o in om:
                                if o <= m * 1.0005:
                                    G.append(None); U.append(None); Rf.append(None)
                                    continue
                                try:
                                    sc = scatter_y(v, N, m, y, o)
                                    rd = P.hawking_reading(sc, o, kappa / (2 * math.pi))
                                    G.append(rd.get("Gamma")); U.append(sc["unitarity"]); Rf.append(rd.get("R"))
                                except RuntimeError as e:
                                    G.append(None); U.append(str(e)); Rf.append(None)
                            ok = [(o, g) for o, g in zip(om, G) if g is not None and g > 0]
                            low = [(math.log(o), math.log(g)) for o, g in ok if o <= 2.5 * max(kappa / 4, m * 1.01)]
                            power = float(np.polyfit(*zip(*low), 1)[0]) if len(low) >= 4 else None
                            row = {"rs": rs, "j": j, "kappa_j": kj, "m_over_kappa": mk, "R_lead_over_rs": RL, "cont": c, "omega_over_kappa": (om / kappa).tolist(),
                                   "Gamma": G, "R": Rf, "unitarity": U, "low_power": power, "seconds": time.time() - t0, "nsites": len(R)}
                            res.append(row)
                            uu = [x for x in U if isinstance(x, float)]
                            print("rs %g j %g kappa_j %+g m/kappa %g R_lead %g r_s cont %s: Gamma at omega/kappa 0.25, 1, 4: %s; low power %s; unitarity max %.1e (%.0f s)" % (
                                rs, j, kj, mk, RL, c, [G[int(np.argmin(np.abs(om / kappa - q)))] for q in (0.25, 1.0, 4.0)], power, max(uu) if uu else float("nan"),
                                time.time() - t0), flush=True)
                            json.dump({"script": "obs_greybody_radial_v1.py", "args": vars(a), "rows": res}, open(a.out_json, "w"), indent=1, default=float)
    L = ["## Gamma_j(omega) on the exact exterior by stationary scattering (obs_greybody_radial_v1.py)", "",
         "| r_s | j | kappa_j | m/kappa | R_lead/r_s | continuation | Gamma at omega/kappa = 0.25, 0.5, 1, 2, 4, 8 | low-frequency power | max unitarity error |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        om = np.array(r["omega_over_kappa"])
        vals = []
        for q in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            g = r["Gamma"][int(np.argmin(np.abs(om - q)))]
            vals.append("%.6f" % g if g is not None else "-")
        uu = [x for x in r["unitarity"] if isinstance(x, float)]
        L.append("| %g | %g | %+g | %g | %g | %s | %s | %s | %s |" % (r["rs"], r["j"], r["kappa_j"], r["m_over_kappa"], r["R_lead_over_rs"], r["cont"], ", ".join(vals),
                 ("%.3f" % r["low_power"]) if r["low_power"] is not None else "-", ("%.1e" % max(uu)) if uu else "-"))
    L += ["", "- every number above is printed by the script; the full Gamma_j(omega) grids are in the JSON record."]
    open(a.md, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
