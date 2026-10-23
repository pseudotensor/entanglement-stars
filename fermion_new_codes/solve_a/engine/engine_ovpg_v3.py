#!/usr/bin/env python
"""engine_ovpg_v3.py (builder analyst; engine spec 28.4 (d) and 28.10: "engine_ovpg_v3 (new; CPU), the OV solve with the lattice band's
native kinetic stress (2/3) <sum_i q_i^2/eps> in place of the thermodynamic pressure, N_B,max and M_max at the six (m a, g) pairs beside the
thermodynamic values; the lattice E 28.655580 at N_B 147.26 (handoff v2) is read against both").

WHAT REPLACES WHAT. Version 2 (engine_ovpg_v2.load_eos_consistent) takes the strand5_eos_v4 table's pressure P(mu) as a cubic spline in mu
through the table's nodes at the run's m a, n = dP/dmu, e = mu n - P (all zero at and below mu = m), and engine_ovpg_v1.shoot integrates
    m' = 4 pi r^2 e(mu),   Phi' = g (m + 4 pi r^3 P(mu)) / (r (r - 2 g m)),   mu = E_F exp(-Phi)   (the local filling: the Tolman condition)
out to the surface mu = m, with N_B = int 4 pi r^2 n(mu) / sqrt(1 - 2 g m/r) dr (engine_ovpg_v1.member). The pressure enters this system only
in the Phi' equation (it gravitates; the support is the local filling n(mu(r))). HERE e(mu) and n(mu) are version 2's, unchanged (the same
spline of the same table and its derivative), and only P(mu) in the Phi' equation is replaced by the native kinetic stress
    p_Q(mu) = (2/3) <sum_i q_i^2 / eps>,   eps = sqrt(m^2 + sum_i sin^2 k_i)   (a = 1),
<X> = int_BZ d^3k/(2 pi)^3 X 1_{eps < mu} over the one zone (all eight cones) and the factor 2 the two spin states: exactly the measure, zone
and degeneracy of strand5_eos_v4's n = 2<1> and e_0 = 2<eps>, evaluated on its own quadrature (strand5_eos_v4.Quad: Gauss-Legendre in the cone
angles ph, al (N2 each) and in th_3 (N3); production at the table's (N2, N3) = (2000, 64), checked at (1000, 48) and (3000, 96)).
The spec writes q_i without defining it; the builder's brief defines q_i = sin k_i. Both readings are computed and labelled:
  (a) q_i = sin k_i (THE SPEC AS BRIEFED; the primary result): p_sin = (1/3) 2<t^2/eps>, t^2 = sum_i sin^2 k_i; it equals -T_0/3 for the
      table's fixed-filling strain coefficient T_0 (T_hat), checked node by node against the table;
  (b) q_i = Q_i = sin(2 k_i)/2, the symbol of the model's momentum operator Q_i = (1/2i)[X_i, D^2] (engine_ops_v1, the two-step current),
      for which d eps/dk_i = Q_i/eps: p_SQ = (1/3) 2<sum_i Q_i^2/eps>, the isotropic native Q-transport stress S^(Q) of the review's P12,
      computed by periodic integration by parts as P - (2/3) 2<t^2 (mu - eps)> and checked against the direct integrand (the cone angle ph
      resolved) at a subset of nodes.
Printed beside: version 2's thermodynamic values (recomputed here with the same maximum search); version 1's (OVPG/ovpg_families.json's
M_max and z_c, with N_B at that z_c: the families' NB_max_continuum); the review's web-derived numbers (REVIEWA_REVIEW_web40e 1.16, S3, I6:
ideas, never sources); the lane's lattice members near each maximum (the family records given by --lattice) read against every closure at
the same N_B on that closure's stable branch (the member at N_B 147.26 with E 28.655580, STAR_family_handoff_v2.md, read from its record).
CONTROLS (labelled; not the spec's computation): (i) reading (b); (ii) the diagnostic p_G0 = chi - e with chi = lim_{b->0} J/b
= (1/3) 2<|Q|^2 delta(mu - eps)> = mu 2<1 - (2/3) t^2> (by parts), the lapse row's susceptibility at the local filling at rest: the b -> 0 limit
of the radial pressure J/b - rho of the review's P12 two-row closure (its eq. (17)); the Ward term W = e + S^(Q) - chi = (2/3) 2<eps t^2> is
printed (chi is also checked by a finite difference of the direct 2<|Q|^2> in mu); (iii) all four pressures again with e, n and P from the
dense zone table itself (kappa^p times cubic splines of X/kappa^p in kappa = sqrt(mu^2 - m^2), p = 3 for n and e and 5 for the pressures
and W, through 180 nodes kappa = m z, z = 0.02 to 3.6), the control on version 2's 20-node table.
Maxima: per closure M(z_c) and N_B(z_c) on version 1's scan (z_c = kappa_c/m, 48 geometric points 0.05 to 3), each maximum refined by bounded
Brent in z_c (xatol 1e-8); with a pressure that is not the table's thermodynamic one the two maxima need not coincide, so both are printed,
with the central filling (z_c, mu_c/m, kappa_c, theta_F = arcsin kappa_c) and, at the M maximum's centre, the dense table's ratios p_sin/P,
p_SQ/P, p_G0/P and W/p_SQ. CPU only. Args: --eos --ovpg --grid --lattice "m:g=json[,json...]" ... --out (json) --md --procs."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import brentq, minimize_scalar  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.path.isdir(os.path.join(HERE, "..", "strand5")):
    sys.path.append(os.path.join(HERE, "..", "strand5"))
import engine_ovpg_v1 as OV1  # noqa: E402
import engine_ovpg_v2 as OV2  # noqa: E402
import strand5_eos_v4 as SE  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
C16 = SE.C16
ZS = np.round(0.02 * np.arange(1, 181), 10)               # the dense table's nodes z = kappa/m
LEVELS = {"coarse": (1000, 48), "production": (2000, 64), "fine": (3000, 96)}
KEYS = ("n", "e", "P", "p_sin", "p_SQ", "p_G0", "W")
REVIEW = [  # REVIEWA_REVIEW_web40e.md (web-derived ideas, never sources): sections 1.16, S3 and I6
    {"m": 0.2, "g": 0.25, "N_B": 149.15, "E": 28.670, "where": "1.16, S3, I6"},
    {"m": 0.1, "g": 0.25, "N_B": 1142.6, "E": None, "where": "I6 (S3: near 1142)"},
    {"m": 0.05, "g": 0.25, "N_B": 9048.0, "E": None, "where": "1.16, S3, I6"},
    {"m": 0.2, "g": 0.0625, "N_B": 1193.2, "E": None, "where": "I6 (1.16: 1193)"},
    {"m": 0.1, "g": 0.0625, "N_B": 9140.0, "E": None, "where": "1.16, S3"},
]
CLOSURES = [  # (key, label, the e/n source, the pressure in the Phi' equation)
    ("v2_thermo", "version 2: thermodynamic P (the table spline), e and n version 2's", "v2", "P_v2"),
    ("v3_native_sin", "version 3 (THE SPEC): native stress q_i = sin k_i, e and n version 2's", "v2", "p_sin"),
    ("v3_native_SQ", "control (i): native stress q_i = Q_i = sin(2k_i)/2 (S^(Q)), e and n version 2's", "v2", "p_SQ"),
    ("v3_chi_diag", "control (ii): p_G0 = chi - e (b -> 0 limit of P12's J/b - rho), e and n version 2's", "v2", "p_G0"),
    ("dense_thermo", "control (iii): thermodynamic P, e, n from the dense zone table", "dense", "P"),
    ("dense_native_sin", "control (iii): native q_i = sin k_i, dense e, n", "dense", "p_sin"),
    ("dense_native_SQ", "control (iii): native S^(Q), dense e, n", "dense", "p_SQ"),
    ("dense_chi_diag", "control (iii): p_G0 = chi - e, dense e, n", "dense", "p_G0"),
]


# ----------------------------------------------------------------------------------------------------------------------------- zone integrals
def zone_point(Q, mh, kappa):
    """the zone integrals at (m, kappa) on strand5_eos_v4's quadrature Q with its measure (Quad.integrals' arrays): n = 2<1> (th_3 analytic),
    e = 2<eps>, A = 2<t^2/eps>, B = 2<t^2>, C = 2<eps t^2>; P = mu n - e, p_sin = A/3, p_SQ = P - (2/3)(mu B - C) (by parts),
    chi = mu (n - (2/3) B) (by parts), p_G0 = chi - e, W = (2/3) C."""
    W_al = Q.Wal(kappa)
    Th = np.arcsin(kappa * np.cos(Q.al))
    th3 = Th[:, None] * Q.u[None, :]
    t2 = (kappa * np.sin(Q.al))[:, None] ** 2 + np.sin(th3) ** 2
    eps = np.sqrt(mh * mh + t2)
    wu = Th[:, None] * Q.wu[None, :]

    def integ(F):
        return C16 * float(W_al @ (F * wu).sum(1))
    n = C16 * float(W_al @ Th)
    e, A, B, C = integ(eps), integ(t2 / eps), integ(t2), integ(eps * t2)
    mu = float(np.sqrt(mh * mh + kappa * kappa))
    P = mu * n - e
    chi = mu * (n - 2.0 / 3.0 * B)
    return {"kappa": float(kappa), "mu": mu, "n": n, "e": e, "P": P, "p_sin": A / 3.0, "p_SQ": P - 2.0 / 3.0 * (mu * B - C), "chi": chi,
            "p_G0": chi - e, "W": 2.0 / 3.0 * C}


def zone_direct_Q(mh, kappa, N2, N3):
    """the direct integrands with the cone angle ph resolved: (1/3) 2<sum_i Q_i^2/eps> and 2<sum_i Q_i^2>, Q_i^2 = sin^2 k_i cos^2 k_i
    (s^2 (1 - s^2) for s_1, s_2; sin^2 th_3 cos^2 th_3), the same coordinates and weights as strand5_eos_v4.Quad."""
    x, w = leggauss(N2)
    ph = np.pi / 4 * (x + 1); wph = np.pi / 4 * w
    u, wu = leggauss(N3)
    u = (u + 1) / 2; wu = wu / 2
    SQ = Q2 = 0.0
    for al, wal in zip(ph, wph):                                  # al uses the same nodes as ph (Quad: self.al = self.ph)
        sa, ca = np.sin(al), np.cos(al)
        s1 = kappa * sa * np.cos(ph); s2 = kappa * sa * np.sin(ph)
        wgt = (1 - s1 * s1) ** -0.5 * (1 - s2 * s2) ** -0.5 * wph * kappa ** 2 * sa * ca * wal
        Th = np.arcsin(kappa * ca)
        s3 = np.sin(Th * u); c3 = np.cos(Th * u)
        eps = np.sqrt(mh * mh + (kappa * sa) ** 2 + s3 * s3)
        q2 = (s1 * s1 * (1 - s1 * s1) + s2 * s2 * (1 - s2 * s2))[:, None] + (s3 * s3 * c3 * c3)[None, :]
        w3 = Th * wu
        SQ += float(wgt @ (q2 / eps[None, :]) @ w3)
        Q2 += float(wgt @ q2 @ w3)
    return C16 * SQ / 3.0, C16 * Q2


def build_table(job):
    """the dense zone table at one m a: production nodes, the quadrature convergence, the check against the eos_v4 table at its own nodes,
    the spline interpolation check at midpoints, the direct S^(Q) and the finite-difference chi at a subset of nodes."""
    mh, eos_path = job
    t0 = time.time()
    Qs = {k: SE.Quad(*v) for k, v in LEVELS.items()}
    prod = [zone_point(Qs["production"], mh, mh * z) for z in ZS]
    tab = {k: np.array([r[k] for r in prod]) for k in ("kappa", "mu") + KEYS + ("chi",)}
    conv_keys = ("n", "e", "P", "p_sin", "p_SQ", "chi", "W")
    coarse = [zone_point(Qs["coarse"], mh, mh * z) for z in ZS]
    sub = list(range(0, len(ZS), 10)) + [len(ZS) - 1]
    fine = {i: zone_point(Qs["fine"], mh, mh * ZS[i]) for i in sub}
    conv = {"coarse_vs_production": {k: float(max(abs(c[k] / p[k] - 1) for c, p in zip(coarse, prod))) for k in conv_keys},
            "fine_vs_production_subset": {k: float(max(abs(fine[i][k] / prod[i][k] - 1) for i in sub)) for k in conv_keys},
            "fine_nodes_z": [float(ZS[i]) for i in sub]}
    # the eos_v4 table at its own nodes (same quadrature level): n vs rho, e vs e_hat, P vs P_hat, p_sin vs -T_hat/3
    t = np.load(eos_path)
    sel = np.abs(t["m_hat"] - mh) < 1e-12
    tchk = {"n_vs_rho": 0.0, "e_vs_e_hat": 0.0, "P_vs_P_hat": 0.0, "p_sin_vs_minus_T_hat_over_3": 0.0, "nodes": 0}
    for k_, rho, eh, Ph, Th in zip(t["kappa"][sel], t["rho"][sel], t["e_hat"][sel], t["P_hat"][sel], t["T_hat"][sel]):
        if k_ > mh * ZS[-1]:
            continue
        r = zone_point(Qs["production"], mh, float(k_))
        tchk["n_vs_rho"] = max(tchk["n_vs_rho"], abs(r["n"] / rho - 1))
        tchk["e_vs_e_hat"] = max(tchk["e_vs_e_hat"], abs(r["e"] / eh - 1))
        tchk["P_vs_P_hat"] = max(tchk["P_vs_P_hat"], abs(r["P"] / Ph - 1))
        tchk["p_sin_vs_minus_T_hat_over_3"] = max(tchk["p_sin_vs_minus_T_hat_over_3"], abs(r["p_sin"] / (-Th / 3.0) - 1))
        tchk["nodes"] += 1
    # the splines (kappa^p times splines of X/kappa^p) against direct values at midpoints
    spl = splines_from(tab)
    mids = 0.5 * (ZS[:-1] + ZS[1:])[::3]
    ichk = {k: 0.0 for k in KEYS}
    ichk_z01 = {k: 0.0 for k in KEYS}
    for z in mids:
        r = zone_point(Qs["production"], mh, mh * z)
        for k in KEYS:
            d = abs(float(spl[k](mh * z)) / r[k] - 1)
            ichk[k] = max(ichk[k], d)
            if z >= 0.1:
                ichk_z01[k] = max(ichk_z01[k], d)
    # S^(Q) by its direct integrand and chi = (1/3) d 2<|Q|^2>/dmu (central differences in kappa, Richardson) at a subset of nodes
    dchk = []
    for z in (0.1, 0.5, 1.0, 2.0, 3.0):
        k_ = mh * z
        r = zone_point(Qs["production"], mh, k_)
        SQd = {lv: zone_direct_Q(mh, k_, *lv)[0] for lv in ((200, 32), (400, 48))}

        def dQ2(h):
            return (zone_direct_Q(mh, k_ * (1 + h), 400, 48)[1] - zone_direct_Q(mh, k_ * (1 - h), 400, 48)[1]) / (2 * h * k_)
        d1, d2 = dQ2(1e-3), dQ2(5e-4)
        chi_fd = (4 * d2 - d1) / 3 * (r["mu"] / k_) / 3.0
        dchk.append({"z": z, "p_SQ_by_parts": r["p_SQ"], "p_SQ_direct_200_32": SQd[(200, 32)], "p_SQ_direct_400_48": SQd[(400, 48)],
                     "rel_direct_400_vs_parts": SQd[(400, 48)] / r["p_SQ"] - 1, "chi_by_parts": r["chi"], "chi_fd_direct": chi_fd,
                     "rel_chi_fd_vs_parts": chi_fd / r["chi"] - 1, "ward_e_plus_SQ_minus_chi_minus_W_over_W": (r["e"] + r["p_SQ"] - r["chi"] - r["W"]) / r["W"]})
    ratios = []
    for z in (0.1, 0.5, 0.83, 1.0, 2.0, 3.0):
        k_ = mh * z
        P = float(spl["P"](k_))
        ratios.append({"z": z, "p_sin_over_P": float(spl["p_sin"](k_)) / P, "p_SQ_over_P": float(spl["p_SQ"](k_)) / P,
                       "p_G0_over_P": float(spl["p_G0"](k_)) / P, "W_over_p_SQ": float(spl["W"](k_)) / float(spl["p_SQ"](k_))})
    return {"m": mh, "nodes": len(ZS), "z_range": [float(ZS[0]), float(ZS[-1])], "table": {k: v.tolist() for k, v in tab.items()},
            "convergence": conv, "eos_v4_check": tchk, "interp_check_max_rel": ichk, "interp_check_max_rel_z_ge_0.1": ichk_z01,
            "direct_checks": dchk, "ratios": ratios, "seconds": time.time() - t0}


POW = {"n": 3, "e": 3, "P": 5, "p_sin": 5, "p_SQ": 5, "p_G0": 5, "W": 5}


def splines_from(tab):
    """kappa -> quantity: kappa^p times a cubic spline of quantity/kappa^p (smooth, with a finite limit at kappa = 0: p = 3 for n and e,
    5 for the pressures and W), through the table's nodes (not-a-knot ends)."""
    k = np.asarray(tab["kappa"], float)
    out = {}
    for key in KEYS:
        s = CubicSpline(k, np.asarray(tab[key], float) / k ** POW[key])
        out[key] = (lambda s_, p_: (lambda kk: np.asarray(kk, float) ** p_ * s_(kk)))(s, POW[key])
    return out


def eos_functions(mh, tab):
    """mu -> quantity on the dense table (zero at and below mu = m; an error beyond the table's last node)."""
    spl = splines_from(tab)
    kmax = float(tab["kappa"][-1])

    def mk(key):
        sp_ = spl[key]

        def F(u):
            if u <= mh:
                return 0.0
            k = (u * u - mh * mh) ** 0.5
            if k > kmax * (1 + 1e-12):
                raise ValueError("mu %.6g beyond the dense table (kappa %.6g > %.6g)" % (u, k, kmax))
            return float(sp_(k))
        return F
    return {key: mk(key) for key in KEYS}, spl


# --------------------------------------------------------------------------------------------------------------------------- the OV families
def safe_member(m, g, eos3, z):
    try:
        return OV1.member(m, g, eos3[0], eos3[1], eos3[2], float(z))
    except Exception as ex:                                      # a shot that finds no surface is recorded, not skipped silently
        return {"z_c": float(z), "error": str(ex)[:200]}


def summarize(mem, m):
    out = {k: mem[k] for k in ("z_c", "mu_c", "M", "N_B", "R_surf", "R_999", "compactness_surf", "EF_over_m", "N_centre", "min_N_minus_absv")}
    out["kappa_c"] = m * mem["z_c"]
    out["mu_c_over_m"] = mem["mu_c"] / m
    out["theta_F_c"] = float(np.arcsin(min(1.0, out["kappa_c"])))
    return out


def maxima(m, g, eos3, zs):
    rows = [safe_member(m, g, eos3, z) for z in zs]
    ok = [r for r in rows if "M" in r]
    res = {"scan": [{"z_c": r["z_c"], "M": r["M"], "N_B": r["N_B"]} if "M" in r else {"z_c": r["z_c"], "error": r["error"]} for r in rows],
           "z_first_ok": ok[0]["z_c"] if ok else None}
    for key in ("M", "N_B"):
        vals = [r[key] for r in ok]
        i = int(np.argmax(vals))
        if 0 < i < len(ok) - 1:
            r = minimize_scalar(lambda z: -safe_member(m, g, eos3, z).get(key, -np.inf), bounds=(ok[i - 1]["z_c"], ok[i + 1]["z_c"]),
                                method="bounded", options={"xatol": 1e-8})
            mem = safe_member(m, g, eos3, float(r.x))
            res["max_" + key] = {"bracketed": True, "nfev": int(r.nfev), **summarize(mem, m)}
        else:
            res["max_" + key] = {"bracketed": False, "note": "the scan's largest %s is at its end (z_c %g)" % (key, ok[i]["z_c"]), **summarize(ok[i], m)}
    return res


def mass_at_NB(m, g, eos3, NB, z_lo, z_top):
    """the closure's member with the given N_B on its stable branch (z_c between the scan's most dilute member and the N_B maximum): brentq."""
    f = lambda z: safe_member(m, g, eos3, z)["N_B"] - NB
    flo, fhi = f(z_lo), f(z_top)
    if fhi < 0:
        return None, "above this closure's N_B maximum"
    if flo > 0:
        return None, "below the scan's most dilute member"
    z = brentq(f, z_lo, z_top, xtol=1e-13, rtol=1e-12, maxiter=200)
    return summarize(safe_member(m, g, eos3, z), m), None


def lattice_members(paths, NB_lo, NB_hi):
    out, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except ValueError:                                       # a record being rewritten by a running job: skipped, named in the md
            out.append({"N_B": float("nan"), "E": float("nan"), "min_N_minus_absv": None, "record": os.path.basename(p) + " (unreadable now)",
                        "script": None, "ell": None, "nsites": None})
            continue
        for q in d["members"]:
            if q.get("failed"):
                continue
            key = round(q["N_B"], 3)
            if key in seen or not (NB_lo <= q["N_B"] <= NB_hi):
                continue
            seen.add(key)
            out.append({"N_B": q["N_B"], "E": q["E"], "min_N_minus_absv": q.get("min_N_minus_absv"), "record": os.path.basename(p),
                        "script": d.get("script"), "ell": d["args"].get("ell"), "nsites": d["args"].get("nsites")})
    return sorted(out, key=lambda q: q["N_B"])


def run_pair(job):
    m, g, eos_path, ovpg_path, tabs, lat_paths, zs, win = job
    t0 = time.time()
    e2, P2, n2 = OV2.load_eos_consistent(eos_path, m)
    F, spl = eos_functions(m, tabs[m])
    eos = {"v2": (e2, P2, n2), "dense": (F["e"], F["P"], F["n"])}
    press = {"P_v2": P2, "P": F["P"], "p_sin": F["p_sin"], "p_SQ": F["p_SQ"], "p_G0": F["p_G0"]}
    out = {"m": m, "g": g, "closures": {}}
    # version 1 (the families' comparator): OVPG/ovpg_families.json's maximum, N_B at its z_c with version 1's table equation of state
    fam = [f_ for f_ in json.load(open(ovpg_path))["families"] if abs(f_["m"] - m) < 1e-12 and abs(f_["g"] - g) < 1e-12]
    if fam and fam[0].get("maximum"):
        e1, P1, n1 = OV1.load_eos(eos_path, m)
        z1 = fam[0]["maximum"]["z_c"]
        mem1 = OV1.member(m, g, e1, P1, n1, z1)
        out["version1"] = {"z_c": z1, "M_max": fam[0]["maximum"]["M_max"], "N_B_at_z_c": mem1["N_B"], "M_at_z_c": mem1["M"]}
    for key, label, src, pk in CLOSURES:
        e_, _, n_ = eos[src]
        eos3 = (e_, press[pk], n_)
        r = maxima(m, g, eos3, zs)
        r["label"] = label
        kc = r["max_M"]["kappa_c"]
        r["centre_ratios_at_M_max"] = {"p_sin_over_P": float(spl["p_sin"](kc) / spl["P"](kc)), "p_SQ_over_P": float(spl["p_SQ"](kc) / spl["P"](kc)),
                                       "p_G0_over_P": float(spl["p_G0"](kc) / spl["P"](kc)), "W_over_p_SQ": float(spl["W"](kc) / spl["p_SQ"](kc))}
        out["closures"][key] = r
    # the lattice members near the maximum against every closure at the same N_B
    lat = []
    if lat_paths:
        NBref = out["closures"]["v2_thermo"]["max_N_B"]["N_B"]
        for q in lattice_members(lat_paths, win[0] * NBref, win[1] * NBref):
            row = dict(q)
            if not np.isfinite(q["N_B"]):
                lat.append(row)
                continue
            for key, label, src, pk in CLOSURES:
                c = out["closures"][key]
                e_, _, n_ = eos[src]
                mem, note = mass_at_NB(m, g, (e_, press[pk], n_), q["N_B"], c["z_first_ok"], c["max_N_B"]["z_c"])
                row[key] = {"M": mem["M"], "rel": q["E"] / mem["M"] - 1, "z_c": mem["z_c"]} if mem else {"note": note}
            lat.append(row)
    out["lattice"] = lat
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------------------------------------------------------------------------- output
def fmt(x, f):
    return (f % x) if (x is not None and np.isfinite(x)) else "-"


def write_md(res, path):
    L = ["# engine_ovpg_v3: the continuum turning points with the lattice band's native kinetic stress (printed by engine_ovpg_v3.py)", "",
         "What replaces what: version 2's OV system (engine_ovpg_v2: e and n from the strand5_eos_v4 table's P spline, n = dP/dmu, e = mu n - P; "
         "the local filling mu = E_F exp(-Phi)) with only the pressure in Phi' = g (m + 4 pi r^3 p)/(r (r - 2 g m)) replaced by the native kinetic "
         "stress p_Q = (2/3) <sum_i q_i^2/eps> (<X> = int_BZ d^3k/(2 pi)^3 X 1_{eps < mu}, the factor 2 the spin states: the measure, zone and "
         "degeneracy of the table's n = 2<1> and e = 2<eps>); e(mu) and n(mu) unchanged. The spec as briefed takes q_i = sin k_i (p_sin = -T_0/3 of "
         "the table); the reading q_i = Q_i = sin(2 k_i)/2 (the model's momentum symbol, the review's S^(Q)) and the diagnostic p_G0 = chi - e (the "
         "b -> 0 limit of the radial pressure J/b - rho of the review's P12 two-row closure) are controls, as is every closure on the dense zone table. "
         "The review's numbers are web-derived ideas (REVIEWA_REVIEW_web40e), printed for comparison only.", ""]
    L += ["## 1. The zone tables (strand5_eos_v4.Quad; production (N2, N3) = (2000, 64); %d nodes z = kappa/m from %g to %g)" % (
        len(ZS), ZS[0], ZS[-1]), ""]
    L += ["| m a | quadrature: max rel diff (1000, 48) vs (2000, 64) over n, e, P, p_sin, p_SQ, chi, W | (3000, 96) vs (2000, 64), 19 nodes | "
          "the eos_v4 table at its own nodes: n/rho - 1, e/e_hat - 1, P/P_hat - 1, p_sin/(-T_hat/3) - 1 (nodes) | spline vs direct at midpoints, max rel (z >= 0.1) | seconds |",
          "|---|---|---|---|---|---|"]
    for T in res["tables"]:
        c = T["convergence"]
        L.append("| %g | %s | %s | %.1e, %.1e, %.1e, %.1e (%d) | %s | %.0f |" % (
            T["m"], ", ".join("%.1e" % c["coarse_vs_production"][k] for k in ("n", "e", "P", "p_sin", "p_SQ", "chi", "W")),
            ", ".join("%.1e" % c["fine_vs_production_subset"][k] for k in ("n", "e", "P", "p_sin", "p_SQ", "chi", "W")),
            T["eos_v4_check"]["n_vs_rho"], T["eos_v4_check"]["e_vs_e_hat"], T["eos_v4_check"]["P_vs_P_hat"],
            T["eos_v4_check"]["p_sin_vs_minus_T_hat_over_3"], T["eos_v4_check"]["nodes"],
            ", ".join("%s %.1e" % (k, T["interp_check_max_rel_z_ge_0.1"][k]) for k in KEYS), T["seconds"]))
    L += ["", "The direct integrands against the by-parts forms (S^(Q) with the cone angle resolved; chi = (1/3) d 2<|Q|^2>/dmu by central differences), "
          "and the Ward identity e + S^(Q) - chi = W = (2/3) 2<eps t^2>:", "",
          "| m a | z | p_SQ by parts | direct (200, 32) | direct (400, 48) | rel (400, 48) | chi by parts | chi by differences | rel | (e + S^Q - chi - W)/W |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for T in res["tables"]:
        for d in T["direct_checks"]:
            L.append("| %g | %g | %.10e | %.10e | %.10e | %.1e | %.10e | %.10e | %.1e | %.1e |" % (
                T["m"], d["z"], d["p_SQ_by_parts"], d["p_SQ_direct_200_32"], d["p_SQ_direct_400_48"], d["rel_direct_400_vs_parts"],
                d["chi_by_parts"], d["chi_fd_direct"], d["rel_chi_fd_vs_parts"], d["ward_e_plus_SQ_minus_chi_minus_W_over_W"]))
    L += ["", "The pressures against the thermodynamic one on the dense table (z = kappa/m; z near 0.83 is the centre at the thermodynamic maximum):", "",
          "| m a | z | p_sin/P | p_SQ/P | p_G0/P = (chi - e)/P | W/p_SQ |", "|---|---|---|---|---|---|"]
    for T in res["tables"]:
        for r in T["ratios"]:
            L.append("| %g | %g | %.6f | %.6f | %.6f | %.5f |" % (T["m"], r["z"], r["p_sin_over_P"], r["p_SQ_over_P"], r["p_G0_over_P"], r["W_over_p_SQ"]))
    rev = {(r["m"], r["g"]): r for r in REVIEW}
    L += ["", "## 2. The turning points: the spec (native stress q_i = sin k_i, version 2's e and n) beside version 2's thermodynamic values", "",
          "| m a | g | version 1 (families' comparator): N_B at its z_c, M_max | version 2 thermodynamic: N_B,max; M_max (z_c) | "
          "version 3, native q = sin k: N_B,max (z_c); M_max (z_c) | version 3's N_B,max/version 2's - 1; M_max ratio - 1 | "
          "version 3's centre at the M maximum: z_c, mu_c/m, kappa_c, theta_F | the review's numbers (web-derived) |",
          "|---|---|---|---|---|---|---|---|"]
    for R in res["pairs"]:
        v1 = R.get("version1") or {}
        a, b = R["closures"]["v2_thermo"], R["closures"]["v3_native_sin"]
        rv = rev.get((R["m"], R["g"]))
        L.append("| %g | %g | %s, %s | %.4f; %.6f (%.6f) | %.4f (%.6f); %.6f (%.6f) | %+.3e; %+.3e | %.5f, %.5f, %.6f, %.6f | %s |" % (
            R["m"], R["g"], fmt(v1.get("N_B_at_z_c"), "%.4f"), fmt(v1.get("M_max"), "%.6f"),
            a["max_N_B"]["N_B"], a["max_M"]["M"], a["max_M"]["z_c"], b["max_N_B"]["N_B"], b["max_N_B"]["z_c"], b["max_M"]["M"], b["max_M"]["z_c"],
            b["max_N_B"]["N_B"] / a["max_N_B"]["N_B"] - 1, b["max_M"]["M"] / a["max_M"]["M"] - 1,
            b["max_M"]["z_c"], b["max_M"]["mu_c_over_m"], b["max_M"]["kappa_c"], b["max_M"]["theta_F_c"],
            ("N_B %s%s (%s)" % (fmt(rv["N_B"], "%g"), (", E a %.3f" % rv["E"]) if rv["E"] else "", rv["where"])) if rv else "-"))
    L += ["", "Every closure (the controls included): N_B,max and M_max, the invariants N_B,max g^1.5 (m a)^3 and M_max g^1.5 (m a)^2, "
          "and the bracket flags (a maximum at the scan's end is printed as not bracketed):", "",
          "| m a | g | closure | N_B,max (z_c) | M_max (z_c) | N_B at the M maximum | N_B,max g^1.5 (m a)^3 | M_max g^1.5 (m a)^2 | "
          "2gM/R_surf at the M maximum | centre ratios at the M maximum: p_sin/P, p_SQ/P, p_G0/P, W/p_SQ | bracketed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for R in res["pairs"]:
        for key, label, _, _ in CLOSURES:
            c = R["closures"][key]
            cr = c["centre_ratios_at_M_max"]
            L.append("| %g | %g | %s | %.4f (%.6f) | %.6f (%.6f) | %.4f | %.6f | %.6f | %.4f | %.4f, %.4f, %.4f, %.4f | %s |" % (
                R["m"], R["g"], label, c["max_N_B"]["N_B"], c["max_N_B"]["z_c"], c["max_M"]["M"], c["max_M"]["z_c"], c["max_M"]["N_B"],
                c["max_N_B"]["N_B"] * R["g"] ** 1.5 * R["m"] ** 3, c["max_M"]["M"] * R["g"] ** 1.5 * R["m"] ** 2, c["max_M"]["compactness_surf"],
                cr["p_sin_over_P"], cr["p_SQ_over_P"], cr["p_G0_over_P"], cr["W_over_p_SQ"],
                "yes" if (c["max_N_B"]["bracketed"] and c["max_M"]["bracketed"]) else "NO"))
    L += ["", "## 3. The lattice members near each maximum read against every closure at the same N_B (E/E_cont - 1 on the closure's stable branch; "
          "'above' where N_B exceeds the closure's N_B,max)", ""]
    hdr = "| m a | g | record | N_B | E (lattice) | min(N - abs v) | " + " | ".join(k for k, _, _, _ in CLOSURES) + " |"
    L += [hdr, "|" + "---|" * (6 + len(CLOSURES))]
    for R in res["pairs"]:
        for q in R["lattice"]:
            if CLOSURES[0][0] not in q:
                L.append("| %g | %g | %s | - | - | - |%s" % (R["m"], R["g"], q["record"], " - |" * len(CLOSURES)))
                continue
            cells = []
            for key, _, _, _ in CLOSURES:
                c = q[key]
                cells.append(("%.6f (%+.2e)" % (c["M"], c["rel"])) if "M" in c else "above")
            L.append("| %g | %g | %s | %.3f | %.6f | %s | %s |" % (R["m"], R["g"], q["record"], q["N_B"], q["E"], fmt(q["min_N_minus_absv"], "%.5f"), " | ".join(cells)))
    L += ["", "- every number above is printed by engine_ovpg_v3.py from the zone quadrature, the OV integrations and the family records named; "
          "the review's numbers are typed constants quoted from REVIEWA_REVIEW_web40e.md (sections named), web-derived ideas, never sources.",
          "- runtime %.0f s." % res["seconds"]]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos", default="EOS/eos_v4_tables.npz")
    ap.add_argument("--ovpg", default="OVPG/ovpg_families.json")
    ap.add_argument("--grid", default="0.2:0.25,0.2:0.0625,0.1:0.25,0.1:0.0625,0.05:0.25,0.1:0.015625")
    ap.add_argument("--zcs", default=",".join("%.4g" % x for x in np.geomspace(0.05, 3.0, 48)), help="version 1's scan")
    ap.add_argument("--lattice", nargs="*", default=[], help='"m:g=json[,json...]": the family records whose members near the maximum are read')
    ap.add_argument("--window", default="0.9,1.15", help="the lattice members read: N_B in this range times version 2's N_B,max")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    t0 = time.time()
    pairs = [tuple(float(x) for x in s.split(":")) for s in a.grid.split(",")]
    ms = sorted(set(p[0] for p in pairs), reverse=True)
    lat = {}
    for s in a.lattice:
        k, v = s.split("=", 1)
        lat[tuple(float(x) for x in k.split(":"))] = v.split(",")
    with mp.get_context("fork").Pool(min(a.procs, len(ms))) as pool:
        tables = pool.map(build_table, [(mh, a.eos) for mh in ms])
    for T in tables:
        print("table m %g: %d nodes, %.0f s; convergence coarse %s; eos_v4 check %s" % (
            T["m"], T["nodes"], T["seconds"], T["convergence"]["coarse_vs_production"], T["eos_v4_check"]), flush=True)
    tabs = {T["m"]: T["table"] for T in tables}
    zs = [float(x) for x in a.zcs.split(",")]
    win = [float(x) for x in a.window.split(",")]
    jobs = [(m, g, a.eos, a.ovpg, tabs, lat.get((m, g)), zs, win) for m, g in pairs]
    with mp.get_context("fork").Pool(min(a.procs, len(jobs))) as pool:
        out = pool.map(run_pair, jobs)
    for R in out:
        a2, b3 = R["closures"]["v2_thermo"], R["closures"]["v3_native_sin"]
        print("m %g g %g: v2 N_B,max %.4f M_max %.6f | v3 (q = sin k) N_B,max %.4f M_max %.6f | dense thermo %.4f, dense chi %.4f (%.0f s)" % (
            R["m"], R["g"], a2["max_N_B"]["N_B"], a2["max_M"]["M"], b3["max_N_B"]["N_B"], b3["max_M"]["M"],
            R["closures"]["dense_thermo"]["max_N_B"]["N_B"], R["closures"]["dense_chi_diag"]["max_N_B"]["N_B"], R["seconds"]), flush=True)
    res = {"script": "engine_ovpg_v3.py", "args": vars(a), "review_numbers": REVIEW, "closures": [list(c) for c in CLOSURES],
           "tables": tables, "pairs": out, "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    write_md(res, a.md)
    print("done %.0f s" % res["seconds"], flush=True)


if __name__ == "__main__":
    main()
