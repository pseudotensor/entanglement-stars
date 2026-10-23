#!/usr/bin/env python
"""C_readout_selftest_v2.py (the busi lane): the self-test of C_readout_v2.py on manufactured records. No real file is read.
1. The formula's integral: the shell integral I_L(R) = 4 pi int r^2 N (rho_L + u J_L) dr against a closed form (a grid-exact case and a
   continuum case with N = N0 (1 + d r^2), u = -c r, rho_L = rho0, J_L = j0 r), and the node trapezoid against the same closed form.
2. The root displacement delta R = 2 G delta M(R) / D_R against the exact root of r - 2 G (M_H + delta M) on an analytic M_H (two roots),
   the error scaling with the perturbation's square.
3. The smoothing estimate's kernel preserves the total.
4. A manufactured two-run history (a collapsing analytic mass function on 256 sites, roots from the fourth record, a light source of known
   densities, the json totals in the unweighted convention): compute_b reads off 'unweighted', s = +1, the shell normalization, reproduces
   I_L against a fine quadrature and delta R_L against the analytic coefficient; four manufactured C's give 'no conflict', 'CONFLICT (sign)',
   'CONFLICT (order of magnitude)' and the formation-shift sign conflict; compute_a's difference series equal the manufactured displacements.
5. A missing key stops loudly: eps_L2 absent from the fields mapping and from a written npz, M absent from the s11 mapping, trapping_zeros
   absent from a record json, a sourcing channel absent from kappas_light.
Exit code 0 when every check passes, 1 otherwise; every check prints PASS or FAIL with its numbers."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import math
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import C_readout_v2 as V2  # noqa: E402
from C_readout_v1 import MissingKey, Inconsistent, ReadoutError, Series  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, (": " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


# ----------------------------------------------------------------------------------------------------------------- 1. the integral
def test_integral():
    # grid-exact: constant shells, R at a shell edge
    R = np.arange(1.0, 51.0)
    a, edges = V2.shell_edges(R)
    N = np.full(R.size, 0.8)
    f = np.full(R.size, 0.25)
    c0 = V2.cum0(N * f)
    got = float(V2.enclosed(edges, c0, 20.5))
    check("integral, grid-exact (20 shells of 0.8 x 0.25)", abs(got - 20 * 0.2) < 1e-12, "%.15g against 4" % got)
    # continuum closed form on a fine grid
    aa, R0, N0, d, c, rho0, j0 = 0.05, 10.0, 0.9, 1e-3, 0.02, 1.0e-3, 5.0e-4
    R = np.arange(aa / 2, 30.0, aa)
    a, edges = V2.shell_edges(R)
    Rend = R0 + 0.0
    N = N0 * (1.0 + d * R ** 2)
    u = -c * R
    rho = np.full(R.size, rho0)
    J = j0 * R
    shell_e = 4.0 * math.pi * R ** 2 * rho * a
    shell_j = 4.0 * math.pi * R ** 2 * J * a
    c0 = V2.cum0(N * (shell_e + u * shell_j))
    got = float(V2.enclosed(edges, c0, Rend))
    exact = 4.0 * math.pi * N0 * (rho0 * Rend ** 3 / 3.0 + (d * rho0 - c * j0) * Rend ** 5 / 5.0 - d * c * j0 * Rend ** 7 / 7.0)
    check("integral, continuum closed form (shell route)", abs(got / exact - 1.0) < 2e-3, "%.10g against %.10g (rel %.2e)" % (got, exact, got / exact - 1.0))
    rn = np.arange(0.0, 30.0 + 1e-9, 0.5)
    fn = 4.0 * math.pi * rn ** 2 * N0 * (1.0 + d * rn ** 2) * (rho0 - c * rn * j0 * rn)
    cn = V2.cumtrapz0(rn, fn)
    gotn = float(np.interp(Rend, rn, cn))
    check("integral, continuum closed form (node trapezoid, h = 0.5)", abs(gotn / exact - 1.0) < 5e-3, "%.10g against %.10g (rel %.2e)" % (gotn, exact, gotn / exact - 1.0))   # busi fix: the node trapezoid's O(h^2) error at h 0.5 on this integrand is 3.3e-3 (the shell route on the same integral meets 2e-3); 5e-3 states that accuracy


# --------------------------------------------------------------------------------------------------------- 2. the root displacement
def bisect(f, lo, hi, it=200):
    flo = f(lo)
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


def test_root_displacement():
    g, A, b, cL = 0.25, 0.012, 20.0, 10.0

    def MH(r):
        x = r / b
        return A * r ** 3 / (1.0 + x ** 3)

    def MHp(r):
        x = r / b
        return 3.0 * A * r ** 2 / (1.0 + x ** 3) ** 2

    def dM(r, eta):
        return eta * r ** 3 / (1.0 + (r / cL) ** 3)
    rp = 2.0 ** (1.0 / 3.0) * b
    roots0 = (bisect(lambda r: r - 2 * g * MH(r), 1e-3, rp), bisect(lambda r: r - 2 * g * MH(r), rp, 10 * b))
    ok_all = True
    for label, R0 in (("inner", roots0[0]), ("outer", roots0[1])):
        D = 1.0 - 2.0 * g * MHp(R0)
        errs = []
        for eta in (1e-4, 1e-5):
            lo, hi = (1e-3, rp) if label == "inner" else (rp, 10 * b)
            R1 = bisect(lambda r: r - 2 * g * (MH(r) + dM(r, eta)), lo, hi)
            pred = 2.0 * g * dM(R0, eta) / D
            errs.append(abs((R1 - R0) - pred) / abs(pred))
        ok = errs[1] < 2e-2 and errs[1] < 0.3 * errs[0]
        ok_all &= ok
        check("root displacement delta R = 2 G delta M / D_R at the %s root (R0 %.4f, D_R %.4f)" % (label, R0, D), ok, "relative errors %.2e (eta 1e-4), %.2e (eta 1e-5)" % tuple(errs))
    check("D_R > 0 at the outer root and < 0 at the inner root", (1.0 - 2.0 * g * MHp(roots0[1])) > 0 > (1.0 - 2.0 * g * MHp(roots0[0])))
    return ok_all


# ---------------------------------------------------------------------------------------------------------------- 3. the kernel
def test_kernel():
    R = np.arange(0.0, 300.0)
    K = V2.gaussian_shell_kernel(R, 4.0)
    cs = K.sum(axis=0)
    inner = R < 300.0 - 30.0
    check("kernel estimate preserves the total (column sums 1 away from the end)", float(np.max(np.abs(cs[inner] - 1.0))) < 1e-6, "max |sum - 1| %.2e" % float(np.max(np.abs(cs[inner] - 1.0))))
    e = np.exp(-(R - 100.0) ** 2 / 50.0)
    check("kernel estimate: a bump keeps its total", abs((K @ e).sum() / e.sum() - 1.0) < 1e-6)


# --------------------------------------------------------------------------------------------- 4. the manufactured two-run history
class H:
    nsites, nrec, b, g, cJ, sigL, rho0, tau, rs, ell = 256, 9, 20.0, 0.25, 0.3, 10.0, 2.0e-7, 2.0, 32.0, 4.0
    wts = {1: 16.0, 2: 32.0, 3: 48.0}
    shape = {1: 1.0, 2: 0.5, 3: 0.25}   # the unweighted channel densities as multiples of rho_L; J = 0.5 eps per channel


def hist_A(k):
    return 0.006 + 0.0012 * k


def MH(r, k):
    x = r / H.b
    return hist_A(k) * r ** 3 / (1.0 + x ** 3)


def MHp(r, k):
    x = r / H.b
    return 3.0 * hist_A(k) * r ** 2 / (1.0 + x ** 3) ** 2


def rho_over(r, k):
    """M'/(4 pi r^2) = 3 A / (4 pi (1 + x^3)^2), finite at r = 0."""
    x = r / H.b
    return 3.0 * hist_A(k) / (4.0 * math.pi * (1.0 + x ** 3) ** 2)


def ufun(r, k):
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(r > 0, 2.0 * H.g * hist_A(k) * r ** 2 / (1.0 + (r / H.b) ** 3), 0.0)
    return -np.sqrt(q)


def Nfun(r):
    return 1.0 - 0.3 * np.exp(-(np.asarray(r, dtype=float) / 30.0) ** 2)


def rhoL(r):
    return H.rho0 * np.exp(-np.asarray(r, dtype=float) ** 2 / (2.0 * H.sigL ** 2))


def roots_of(k):
    rp = 2.0 ** (1.0 / 3.0) * H.b
    if rp - 2 * H.g * MH(rp, k) >= 0:
        return None
    return (bisect(lambda r: r - 2 * H.g * MH(r, k), 1e-3, rp), bisect(lambda r: r - 2 * H.g * MH(r, k), rp, 10 * H.b))


def wL():
    """The weighted sourcing light density as a multiple of rho_L: eps and J."""
    we = sum(H.wts[c] * H.shape[c] for c in (1, 2))
    return we, 0.5 * we


def I_L_fine(R, k):
    r = np.linspace(0.0, R, 40001)
    we, wj = wL()
    f = 4.0 * math.pi * r ** 2 * Nfun(r) * (we * rhoL(r) + ufun(r, k) * wj * rhoL(r))
    return float(np.trapz(f, r))


def analytic_dR(R, k):
    D = 1.0 - 2.0 * H.g * MHp(R, k)
    return 2.0 * H.g * (I_L_fine(R, k) / float(Nfun(R))) / D, D


def make_bundle(coupling, root_shift=None, drop_first_root=False):
    """The run json (dict), the fields mapping and the s11 mapping of one manufactured history. root_shift(k, label, R) -> the displacement
    applied to C's roots; drop_first_root removes the roots of C's first trapped record (a later formation)."""
    R = np.arange(H.nsites, dtype=float)
    rn = np.arange(0.0, H.nsites, 4.0)
    nk = H.nrec
    Nf, vf, eps_tot, J_tot = [], [], [], []
    epsL = {c: [] for c in (1, 2, 3)}
    JL = {c: [] for c in (1, 2, 3)}
    s11 = {key: [] for key in ("rho_ell", "J_ell", "N", "v", "M", "rho_ell_ball", "J_ell_ball", "rho_ell_pairs", "J_ell_pairs", "rho_ell_light_fed", "J_ell_light_fed")}
    recs = []
    first_root_k = None
    for k in range(nk):
        u = ufun(R, k)
        NN = Nfun(R)
        vv = NN * u
        eH = (1.0 - H.cJ * u) * MHp(R, k)
        jH = H.cJ * MHp(R, k)
        Nf.append(NN)
        vf.append(vv)
        eps_tot.append(eH)
        J_tot.append(jH)
        base = 4.0 * math.pi * R ** 2 * rhoL(R)
        for c in (1, 2, 3):
            epsL[c].append(H.shape[c] * base)
            JL[c].append(0.5 * H.shape[c] * base)
        un = ufun(rn, k)
        we, wj = wL()
        s11["rho_ell"].append((1.0 - H.cJ * un) * rho_over(rn, k))
        s11["J_ell"].append(H.cJ * rho_over(rn, k))
        s11["N"].append(Nfun(rn))
        s11["v"].append(Nfun(rn) * un)
        s11["M"].append(MH(rn, k))
        s11["rho_ell_ball"].append((1.0 - H.cJ * un) * rho_over(rn, k))
        s11["J_ell_ball"].append(H.cJ * rho_over(rn, k))
        s11["rho_ell_pairs"].append(np.zeros(rn.size))
        s11["J_ell_pairs"].append(np.zeros(rn.size))
        s11["rho_ell_light_fed"].append(we * rhoL(rn) if coupling == "off" else we * rhoL(rn))
        s11["J_ell_light_fed"].append(wj * rhoL(rn))
        rts = roots_of(k)
        if rts is not None and first_root_k is None:
            first_root_k = k
        drop = drop_first_root and rts is not None and k == first_root_k
        if rts is not None and not drop:
            Ri, Ro = rts
            if root_shift is not None:
                Ri = Ri + root_shift(k, "inner", Ri)
                Ro = Ro + root_shift(k, "outer", Ro)
            D = 1.0 - 2.0 * H.g * MHp(rts[1], k)
            kappa = float(Nfun(rts[1])) * D / (2.0 * rts[1])
            zeros = [Ri, Ro]
            R_t, r_in, kap = Ro, Ri, kappa
            mnv = float(np.min(NN - np.abs(vv)))
        else:
            zeros, R_t, r_in, kap = [], None, None, None
            mnv = float(np.min(NN - np.abs(vv)))
        E = 60.0 - 0.01 * k - (1e-6 * k if coupling == "on" else 0.0)
        chans = {}
        for c in (1, 2, 3):
            es, js = float(np.sum(epsL[c][-1])), float(np.sum(JL[c][-1]))
            chans[str(c)] = {"E_inv": 1e-5 * k * H.shape[c], "E_inv_weighted": H.wts[c] * 1e-5 * k * H.shape[c], "I_E_cut_weighted": 1e-10 * c, "eps_sum": es, "J_sum": js,
                             "eps_sum_weighted": H.wts[c] * es, "w": H.wts[c], "sourcing": c != 3, "fallback": False, "cut_crossing": False}
        recs.append({"T": H.tau * k, "T_over_rs": H.tau * k / H.rs, "E": E, "E_over_E0": E / 60.0, "R_t": R_t, "r_in": r_in, "trapping_zeros": zeros, "min_N_minus_absv": mnv,
                     "r_min_N_minus_absv": float(R[int(np.argmin(NN - np.abs(vv)))]), "M1_I_E": 1e-9 * k, "H_in": 50.0 - 0.1 * k, "E_dust_shells": float(np.sum(eH)), "E_pairs_shells": 0.0,
                     "kappa_T": kap, "converged": True, "iterations": 3, "r_collar": 255.0, "R_t_over_2g_MS": (1.0 if R_t is not None else None),
                     "s11": {"M_min_outside_cell": float(MH(4.25, k)), "r_at_M_min": 4.25, "negative_points_outside_cell": 0, "light_fed": coupling == "on", "light_source_integral": 0.0,
                             "ball_source_integral": float(np.sum(eH)), "pairs_source_integral": 0.0, "compactness_max_2gM_over_r": float(np.max(2 * H.g * MH(rn[1:], k) / rn[1:]))},
                     "light_v36": {"E_inv_heavy": 40.0 + k, "E_inv_light": 1e-4 * k, "E_inv_light_sourcing": 0.9e-4 * k, "I_E_light": 1e-10, "I_E_light_sourcing": 0.5e-10, "H_in_light": 1e-4 * k,
                                   "coupling": coupling, "certified_on_record": True, "fed": coupling == "on", "E_inv_light_sourcing_over_E": 0.9e-4 * k / E, "channels": chans}})
    run = {"script": "manufactured", "args": {"tau": H.tau, "rs": H.rs, "ell": H.ell, "g": H.g, "light_kappas": "1,2", "light_guard_kappas": "3", "light_coupling": coupling, "nsites": H.nsites,
                                               "tag": "manufactured_" + coupling, "mix": "newton"}, "head": {"E0": 60.0}, "records": recs}
    fields = {"R": R, "T": np.array([r["T"] for r in recs]), "N": np.array(Nf), "v": np.array(vf), "eps_tot": np.array(eps_tot), "J_tot": np.array(J_tot),
              "eps_light_fed": np.zeros((nk, H.nsites)), "J_light_fed": np.zeros((nk, H.nsites)), "kappas_light": np.array([1, 2, 3]), "weights_light": np.array([16.0, 32.0, 48.0]),
              "light_sourcing": np.array([1, 1, 0]), "light_meta": np.array("manufactured: unweighted per-channel shell densities"), "g": np.array(H.g), "ell": np.array(4), "tau": np.array(H.tau)}
    for c in (1, 2, 3):
        fields["eps_L%d" % c] = np.array(epsL[c])
        fields["J_L%d" % c] = np.array(JL[c])
    s11m = {"T": np.array([r["T"] for r in recs]), "r_nodes": np.tile(rn, (nk, 1))}
    for key, val in s11.items():
        s11m[key] = np.array(val)
    return run, fields, s11m


def test_end_to_end():
    runK, FK, ZKm = make_bundle("off")
    SK, mK = V2.extract_min(runK, "control")
    FKd = V2.fields_from_mapping(FK, list(FK.keys()), "control_fields")
    ZK = V2.s11_from_mapping(ZKm, list(ZKm.keys()), "control_s11")
    shifts = {}

    def shift_exact(k, label, R0):
        d, _ = analytic_dR(R0, k)
        shifts[(k, label)] = d
        return d
    cases = [("no conflict", lambda k, lb, R0: shift_exact(k, lb, R0), False, "no conflict"),
             ("sign", lambda k, lb, R0: -analytic_dR(R0, k)[0], False, "CONFLICT (sign)"),
             ("order of magnitude", lambda k, lb, R0: 30.0 * analytic_dR(R0, k)[0], False, "CONFLICT (order of magnitude)"),
             ("formation shift", lambda k, lb, R0: analytic_dR(R0, k)[0], True, "CONFLICT (sign)"),
             ("below floor", lambda k, lb, R0: 1e-7, False, "undecided")]
    for name, shift, drop, expect in cases:
        runC, FC, ZCm = make_bundle("on", root_shift=shift, drop_first_root=drop)
        SC, mC = V2.extract_min(runC, "run")
        ZC = V2.s11_from_mapping(ZCm, list(ZCm.keys()), "s11")
        iK = V2.align(SC.d["T_over_rs"], SK.d["T_over_rs"])
        pairs = [(k, j) for k, j in enumerate(iK) if j is not None]
        rev = {j: k for k, j in pairs}
        D, F, Asum = V2.compute_a(SC, mC, SK, mK, iK, rev, pairs, None)
        try:
            B = V2.compute_b(SC, mC, SK, mK, iK, rev, pairs, FKd, ZK, ZC, {"sigma_smooth": None, "eps_M": None})
        except ReadoutError as e:
            check("end to end (%s): compute_b runs" % name, False, "%s: %s" % (type(e).__name__, e))
            continue
        if name == "no conflict":
            conv = B["conventions"]
            check("conventions: the light arrays read off as unweighted", all(conv["L%d" % c]["stored_as"] == "unweighted" for c in (1, 2)))
            check("conventions: the heavy densities read off as shell quantities of eps_tot - eps_light_fed", conv["heavy_site_densities"]["normalization"] == "shell" and conv["heavy_site_densities"]["chosen"].startswith("eps_tot - eps_light_fed"))
            check("conventions: the row's sign s = +1 from the root condition and the node route", B["sign_s"] == 1 and conv["row_sign_s"]["from_node_route_control"] == "s=+1,volume",
                  "root condition median deviations %s" % {s: conv["row_sign_s"]["root_condition"][s]["median_abs_dev"] for s in (1, -1)})
            check("route K populated on the manufactured s11", B["route_K_populated"])
            outer = [r for r in B["roots"] if r["root"] == "outer"]
            check("outer roots found on the trapped records", len(outer) == sum(1 for x in SK.d["R_t"] if x is not None) and len(outer) > 0, "%d" % len(outer))
            worst_I, worst_K, worst_D, worst_u, worst_r = 0.0, 0.0, 0.0, 0.0, 0.0
            for r in outer:
                If = I_L_fine(r["R"], r["j_K"])
                worst_I = max(worst_I, abs(r["I_L_sites"] / If - 1.0))
                worst_K = max(worst_K, abs(r["I_L_nodes"] / If - 1.0))
                worst_D = max(worst_D, abs(r["D_R_rho_over_kappa"] - 1.0))
                worst_u = max(worst_u, abs(r["u_plus_1"]))
                worst_r = max(worst_r, abs(r["ratio_measured_over_dR_L"] - 1.0))
            check("I_L (site route) against the fine quadrature on every outer root", worst_I < 5e-3, "worst rel %.2e" % worst_I)
            check("I_L (node route K) against the fine quadrature", worst_K < 3e-2, "worst rel %.2e" % worst_K)
            check("CF4: D_R from the heavy densities equals D_R from kappa_T", worst_D < 2e-2, "worst |ratio - 1| %.2e" % worst_D)
            check("u(R) + 1 = 0 at the dense roots", worst_u < 1e-3, "worst %.2e" % worst_u)
            check("delta R_L reproduces the manufactured analytic displacement (ratio 1)", worst_r < 1e-2, "worst |ratio - 1| %.2e" % worst_r)
            inner = [r for r in B["roots"] if r["root"] == "inner"]
            worst_i = max(abs(r["ratio_measured_over_dR_L"] - 1.0) for r in inner) if inner else None
            check("the inner root: delta R_L (kappa from the sites, D_R < 0) reproduces the displacement", worst_i is not None and worst_i < 3e-2, "worst |ratio - 1| %s" % worst_i)
            m = B["margin_control"]
            check("margin on the control: positive on every record (M_L > 0 here)", all(x > 0 for x in m["min_margin"]) and all(x > 0 for x in m["min_margin_B"]))
            mc = B["margin_C_actual"]
            check("C's actual-field split: the components reproduce M at the nodes (trapezoid)", max(mc["components_vs_M_rel_residual"]) < 2e-2, "worst %.2e" % max(mc["components_vs_M_rel_residual"]))
            dR = [x for x in D["d.R_t"] if x is not None]
            man = [shifts[(k, "outer")] for k in range(mK["n"]) if (k, "outer") in shifts]
            check("(a): the difference series d.R_t equals the manufactured displacement", len(dR) == len(man) and all(abs(x - y) < 1e-12 for x, y in zip(dR, man)))
            check("(a): the formation clocks coincide, dT_records 0", F["dT_records"] == 0 and F["C"]["k"] == F["control"]["j"])
            check("(a): dE/E is the manufactured -1e-6 k / E at 20 x tau", abs(D["rel.E"][5] - (-5e-6 / SK.d["E"][5])) < 1e-15)
        kfC = F["C"]["k"] if F["C"] else None
        jfK = F["control"]["j"] if F["control"] else None
        V = V2.judge(B, kfC, jfK, rev, SC.d["T_over_rs"], SK.d["T_over_rs"], V2.FLOOR_DR_T, None, H.tau / H.rs)
        check("verdict (%s): %s" % (name, expect), V["verdict"].startswith(expect), V["verdict"])
        if drop:
            check("(a): the formation shift is +1 record in the manufactured later-forming C", F["dT_records"] == 1)


# ---------------------------------------------------------------------------------------------------------- 5. missing keys
def test_missing_keys():
    runK, FK, ZKm = make_bundle("off")
    FK2 = dict(FK)
    del FK2["eps_L2"]
    try:
        V2.fields_from_mapping(FK2, list(FK2.keys()), "control_fields")
        check("missing key: eps_L2 absent from the fields mapping stops with MissingKey", False, "no exception")
    except MissingKey as e:
        check("missing key: eps_L2 absent from the fields mapping stops with MissingKey", "eps_L2" in str(e), str(e)[:120])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "fields.npz")
        np.savez(p, **FK2)
        try:
            V2.load_fields(p, "control_fields_npz")
            check("missing key: eps_L2 absent from a written npz stops with MissingKey", False, "no exception")
        except MissingKey as e:
            check("missing key: eps_L2 absent from a written npz stops with MissingKey", "eps_L2" in str(e))
        p2 = os.path.join(td, "fields_ok.npz")
        np.savez(p2, **FK)
        try:
            F3 = V2.load_fields(p2, "control_fields_npz")
            check("the complete written npz loads (light_meta read)", F3["light_meta_str"] is not None and F3["N"].shape == (H.nrec, H.nsites))
        except ReadoutError as e:
            check("the complete written npz loads", False, str(e))
    Z2 = dict(ZKm)
    del Z2["M"]
    try:
        V2.s11_from_mapping(Z2, list(Z2.keys()), "control_s11")
        check("missing key: M absent from the s11 mapping stops with MissingKey", False, "no exception")
    except MissingKey as e:
        check("missing key: M absent from the s11 mapping stops with MissingKey", "'M'" in str(e))
    import copy
    r2 = copy.deepcopy(runK)
    del r2["records"][3]["trapping_zeros"]
    try:
        V2.extract_min(r2, "control")
        check("missing key: trapping_zeros absent from a record stops with MissingKey", False, "no exception")
    except MissingKey as e:
        check("missing key: trapping_zeros absent from a record stops with MissingKey", "trapping_zeros" in str(e) and "records[3]" in str(e))
    r3 = copy.deepcopy(runK)
    r3["args"]["light_kappas"] = "1,2,4"
    try:
        V2.extract_min(r3, "control")
        check("missing key: a sourcing channel absent from channels stops with MissingKey", False, "no exception")
    except MissingKey as e:
        check("missing key: a sourcing channel absent from channels stops with MissingKey", "channels" in str(e) and "4" in str(e), str(e)[:160])   # busi fix: the stop names the channels mapping and the key 4, not the joined string "channels.4"
    # the weighting hypothesis the other way round: arrays stored weighted
    FK4 = dict(FK)
    for c in (1, 2, 3):
        FK4["eps_L%d" % c] = FK["eps_L%d" % c] * H.wts[c]
        FK4["J_L%d" % c] = FK["J_L%d" % c] * H.wts[c]
    runC, FC, ZCm = make_bundle("on", root_shift=lambda k, lb, R0: analytic_dR(R0, k)[0])
    SC, mC = V2.extract_min(runC, "run")
    SK, mK = V2.extract_min(runK, "control")
    iK = V2.align(SC.d["T_over_rs"], SK.d["T_over_rs"])
    pairs = [(k, j) for k, j in enumerate(iK) if j is not None]
    rev = {j: k for k, j in pairs}
    B = V2.compute_b(SC, mC, SK, mK, iK, rev, pairs, V2.fields_from_mapping(FK4, list(FK4.keys()), "f"), V2.s11_from_mapping(ZKm, list(ZKm.keys()), "z"),
                     V2.s11_from_mapping(ZCm, list(ZCm.keys()), "zc"), {"sigma_smooth": None, "eps_M": None})
    outer = [r for r in B["roots"] if r["root"] == "outer"]
    check("weighting read off as 'weighted' when the arrays are stored weighted, the same delta R_L", B["conventions"]["L1"]["stored_as"] == "weighted"
          and max(abs(r["ratio_measured_over_dR_L"] - 1.0) for r in outer) < 1e-2)
    FK5 = dict(FK)
    FK5["eps_L1"] = FK["eps_L1"] * 3.0
    try:
        V2.compute_b(SC, mC, SK, mK, iK, rev, pairs, V2.fields_from_mapping(FK5, list(FK5.keys()), "f"), V2.s11_from_mapping(ZKm, list(ZKm.keys()), "z"), None, {"sigma_smooth": None, "eps_M": None})
        check("a stored array that fits neither weighting stops with Inconsistent", False, "no exception")
    except Inconsistent as e:
        check("a stored array that fits neither weighting stops with Inconsistent", "neither" in str(e))


def main():
    print("C_readout_selftest_v2: manufactured records only; C_readout_v2 %s" % V2.VERSION)
    test_integral()
    test_root_displacement()
    test_kernel()
    test_end_to_end()
    test_missing_keys()
    if FAILS:
        print("SELFTEST FAIL: %d check(s): %s" % (len(FAILS), "; ".join(FAILS)))
        return 1
    print("SELFTEST PASS: every check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
